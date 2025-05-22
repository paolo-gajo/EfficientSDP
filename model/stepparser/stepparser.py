import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.encoder import Encoder
from model.parser import SimpleParser
from model.tagger import Tagger
from model.decoder import GraphDecoder
import numpy as np
import warnings
from debug.model_debugging import nan_checker, check_param_norm, check_grad_norm

class StepParser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.tagger = Tagger(self.config)
        self.parser = SimpleParser.get_model(self.config) # base setting
        self.decoder = GraphDecoder(config=config,
                                    tag_representation_dim=self.parser.tag_representation_dim,
                                    n_edge_labels = self.parser.n_edge_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.mode = "train"

    def forward(self, model_input):
        # encoder
        model_input = {k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input.items()}
        model_input = {k: [el.to(self.config["device"]) for el in v] if (isinstance(v, list) and isinstance(v[0], torch.Tensor)) else v
            for k, v in model_input.items()}
        encoder_input = {k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input["encoded_input"].items()}
        encoder_output = self.encoder(encoder_input)

        # determine which representation mode to use
        if self.config["rep_mode"] == "words":
            tagger_labels = model_input["pos_tags"]
            mask = encoder_input["words_mask_custom"]
            head_indices = model_input["head_indices"]
            head_tags = model_input["head_tags"]
        elif self.config["rep_mode"] == "tokens":
            tagger_labels = model_input["pos_tags_tokens"]
            mask = encoder_input["attention_mask"]
            head_indices = model_input["head_indices_tokens"]
            head_tags = model_input["head_tags_tokens"]

        # use appropriate tags based on mode
        if self.mode in ["train", "validation"]:
            head_tags, head_indices = head_tags, head_indices
        elif self.mode == "test":
            head_tags, head_indices = None, None

        # tagging
        tagger_output = self.tagger(encoder_output, mask=mask, labels=tagger_labels)

        # convert preds/golds to one-hot
        pos_tags_pred_one_hot = self.tagger.get_predicted_classes_as_one_hot(tagger_output.logits)
        pos_tags_gt = torch.nn.functional.one_hot(tagger_labels, num_classes=self.config["n_tags"])

        # use predicted or ground truth tags (oracle)
        pos_tags_parser_one_hot = pos_tags_pred_one_hot if self.config["use_pred_tags"] else pos_tags_gt

        if self.config['tag_embedding_type'] == 'linear':
            pos_tags = pos_tags_parser_one_hot.float()
        elif self.config['tag_embedding_type'] == 'none':
            pos_tags = None
        else:
            raise ValueError('parameter `tag_embedding_type` can only be == `linear` or `embedding`')

        # parsing
        parser_output = self.parser(
            encoded_text_input=encoder_output,
            pos_tags=pos_tags,
            mask=mask,
            head_tags=head_tags,
            head_indices=head_indices,
        )

        decoder_output = self.decoder(
            head_tag = parser_output['head_tag'],
            dept_tag = parser_output['dept_tag'],
            head_indices = parser_output['head_indices'],
            head_tags = parser_output['head_tags'],
            attended_arcs = parser_output['attended_arcs'],
            mask = parser_output['mask'],
            metadata = parser_output['metadata'],
        )

        # calculate loss or return predictions
        if self.mode in ["train", "validation"]:
            loss = (tagger_output.loss * self.config["tagger_lambda"]
                    + decoder_output["loss"] * self.config["parser_lambda"]
                    )
            return loss
        elif self.mode == "test":
            tagger_human_readable = self.tagger.make_output_human_readable(tagger_output, mask)
            decoder_human_readable = self.decoder.make_output_human_readable(decoder_output)
            if self.config["rep_mode"] == "words":
                output_as_list_of_dicts = self.get_output_as_list_of_dicts_words(
                    tagger_human_readable, decoder_human_readable, model_input
                )
            elif self.config["rep_mode"] == "tokens":
                output_as_list_of_dicts = self.get_output_as_list_of_dicts_tokens(
                    tagger_human_readable, decoder_human_readable, model_input, mask
                )
            return output_as_list_of_dicts

    def log_gradients(self):
        """
        Logs gradient norms for all components after backward pass
        Call this method after loss.backward() but before optimizer.step()
        """
        grad_debug_dict = {}
        
        # check gradient norms for each component
        encoder_grad_norm = check_grad_norm(self.encoder)
        grad_debug_dict['encoder_grad_norm'] = encoder_grad_norm.item()
        
        tagger_grad_norm = check_grad_norm(self.tagger)
        grad_debug_dict['tagger_grad_norm'] = tagger_grad_norm.item()
        
        parser_grad_norm = check_grad_norm(self.parser)
        grad_debug_dict['parser_grad_norm'] = parser_grad_norm.item()
        
        decoder_grad_norm = check_grad_norm(self.decoder)
        grad_debug_dict['decoder_grad_norm'] = decoder_grad_norm.item()
        
        # overall model gradient norm
        overall_grad_norm = check_grad_norm(self)
        grad_debug_dict['overall_grad_norm'] = overall_grad_norm.item()
        
        self.grad_debug_list.append(grad_debug_dict)
        return grad_debug_dict

    def freeze_tagger(self):
        """Freeze tagger if asked for!"""
        for param in self.tagger.parameters():
            param.requires_grad = False

    def freeze_parser(self):
        """Freeze parser if asked for!"""
        for param in self.parser.parameters():
            param.requires_grad = False

    def get_output_as_list_of_dicts_words(
        self, tagger_output, parser_output, model_input
    ):
        """
        Returns list of dictionaries, each element in the dictionary is
        1 item in the batch, list has same length as batchsize. The dictionary
        will contain 7 fields, 'words', 'head_tags_gt', 'head_tags_pred', 'pos_tags_gt',
        'pos_tags_pred', 'head_indices_gt', 'head_indices_pred'. During evalution, all fields
        should have exactly identical length, during testing, '*_gt' keys() will have empty
        tensors.
        """
        outputs = []
        batch_size = len(tagger_output)

        for i in range(batch_size):
            elem_dict = {}

            # find non-masked indices
            valid_input_indices = (
                torch.where(model_input["encoded_input"]["words_mask_custom"][i] == 1)[
                    0
                ]
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )

            input_length = len(valid_input_indices)

            elem_dict["words"] = np.array(model_input["words"][i])[
                valid_input_indices
            ].tolist()

            elem_dict["head_tags_gt"] = (
                model_input["head_tags"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["head_tags_pred"] = [
                int(el) for el in parser_output["predicted_dependencies"][i]
            ]

            elem_dict["head_indices_gt"] = (
                model_input["head_indices"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["head_indices_pred"] = [
                int(el) for el in parser_output["predicted_heads"][i]
            ]

            elem_dict["pos_tags_gt"] = (
                model_input["pos_tags"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["pos_tags_pred"] = tagger_output[i] if self.config['use_pred_tags'] else tagger_output[i][:input_length]

            assert np.all(
                [len(elem_dict[key]) == input_length for key in elem_dict]
            ), "Predictions are not same length as input!"

            # append
            outputs.append(elem_dict)

        return outputs

    def set_mode(self, mode="train"):
        """
        This function will determine if loss should be computed or evaluation metrics
        """
        assert mode in [
            "train",
            "test",
            "validation",
        ], f"Mode {mode} is not valid. Mode should be among ['train', 'test', 'validation'] "
        self.tagger.set_mode(mode)
        self.mode = mode