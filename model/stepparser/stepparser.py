import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.encoder import Encoder
from model.parser import SimpleParser, GATParser
from model.tagger import Tagger
from model.decoder import GraphDecoder, masked_log_softmax
import numpy as np
import warnings
from debug.model_debugging import nan_checker, check_param_norm, check_grad_norm
import copy

class StepParser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.tagger = Tagger(self.config)

        # select the type of parser
        if self.config['parser_type'] == 'simple':
            self.parser = SimpleParser.get_model(self.config) # base setting
        elif self.config['parser_type'] == 'gat':
            self.parser = GATParser.get_model(self.config)
        self.decoder = GraphDecoder(config=config,
                                    tag_representation_dim=self.parser.tag_representation_dim,
                                    n_edge_labels = self.parser.n_edge_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.mode = "train"
        self.parser.current_step, self._current_step = 0, 0        

    def forward(self, model_input):
        # encoder
        model_input = {k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input.items()}
        model_input = {k: [el.to(self.config["device"]) for el in v] if (isinstance(v, list) and isinstance(v[0], torch.Tensor)) else v
            for k, v in model_input.items()}
        encoder_input = {k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input["encoded_input"].items()}
        
        if self.config['procedural']:
            encoder_input['step_indices'] = model_input["step_indices_tokens"]
            encoder_input['graph_laplacian'] = model_input["graph_laplacian"]

        if self.config["use_step_mask"]:
            # Create step mask for attention
            encoder_input["attention_mask"] = self.make_step_mask(
                encoder_input["attention_mask"],
                model_input["step_graph"],
                model_input["step_indices_tokens"],)
            encoder_input["words_mask_custom"] = self.token_mask_to_word_mask(
                encoder_input["attention_mask"], encoder_input)

        # Run encoder to get token/word representations
        encoder_output = self.encoder(encoder_input)

        # Determine which representation mode to use
        if self.config["rep_mode"] == "words":
            tagger_labels = model_input["pos_tags"]
            mask = encoder_input["words_mask_custom"]
            head_indices = model_input["head_indices"]
            head_tags = model_input["head_tags"]
            step_indices = model_input["step_indices"] if self.config['procedural'] else None
        elif self.config["rep_mode"] == "tokens":
            tagger_labels = model_input["pos_tags_tokens"]
            mask = encoder_input["attention_mask"]
            head_indices = model_input["head_indices_tokens"]
            head_tags = model_input["head_tags_tokens"]
            step_indices = model_input["step_indices_tokens"]

        # Use appropriate tags based on mode
        # NOTE: step graph settings are WIP
        if self.mode in ["train", "validation"]:
            head_tags, head_indices = head_tags, head_indices
        elif self.mode == "test":
            head_tags, head_indices = None, None

        # Tagging
        tagger_output = self.tagger(encoder_output, mask=mask, labels=tagger_labels)

        pos_tags_pred_one_hot = self.tagger.get_predicted_classes_as_one_hot(tagger_output.logits)

        # Ground-truth tags
        try:
            pos_tags_gt = torch.nn.functional.one_hot(tagger_labels, num_classes=self.config["n_tags"])
        except:
            warnings.warn("Ground truth tags are unavailable, using predicted tags for all purposes.")
            pos_tags_gt = pos_tags_pred_one_hot

        # Use predicted or ground truth tags based on config
        pos_tags_parser_one_hot = pos_tags_pred_one_hot if self.config["use_pred_tags"] else pos_tags_gt
        pos_tags_parser_cls_idx = torch.argmax(tagger_output.logits, dim=-1) if self.config["use_pred_tags"] else tagger_labels

        pos_tags_dict = {
            'pos_tags_one_hot': pos_tags_parser_one_hot.float(),
            'pos_tags_labels': pos_tags_parser_cls_idx,
        }

        if self.config['tag_embedding_type'] == 'linear':
            pos_tags = pos_tags_dict['pos_tags_one_hot']
        elif self.config['tag_embedding_type'] == 'embedding':
            pos_tags = pos_tags_dict['pos_tags_labels']
        elif self.config['tag_embedding_type'] == 'none':
            pos_tags = None
        else:
            raise ValueError('parameter `tag_embedding_type` can only be == `linear` or `embedding`')

        # Parsing
        parser_output = self.parser(
            encoded_text_input=encoder_output,
            pos_tags=pos_tags,
            mask=mask,
            head_tags=head_tags,
            head_indices=head_indices,
            step_indices=step_indices,
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

        gnn_losses = parser_output.get('gnn_losses', [])
        decoder_mask = decoder_output['mask']
        # Calculate loss or return predictions
        if self.mode in ["train", "validation"]:
            loss = (tagger_output.loss * self.config["tagger_lambda"]
                    + decoder_output["loss"] * self.config["parser_lambda"]
                    )
            if len(gnn_losses) > 0:
                loss += sum(gnn_losses)/len(gnn_losses) * self.config["parser_lambda"]
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
            if self.config['output_edge_scores']:
                scores = parser_output['attended_arcs']
                softmax_scores = F.softmax(scores, dim = -1)
                masked_log_softmax_scores = masked_log_softmax(scores, decoder_mask.float())
                score_var = torch.var(scores.view(scores.shape[0], -1), dim=1, unbiased=False)
                softmax_score_var = torch.var(softmax_scores.view(softmax_scores.shape[0], -1), dim=1, unbiased=False)
                masked_log_softmax_score_var = torch.var(masked_log_softmax_scores.view(masked_log_softmax_scores.shape[0], -1), dim=1, unbiased=False)
                for i in range(len(output_as_list_of_dicts)):
                    output_as_list_of_dicts[i]['attn_scores_var'] = score_var.tolist()[i]
                    output_as_list_of_dicts[i]['attn_scores_softmax_var'] = softmax_score_var.tolist()[i]
                    output_as_list_of_dicts[i]['attn_scores_masked_log_softmax_var'] = masked_log_softmax_score_var.tolist()[i]
            return output_as_list_of_dicts

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, val):
        self._current_step = val
        if hasattr(self, 'parser'):
            self.parser.current_step = val

    def freeze_gnn(self):
        for param in self.parser.conv1_arc.parameters():
            param.requires_grad = False
        for param in self.parser.conv2_arc.parameters():
            param.requires_grad = False
        for param in self.parser.conv1_rel.parameters():
            param.requires_grad = False
        for param in self.parser.conv2_rel.parameters():
            param.requires_grad = False
        for layer in self.parser.arc_bilinear[:-1]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"GNN frozen at step {self.current_step}!")

    def unfreeze_gnn(self):
        for param in self.parser.conv1_arc.parameters():
            param.requires_grad = True
        for param in self.parser.conv2_arc.parameters():
            param.requires_grad = True
        for param in self.parser.conv1_rel.parameters():
            param.requires_grad = True
        for param in self.parser.conv2_rel.parameters():
            param.requires_grad = True
        for layer in self.parser.arc_bilinear[:-1]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"GNN unfrozen at step {self.current_step}!")
        return True
    
    def init_gnn_biaffines(self, optimizer):
        original_arc_bilinear: nn.ModuleList = self.parser.arc_bilinear
        params_to_remove = set(original_arc_bilinear.parameters())
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p not in params_to_remove]
        trained_arc_bilinear = self.parser.arc_bilinear[-1]
        new_layers = [copy.deepcopy(trained_arc_bilinear) for _ in range(self.config['gnn_layers'])]
        new_arc_bilinear: nn.ModuleList = nn.ModuleList(new_layers + [trained_arc_bilinear])
        self.parser.arc_bilinear = new_arc_bilinear
        optimizer.add_param_group({"params": self.parser.arc_bilinear.parameters()})

    def log_gradients(self):
        """
        Logs gradient norms for all components after backward pass
        Call this method after loss.backward() but before optimizer.step()
        """
        grad_debug_dict = {}
        
        # Check gradient norms for each component
        encoder_grad_norm = check_grad_norm(self.encoder)
        grad_debug_dict['encoder_grad_norm'] = encoder_grad_norm.item()
        
        tagger_grad_norm = check_grad_norm(self.tagger)
        grad_debug_dict['tagger_grad_norm'] = tagger_grad_norm.item()
        
        parser_grad_norm = check_grad_norm(self.parser)
        grad_debug_dict['parser_grad_norm'] = parser_grad_norm.item()
        
        decoder_grad_norm = check_grad_norm(self.decoder)
        grad_debug_dict['decoder_grad_norm'] = decoder_grad_norm.item()
        
        # Overall model gradient norm
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