import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from transformers import AutoTokenizer
from model.encoder import Encoder, BERTWordEmbeddings
from model.parser import BiaffineDependencyParser, GNNParser, GCNParser, GATParser
from model.tagger import Tagger
from model.gnn import GATNet, MPNNNet
from model.decoder import GraphDecoder
import numpy as np
import warnings
from typing import Set, Tuple, List
import matplotlib.pyplot as plt


class StepParser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.config["encoder_output_dim"] = (self.encoder.encoder.embeddings.word_embeddings.weight.shape[-1])

        if self.config["use_gnn"] == "gat":
            self.gnn = GATNet(
                self.config["encoder_output_dim"],
                self.config["encoder_output_dim"],
                num_layers=3,
                heads=8,
            )
        if self.config["use_gnn"] == "mpnn":
            self.gnn = MPNNNet(
                input_dim=self.config["encoder_output_dim"],
                output_dim=self.config["encoder_output_dim"],
                num_layers=3,  # or adjust as needed
                dropout=0.2,
                aggr="mean",  # mean aggregation as specified
            )
        self.tagger = Tagger(self.config)
        if self.config['parser_type'] == 'mtrfg':
            self.parser = BiaffineDependencyParser.get_model(self.config)
        elif self.config['parser_type'] == 'gnn':
            self.parser = GNNParser.get_model(self.config)
        elif self.config['parser_type'] == 'gcn':
            self.parser = GCNParser.get_model(self.config)
        elif self.config['parser_type'] == 'gat':
            self.parser = GATParser.get_model(self.config)
        self.decoder = GraphDecoder(config=config,
                                    tag_representation_dim=self.parser.tag_representation_dim,
                                    n_edge_labels = self.parser.n_edge_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.mode = "train"

    def forward(self, model_input):
        # encoder
        model_input = {
            k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input.items()
        }

        model_input = {
            k: [el.to(self.config["device"]) for el in v] if (isinstance(v, list) and isinstance(v[0], torch.Tensor)) else v
            for k, v in model_input.items()
        }
        
        encoder_input = {
            k: v.to(self.config["device"]) if isinstance(v, torch.Tensor) else v
            for k, v in model_input["encoded_input"].items()
        }
        if self.config['procedural']:
            encoder_input['step_indices'] = model_input["step_indices_tokens"]
            encoder_input['graph_laplacian'] = model_input["graph_laplacian"]

        if self.config["use_step_mask"]:
            # Create step mask for attention
            encoder_input["attention_mask"] = self.make_step_mask(
                encoder_input["attention_mask"],
                model_input["step_graph"],
                model_input["step_indices_tokens"],
            )
            encoder_input["words_mask_custom"] = self.token_mask_to_word_mask(
                encoder_input["attention_mask"], encoder_input
            )
        
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
        if self.mode in ["train", "validation"]:
            head_tags, head_indices = head_tags, head_indices
            step_graphs = model_input['step_graph'] if self.config['procedural'] else None
            graph_laplacian = model_input['graph_laplacian'] if self.config['procedural'] else None
            edge_index=model_input["edge_index"] if self.config['procedural'] else None
        elif self.mode == "test":
            head_tags, head_indices = None, None
            step_graphs = None
            graph_laplacian = model_input['graph_laplacian'] if self.config['procedural'] else None
            # TODO: THIS NEEDS TO BE NONE AFTER I'M DONE TESTING!
            # IT NEEDS TO BE INFERRED DURING EVAL/TEST!
            # OR PUT A FLAG TO LET ME CONTROL WHETHER TO USE ORACLE/NOT ORACLE FOR LAPLACIAN
            edge_index=None

        # GNN processing
        gnn_out_pooled = None
        if self.config["use_gnn"] in ["mpnn", "gat"]:
            encoder_output, gnn_out_pooled = self.gnn.process_step_representations(
                encoder_output=encoder_output,
                step_indices=step_indices,
                edge_index_batch=edge_index,
            )

        # Tagging
        tagger_output = self.tagger(
            encoder_output, mask=mask, labels=tagger_labels
        )
        pos_tags_pred = self.tagger.get_predicted_classes_as_one_hot(tagger_output.logits)

        # Ground-truth tags
        try:
            pos_tags_gt = torch.nn.functional.one_hot(
                tagger_labels, num_classes=self.config["n_tags"]
            )
        except:
            warnings.warn(
                "Ground truth tags are unavailable, using predicted tags for all purposes."
            )
            pos_tags_gt = pos_tags_pred

        # Use predicted or ground truth tags based on config
        pos_tags_parser = pos_tags_pred if self.config["use_pred_tags"] else pos_tags_gt

        pos_tags_dict = {
            'pos_tags_one_hot': pos_tags_parser.float(),
            'pos_tags_labels': tagger_labels,
        }

        # Parsing
        parser_output = self.parser(
            encoded_text_input=encoder_output,
            pos_tags=pos_tags_dict, # if self.config['one_hot_tags'] else tagger_output.logits,
            mask=mask,
            head_tags=head_tags,
            head_indices=head_indices,
            step_indices=step_indices,
            graph_laplacian=graph_laplacian,
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

        # Calculate loss or return predictions
        if self.mode in ["train", "validation"]:
            loss = (tagger_output.loss * self.config["tagger_lambda"]
                    + decoder_output["loss"] * self.config["parser_lambda"]
                    )
            if self.config["parser_type"] == 'gnn' \
                and self.config["gnn_enc_layers"] > 0 \
                and len(parser_output["gnn_losses"]) > 0:
                loss += sum(parser_output["gnn_losses"])/len(parser_output["gnn_losses"]) * self.config["parser_lambda"]
            return loss
        elif self.mode == "test":
            tagger_human_readable = self.tagger.make_output_human_readable(tagger_output, mask)
            parser_human_readable = self.decoder.make_output_human_readable(decoder_output)
            if self.config["rep_mode"] == "words":
                output_as_list_of_dicts = self.get_output_as_list_of_dicts_words(
                    tagger_human_readable, parser_human_readable, model_input
                )
            elif self.config["rep_mode"] == "tokens":
                output_as_list_of_dicts = self.get_output_as_list_of_dicts_tokens(
                    tagger_human_readable, parser_human_readable, model_input, mask
                )
            return output_as_list_of_dicts

    def make_step_mask(
        self,
        attention_mask: torch.Tensor,
        step_graph: Set[Tuple[int, int]],
        step_idx_tokens: torch.Tensor,
    ):
        """
        This function takes in an attention mask, a step graph, and per-token step indices
        to return a mask where the ones (1) are only present in areas
        relative to nodes connected within the step graph
        """
        step_idx_tokens = step_idx_tokens.to(self.config["device"])

        _, seq_len = attention_mask.shape
        step_mask_list = []

        for nodes, step_indices, attn_mask in zip(
            step_graph, step_idx_tokens, attention_mask
        ):
            # get inverse edge directions
            nodes_reverse = set([tuple(sorted(el, reverse=True)) for el in nodes])
            # nodes = nodes.union(nodes_reverse)
            nodes = nodes_reverse
            unique_nodes = set(step_indices.tolist()).difference({0})

            # Initialize mask with zeros
            mask = torch.zeros(
                (seq_len, seq_len), dtype=torch.float32, device=self.config["device"]
            )

            # Add self-loops
            for i in unique_nodes:
                mask += (
                    (step_indices[:, None] == i) & (step_indices[None, :] == i)
                ).int()

            # Add edges
            for src, tgt in nodes:
                mask += (
                    (step_indices[:, None] == src) & (step_indices[None, :] == tgt)
                ).int()

            pad_limit = torch.max(torch.where(attn_mask == 1)[0])

            # NOTE: these two lines include the SEP tokens, don't remove
            mask[: pad_limit + 1, pad_limit : pad_limit + 1] = 1
            mask[pad_limit : pad_limit + 1, : pad_limit + 1] = 1
            # NOTE: this line includes the original padding, don't remove
            mask[pad_limit:, : pad_limit + 1] = 1

            # NOTE: the line below is the equivalent of using the original mask
            # mask[:,:pad_limit+1] = 1

            step_mask_list.append(mask)

        # save_heatmap(mask, filename='step_mask.pdf')

        batch_step_mask = torch.stack(step_mask_list, dim=0)
        return batch_step_mask

    def token_mask_to_word_mask(self, token_step_mask, encoder_input):
        """
        Convert a 3D token-level step mask to a word-level step mask using word_ids_custom.

        Args:
            token_step_mask (torch.Tensor): Shape (batch, seq_len, seq_len) with 1s for active token pairs.
            encoder_input (dict): Contains 'word_ids_custom' (shape [batch, seq_len]) and 'words_mask_custom'.

        Returns:
            torch.Tensor: Word-level mask (batch, max_words, max_words) where 1s indicate active word pairs.
        """
        batch_size, seq_len, _ = token_step_mask.shape
        max_words = encoder_input["words_mask_custom"].shape[1]
        device = token_step_mask.device

        word_step_mask = torch.zeros(
            (batch_size, max_words, max_words), dtype=torch.long, device=device
        )

        for b in range(batch_size):
            word_ids = encoder_input["word_ids_custom"][b]  # (seq_len)
            words_mask = encoder_input["words_mask_custom"][b]  # (max_words)

            current_token_mask = token_step_mask[b].float()  # (seq_len, seq_len)
            # save_heatmap(current_token_mask, f'current_token_mask_{b}.pdf')
            # Create a new tensor for mapped word IDs
            # Map all special tokens (-100) to a temporary index and ensure all indices are valid
            mapped_word_ids = word_ids.clone()

            # Create a mask for valid word IDs (not -100 and within range)
            valid_mask = (mapped_word_ids >= 0) & (mapped_word_ids < max_words)

            # For invalid indices, we'll temporarily use 0 (we'll filter these out later)
            mapped_word_ids = torch.where(
                valid_mask, mapped_word_ids, torch.zeros_like(mapped_word_ids)
            )

            # Create one-hot matrix: (max_words, seq_len)
            W = (
                torch.nn.functional.one_hot(mapped_word_ids, num_classes=max_words)
                .float()
                .permute(1, 0)
                .to(device)
            )

            # Apply valid mask to zero out contributions from special tokens
            W = W * valid_mask.float().unsqueeze(0)

            # Compute word interactions: W @ token_mask @ W.T
            word_interactions = torch.matmul(W, torch.matmul(current_token_mask, W.T))
            current_word_mask = (word_interactions > 0).long()

            # Apply word mask to zero out padding
            current_word_mask *= words_mask.unsqueeze(1)
            current_word_mask *= words_mask.unsqueeze(0)

            pad_limit = torch.max(torch.where(words_mask == 1)[0])

            # NOTE: these two lines include the SEP tokens, don't remove
            current_word_mask[: pad_limit + 1, pad_limit : pad_limit + 1] = 1
            current_word_mask[pad_limit : pad_limit + 1, : pad_limit + 1] = 1
            # NOTE: this line includes the original padding, don't remove
            current_word_mask[pad_limit:, : pad_limit + 1] = 1

            # Optionally save heatmap (consider commenting this out for production)
            # save_heatmap(current_word_mask, f'word_step_mask_batch_{b}.pdf')

            word_step_mask[b] = current_word_mask

        return word_step_mask

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

            ## find non-masked indices
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
            elem_dict["pos_tags_pred"] = tagger_output[i]

            elem_dict["word_ids_custom"] = (
                model_input["encoded_input"]["word_ids_custom"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )

            assert np.all(
                [len(elem_dict[key]) == input_length for key in elem_dict]
            ), "Predictions are not same length as input!"

            ## append
            outputs.append(elem_dict)

        return outputs

    def get_output_as_list_of_dicts_tokens(
        self, tagger_output, parser_output, model_input, mask
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

            ## find non-masked indices
            valid_input_indices = (
                torch.where(mask[i] == 1)[0].cpu().detach().numpy().tolist()
            )

            input_length = len(valid_input_indices)

            elem_dict["input_ids"] = np.array(
                model_input["encoded_input"]["input_ids"][i]
            )[valid_input_indices].tolist()

            elem_dict["head_tags_gt"] = (
                model_input["head_tags_tokens"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["head_tags_pred"] = [
                int(el) for el in parser_output["predicted_dependencies"][i]
            ]

            elem_dict["head_indices_gt"] = (
                model_input["head_indices_tokens"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["head_indices_pred"] = [
                int(el) for el in parser_output["predicted_heads"][i]
            ]

            elem_dict["pos_tags_gt"] = (
                model_input["pos_tags_tokens"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )
            elem_dict["pos_tags_pred"] = tagger_output[i]

            elem_dict["word_ids_custom"] = (
                model_input["encoded_input"]["word_ids_custom"][i]
                .cpu()
                .detach()
                .numpy()[valid_input_indices]
                .tolist()
            )

            assert np.all(
                [len(elem_dict[key]) == input_length for key in elem_dict]
            ), "Predictions are not same length as input!"

            ## append
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