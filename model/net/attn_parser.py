import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.encoder import Encoder
from model.parser import SimpleParser, TriParser, GNNParser, GCNParser, GATParser, GATParserUnbatched, GraphRNNBilinear
from model.tagger import Tagger
from model.decoder import BilinearDecoder, masked_log_softmax, GraphDecoder
import numpy as np
import warnings
from typing import Set, Tuple
from debug.model_debugging import nan_checker, check_param_norm, check_grad_norm
from debug.viz import save_batch_heatmap, indices_to_adjacency_matrices
import copy

class AttnParser(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.tagger = Tagger(config)

        # select the type of parser
        if config['parser_type'] == 'simple':
            self.parser = SimpleParser.get_model(config) # base setting
        if config['parser_type'] == 'triaffine':
            self.parser = TriParser.get_model(config) # triaffine parser
        elif config['parser_type'] == 'gnn':
            self.parser = GNNParser.get_model(config)
        elif config['parser_type'] == 'gcn':
            self.parser = GCNParser.get_model(config)
        elif config['parser_type'] == 'gat':
            self.parser = GATParser.get_model(config)
        elif config['parser_type'] == 'gat_unbatched':
            self.parser = GATParserUnbatched.get_model(config)
        elif config['parser_type'] == 'graph_rnn':
            self.parser = GraphRNNBilinear(config)
        self.decoder = GraphDecoder(config=config,
                                    tag_representation_dim=config['tag_representation_dim'],
                                    n_edge_labels = config['n_edge_labels'])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
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
            step_graphs = model_input['step_graph'] if self.config['procedural'] else None
            graph_laplacian = model_input['graph_laplacian'] if self.config['procedural'] else None
            edge_index=model_input["edge_index"] if self.config['procedural'] else None
        elif self.mode == "test":
            head_tags, head_indices = None, None
            step_graphs = None
            graph_laplacian = model_input['graph_laplacian'] if self.config['procedural'] else None
            # TODO: this eventually needs to be None
            # it needs to be inferred during eval
            # (or could use a flag to control whether to use oracle)
            edge_index=None

        # Tagging
        tagger_output = self.tagger(encoder_output, mask=mask, labels=tagger_labels)

        # Parsing
        parser_output = self.parser(
            input=encoder_output,
            tag_embeddings=tagger_output.tag_embeddings,
            mask=mask,
            head_tags=head_tags,
            head_indices=head_indices,
            step_indices=step_indices,
            graph_laplacian=graph_laplacian,
            mode=self.mode,
        )

        decoder_output = self.decoder(
            head_tag = parser_output['head_tag'],
            dep_tag = parser_output['dep_tag'],
            head_indices = parser_output['head_indices'], # gold adjusted for sentinel
            head_tags = parser_output['head_tags'], # gold adjusted for sentinel
            arc_logits = parser_output['arc_logits'],
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
                scores = parser_output['arc_logits']
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