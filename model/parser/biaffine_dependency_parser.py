from typing import Dict, Optional, Tuple, Any, List, Set
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy
import matplotlib.pyplot as plt
from model.gnn import DGM_c, GATNet, get_step_reps, pad_square_matrix, LaplacePE
import sys
from debug.viz import save_heatmap
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


class BiaffineDependencyParser(nn.Module):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Parameters
    encoder_output_dim : ``int``, required
        The output dimension of the text encoder
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        config: Dict,
        encoder: nn.LSTM,
        embedding_dim: int,
        n_edge_labels: int,
        tag_embedder: nn.Linear,
        tag_representation_dim: int,
        arc_representation_dim: int,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config

        if self.config["use_parser_lstm"]:
            self.seq_encoder = encoder
            encoder_dim = self.config["parser_lstm_hidden_size"] * 2
        else:
            encoder_dim = embedding_dim

        if self.config["use_tag_embeddings_in_parser"] and self.config['one_hot_tags']:
            self.tag_embedder = tag_embedder
        self.tag_dropout = nn.Dropout(0.2)
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        if self.config["arc_pred"] == "attn":
            self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)
            if self.config["mhabma"]:
                self.arc_pred = MHABMA(
                    arc_representation_dim,
                    arc_representation_dim,
                    use_input_biases=True,
                    num_heads=16,
                )
            else:
                self.arc_pred = BilinearMatrixAttention(
                    arc_representation_dim,
                    arc_representation_dim,
                    use_input_biases=True,
                )
            if self.config['use_parser_gnn']:
                self.gnn = GATNet(arc_representation_dim, arc_representation_dim, num_layers = 1, heads = 1, dropout=0.2)
        elif self.config["arc_pred"] == "dgm":
            self.arc_pred = DGM_c(self.config,
                                  input_dim=arc_representation_dim,
                                  hidden_dims=arc_representation_dim,
                                  num_layers=2,
                                  num_gnn_layers=1,
                                  conv_type='gat',
                                  heads=4,
                                  apply_diffusion=False,
                                  )

        if self.config['step_bilinear_attn']:
            self.step_mlp_1 = nn.Linear(encoder_dim, arc_representation_dim)
            self.step_mlp_2 = nn.Linear(encoder_dim, arc_representation_dim)
            self.step_bilinear_attn = BilinearMatrixAttention(
                arc_representation_dim, arc_representation_dim, use_input_biases=True
            )

        if self.config['laplacian_pe'] == 'parser':
            self.lap_pe = LaplacePE(embedding_dim=embedding_dim,
                                    max_steps=self.config['max_steps'])
        
        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, n_edge_labels
        )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        if self.config["use_gnn"]:
            self.sentinel_fusion = HeadSentinelFusion(
                encoder_dim + self.config["encoder_output_dim"], encoder_dim
            )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self.apply(self._init_weights)

    def forward(
        self,
        encoded_text_input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        og_mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        gnn_pooled_vector: torch.Tensor = None,
        step_indices: torch.Tensor = None,
        step_reps: torch.Tensor = None,
        step_graphs: torch.Tensor = None,
        graph_laplacian: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        encoded_text_input: torch.FloatTensor
            A tensor of shape [batch_size, sequence_length, embedding_dim] that contains the token embeddings
            for each token in the sequence. `embedding_dim` is the size of the embedding.
        pos_tags: torch.LongTensor
            A tensor of shape [batch_size, sequence_length] that contains the tags ( predicted or groundtruth )
        mask: torch.LongTensor
            A tensor of shape [batch_size, sequence_length] denoting the padded elements in the batch.
            (0 if padding, 1 if non-padding)
        metadata : List[Dict[str, Any]], optional (default=None)
            A list of dictionaries of metadata for each batch element which has keys:
                words : ``List[str]``, required.
                    The tokens in the original sentence.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """

        # encoded_text_input, mask, head_tags, head_indices = self.remove_extra_padding(encoded_text_input, mask, head_tags, head_indices)
        if self.config["use_tag_embeddings_in_parser"]:
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.config["use_parser_lstm"]:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.seq_encoder(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoded_text, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=encoded_text_input.size(1)
            )
        else:
            encoded_text = encoded_text_input

        encoded_text = self._input_dropout(encoded_text)
        batch_size, _, encoding_dim = encoded_text.size()
        head_sentinel = self._head_sentinel
        if gnn_pooled_vector is not None:
            head_sentinel = self.sentinel_fusion(head_sentinel, gnn_pooled_vector)
        head_sentinel = head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], dim=1)
        
        if self.config['procedural']:
            sentinel_step_index = torch.zeros(step_indices.shape[0], dtype=torch.long).unsqueeze(1)
            step_indices = torch.cat([sentinel_step_index.to(self.config['device']), step_indices], dim = 1)

        if not self.config["use_step_mask"]:
            mask_ones = mask.new_ones(batch_size, 1)
            mask = torch.cat([mask_ones, mask], dim=1)
        else:
            # save_heatmap(mask[0], filename='mask_1.pdf')
            ones_limit = max(torch.where(mask[0, -1] == 1)[0])
            mask_ones = mask.new_ones(batch_size, mask.shape[1], 1)
            mask = torch.cat([mask_ones, mask], dim=2)
            mask_ones = mask.new_ones(batch_size, 1, mask.shape[2])
            mask_ones[:, :, ones_limit + 2 :] = 0
            mask = torch.cat([mask_ones, mask], dim=1)
            mask[:, ones_limit + 2 :, :] = 0
            # save_heatmap(mask[0], filename='mask_2.pdf')
        og_mask = torch.cat([og_mask.new_ones(batch_size, 1), og_mask], dim=1)

        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        float_mask = mask.float()
        # save_heatmap(float_mask[0], filename='mask_3.pdf')
        encoded_text = self._dropout(encoded_text)

        if self.config['laplacian_pe'] == 'parser':
            encoded_text = self.lap_pe(input=encoded_text,
                                       graph_laplacian=graph_laplacian,
                                       step_indices=step_indices if self.config['procedural'] else None,
                                       )

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(
            F.elu(self.head_arc_feedforward(encoded_text))
        )

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(
            F.elu(self.head_tag_feedforward(encoded_text))
        )
        child_tag_representation = self._dropout(
            F.elu(self.child_tag_feedforward(encoded_text))
        )

        if self.config["arc_pred"] == "attn":
            # shape (batch_size, sequence_length, arc_representation_dim)
            child_arc_representation = self._dropout(
                F.elu(self.child_arc_feedforward(encoded_text))
            )   

            # shape (batch_size, sequence_length, sequence_length)
            attended_arcs = self.arc_pred(
                head_arc_representation, child_arc_representation
            )
            if self.config['use_parser_gnn']:
                arc_edge_index = []
                head_arc_reps = []
                child_arc_reps = []
                for i, b in enumerate(attended_arcs):
                    arc_edge_index = b.nonzero(as_tuple=False).t()
                    head_arc_reps.append(self.gnn(head_arc_representation[i], arc_edge_index))
                    child_arc_reps.append(self.gnn(child_arc_representation[i], arc_edge_index))
                head_arc_representation = torch.stack(head_arc_reps, dim = 0)
                child_arc_representation = torch.stack(child_arc_reps, dim = 0)

                attended_arcs = self.arc_pred(
                    head_arc_representation, child_arc_representation
                )
        elif self.config["arc_pred"] == "dgm":
            attended_arcs = self.arc_pred(x=head_arc_representation, A=None)["adj"]
        else:
            raise ValueError("arc_pred can either be `attn` or `dgmc`")

        # mask scores before decoding
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf

        if not self.config["use_step_mask"]:
            attended_arcs = (
                attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
            )
        else:
            attended_arcs = attended_arcs + minus_mask

        if self.config['step_bilinear_attn']:
            step_reps = get_step_reps(encoded_text, step_indices)
            step_matrix_1 = F.elu(self.step_mlp_1(step_reps))
            step_matrix_2 = F.elu(self.step_mlp_2(step_reps))
            self.step_bilinear_attn(step_matrix_1, step_matrix_2)
            pass

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation,
                child_tag_representation,
                attended_arcs,
                og_mask,
            )

        predicted_heads = predicted_heads.to(head_tag_representation.device)
        predicted_head_tags = predicted_head_tags.to(head_tag_representation.device)

        if head_indices is not None and head_tags is not None:

            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
                og_mask=og_mask,
            )
            loss = arc_nll + tag_nll

        else:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
                og_mask=og_mask,
            )
            loss = arc_nll + tag_nll

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
            "og_mask": og_mask,
            "words": [meta["words"] for meta in metadata],
        }

        return output_dict
    
    def _init_weights(self, module):
        """
        Initialize module parameters using Xavier Uniform initialization.
        Applies nn.init.xavier_uniform_ to weight and bias tensors.
        For 1D tensors (e.g., biases), temporarily unsqueeze to make them 2D.
        """
        # Initialize weights if they exist and are tensors.
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if module.weight.dim() < 2:
                # For 1D tensors, unsqueeze to apply Xavier uniform.
                weight_unsqueezed = module.weight.unsqueeze(0)
                nn.init.xavier_uniform_(weight_unsqueezed)
                module.weight.data = weight_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.weight)

        # Initialize biases if they exist and are tensors.
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            if module.bias.dim() < 2:
                bias_unsqueezed = module.bias.unsqueeze(0)
                nn.init.xavier_uniform_(bias_unsqueezed)
                module.bias.data = bias_unsqueezed.squeeze(0)
            else:
                nn.init.xavier_uniform_(module.bias)

    @classmethod
    def get_model(cls, config):
        if config["use_tag_embeddings_in_parser"]:
            embedding_dim = (
                config["encoder_output_dim"] + config["tag_embedding_dimension"]
            )
        else:
            embedding_dim = config["encoder_output_dim"]
        n_edge_labels = config["n_edge_labels"]
        encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=config["parser_lstm_hidden_size"],
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        tag_embedder = nn.Linear(config["n_tags"], config["tag_embedding_dimension"])
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=500,
            tag_representation_dim=100,
            dropout=0.3,
            input_dropout=0.3,
            use_mst_decoding_for_validation = config['use_mst_decoding_for_validation']
            
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj

    def remove_extra_padding(self, encoded_text_input, mask, head_tags, head_indices):
        """
        For the inputs, we have padding, and we are going to remove
        additional padding
        """

        largest_batch_input_len = max(
            [torch.sum(mask_elem).item() for mask_elem in mask]
        )
        mask = mask[:, :largest_batch_input_len]
        head_tags = head_tags[:, :largest_batch_input_len]
        head_indices = head_indices[:, :largest_batch_input_len]
        encoded_text_input = encoded_text_input[:, :largest_batch_input_len]

        return encoded_text_input, mask, head_tags, head_indices

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        output_dict.pop("mask")
        mask = output_dict.pop("og_mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = list(instance_tags[1:length])
            # `instance_tags` are the indices of the tags. If the names themselves are needed,
            # you should write a mapping function to do the conversion before the following line
            head_tag_labels.append(instance_tags)
            head_indices.append(instance_heads)

        output_dict["predicted_dependencies"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.Tensor,
        og_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        og_float_mask = og_mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(
            batch_size, get_device_of(attended_arcs)
        ).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        if not self.config["use_step_mask"]:
            normalised_arc_logits = masked_log_softmax(attended_arcs, mask)
            # save_heatmap(normalised_arc_logits[0], filename='normalised_arc_logits_1_og.pdf')
            normalised_arc_logits = (
                normalised_arc_logits
                * float_mask.unsqueeze(2)
                * float_mask.unsqueeze(1)
            )
            # save_heatmap(normalised_arc_logits[0], filename='normalised_arc_logits_2_og.pdf')
        else:
            normalised_arc_logits = masked_log_softmax(attended_arcs, mask)
            # save_heatmap(normalised_arc_logits[0], filename='normalised_arc_logits_1.pdf')
            normalised_arc_logits = normalised_arc_logits * float_mask
            # save_heatmap(normalised_arc_logits[0], filename='normalised_arc_logits_2.pdf')

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        # if not self.config['use_step_mask']:
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, og_mask.unsqueeze(-1)
        )
        # save_heatmap(normalised_head_tag_logits[0], filename='normalised_head_tag_logits_1.pdf')
        normalised_head_tag_logits = (
            normalised_head_tag_logits * og_float_mask.unsqueeze(-1)
        )
        # save_heatmap(normalised_head_tag_logits[0], filename='normalised_head_tag_logits_2.pdf')
        # else:
        #     normalised_head_tag_logits = masked_log_softmax(head_tag_logits,
        #                                                     mask)
        #     save_heatmap(normalised_head_tag_logits[0], filename='normalised_head_tag_logits_1.pdf')
        #     normalised_head_tag_logits = normalised_head_tag_logits
        #     save_heatmap(normalised_head_tag_logits[0], filename='normalised_head_tag_logits_2.pdf')
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        child_index = (
            timestep_index.view(1, sequence_length)
            .expand(batch_size, sequence_length)
            .long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()

        return arc_nll, tag_nll

    def _greedy_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        diag_mask = torch.diag(attended_arcs.new(mask.size(1)).fill_(-numpy.inf))
        # save_heatmap(attended_arcs[0], filename='attended_arcs_diag.pdf')
        attended_arcs = attended_arcs + diag_mask
        # save_heatmap(attended_arcs[0], filename='attended_arcs_nodiag.pdf')
        # save_heatmap(attended_arcs[0], filename='attended_arcs_2.pdf')
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            if not self.config["use_step_mask"]:
                minus_mask = (1 - mask).byte().unsqueeze(2)
            else:
                minus_mask = (1 - mask).byte()
            attended_arcs.masked_fill_(minus_mask.bool(), -numpy.inf)
        # save_heatmap(attended_arcs[0], filename='attended_arcs_3.pdf')
        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding maximum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = (
            head_tag_representation.size()
        )

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [
            batch_size,
            sequence_length,
            sequence_length,
            tag_representation_dim,
        ]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(
            *expanded_shape
        ).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(
            *expanded_shape
        ).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(
            head_tag_representation, child_tag_representation
        )

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)

        """
            Here, before feeding scores to the MST algorithm, we perform softmax
            with temperature, to make most likely label to have score of 1 and rest
            are squashed to 0. This way, MST will take most likely path to build a tree 
            and we won't run into an issue of low precision and high recall. We do this
            softmax for both, arc labels and edge labels. 
        """
        pairwise_head_logits = self.softmax_multiplier * (
            pairwise_head_logits
            - torch.max(pairwise_head_logits, dim=3)[0].unsqueeze(dim=3)
        )
        normalized_pairwise_head_logits = F.log_softmax(
            pairwise_head_logits, dim=3
        ).permute(0, 3, 1, 2)

        # Shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.softmax_multiplier * (
            attended_arcs - torch.max(attended_arcs, dim=2)[0].unsqueeze(dim=2)
        )
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.

        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):

            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)

        return torch.from_numpy(numpy.stack(heads)), torch.from_numpy(
            numpy.stack(head_tags)
        )

    def _get_head_tags(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[
            range_vector, head_indices
        ]
        selected_head_tag_representations = (
            selected_head_tag_representations.contiguous()
        )
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )
        return head_tag_logits

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset)

class MHABMA(nn.Module):
    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        num_heads: int,
        activation=None,
        use_input_biases: bool = False,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if out_features == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(num_heads, matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(
                torch.Tensor(num_heads, out_features, matrix_1_dim, matrix_2_dim)
            )

        self._bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation or Passthrough()
        self.use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        if self.use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], dim=-1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            matrix_2 = torch.cat([matrix_2, bias2], dim=-1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        B_m, N_m, L_m, d_m = matrix_1.shape
        matrix_1 = matrix_1.reshape(B_m * N_m, L_m, d_m)
        matrix_2 = matrix_2.reshape(B_m * N_m, L_m, d_m)
        weight = self._weight_matrix.unsqueeze(0).expand(B_m, -1, -1, -1)
        B_w, N_w, L_w, d_w = weight.shape
        weight = weight.reshape(B_w * N_w, L_w, d_w)
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.bmm(matrix_1, weight)
        final = torch.bmm(intermediate, matrix_2.transpose(1, 2))
        final_biased = final.squeeze(1) + self._bias
        out = final_biased.reshape(B_m, N_m, L_m, L_m)
        out = out.mean(dim = 1)
        return self.activation(out)
       
class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function. This function has
    a matrix of weights `W` and a bias `b`, and the similarity between the two matrices `X`
    and `Y` is computed as `X W Y^T + b`.

    # Parameters

    matrix_1_dim : `int`, required
        The dimension of the matrix `X`, described above.  This is `X.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : `int`, required
        The dimension of the matrix `Y`, described above.  This is `Y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `X W Y^T + b` calculation.  Default is
        linear, i.e. no activation.
    use_input_biases : `bool`, optional (default = `False`)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    out_features : `int`, optional (default = `1`)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to `torch.nn.Bilinear`
        for matrices, rather than vectors.
    """

    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        activation=None,
        use_input_biases: bool = False,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if out_features == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(
                torch.Tensor(out_features, matrix_1_dim, matrix_2_dim)
            )

        self._bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation or Passthrough()
        self.use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        if self.use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], dim=-1)
            matrix_2 = torch.cat([matrix_2, bias2], dim=-1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        final_biased = final.squeeze(1) + self._bias
        return self.activation(final_biased)

class BilinearMatrixAttentionDozatManning(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function. This function has
    a matrix of weights `W` and a bias `b`, and the similarity between the two matrices `X`
    and `Y` is computed as `X W Y^T + b`.

    # Parameters

    matrix_1_dim : `int`, required
        The dimension of the matrix `X`, described above.  This is `X.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : `int`, required
        The dimension of the matrix `Y`, described above.  This is `Y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `X W Y^T + b` calculation.  Default is
        linear, i.e. no activation.
    use_input_biases : `bool`, optional (default = `False`)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    out_features : `int`, optional (default = `1`)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to `torch.nn.Bilinear`
        for matrices, rather than vectors.
    """

    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        activation=None,
        use_input_biases: bool = False,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if out_features == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(
                torch.Tensor(out_features, matrix_1_dim, matrix_2_dim)
            )

        self._bias = nn.Parameter(torch.Tensor(matrix_1_dim))
        self.activation = activation or Passthrough()
        self.use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        if self.use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], dim=-1)
            matrix_2 = torch.cat([matrix_2, bias2], dim=-1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        bias = torch.matmul(matrix_1, self._bias.unsqueeze(1))
        final_biased = final.squeeze(1) + bias
        return self.activation(final_biased)

def masked_log_softmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1
) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.arange(size, dtype=torch.long, device=f"cuda:{device}")
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


class Passthrough(torch.nn.Module):
    def forward(self, output):
        return output


class InputVariationalDropout(torch.nn.Dropout):
    """
    (from AllenNLP)
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.

        # Parameters

        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`

        # Returns

        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(
            ones, self.p, self.training, inplace=False
        )
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor


def decode_mst(
    energy: numpy.ndarray, length: int, has_labels: bool = True
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.

    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arborescences on graphs.

    # Parameters

    energy : `numpy.ndarray`, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is `False`,
        the tensor should have shape (timesteps, timesteps) instead.
    length : `int`, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : `bool`, optional, (default = `True`)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise Exception("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise Exception("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    chu_liu_edmonds(
        length,
        score_matrix,
        current_nodes,
        final_edges,
        old_input,
        old_output,
        representatives,
    )

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]

    return heads, head_type


def chu_liu_edmonds(
    length: int,
    score_matrix: numpy.ndarray,
    current_nodes: List[bool],
    final_edges: Dict[int, int],
    old_input: numpy.ndarray,
    old_output: numpy.ndarray,
    representatives: List[Set[int]],
):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.

    Note that this function operates in place, so variables
    will be modified.

    # Parameters

    length : `int`, required.
        The number of nodes.
    score_matrix : `numpy.ndarray`, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : `List[bool]`, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges : `Dict[int, int]`, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input : `numpy.ndarray`, required.
    old_output : `numpy.ndarray`, required.
    representatives : `List[Set[int]]`, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.

    # Returns

    Nothing - all variables are modified in place.

    """
    # Set the initial graph to be the greedy best one.
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle

            # Add the new edge score to the cycle weight
            # and subtract the edge we're considering removing.
            score = (
                cycle_weight
                + score_matrix[node, node_in_cycle]
                - score_matrix[parents[node_in_cycle], node_in_cycle]
            )

            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    chu_liu_edmonds(
        length,
        score_matrix,
        current_nodes,
        final_edges,
        old_input,
        old_output,
        representatives,
    )

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


def _find_cycle(
    parents: List[int], length: int, current_nodes: List[bool]
) -> Tuple[bool, List[int]]:

    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, list(cycle)

class HeadSentinelFusion(nn.Module):
    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        """
        Args:
            input_dim: The dimension of the concatenated vector.
            output_dim: The desired output dimension (same as the original head vector dim).
            use_nonlinearity: Whether to apply a non-linear activation after projection.
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.use_nonlinearity = use_nonlinearity

    def forward(self, head_sentinel, pooled_vector):
        # Concatenate the two vectors along the last dimension.
        fused_vector = torch.cat([head_sentinel, pooled_vector], dim=-1)
        projected = self.projection(fused_vector)
        if self.use_nonlinearity:
            projected = F.relu(
                projected
            )  # or another activation function like GELU/Tanh
        return projected
