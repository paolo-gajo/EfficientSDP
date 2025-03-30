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
from model.parser.parser_utils import *
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
        arc_representation_dim: int,
        tag_representation_dim: int,
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

        if self.config["use_tag_embeddings_in_parser"]:
            self.tag_embedder = tag_embedder
        self.tag_dropout = nn.Dropout(0.2)
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        if self.config["arc_pred"] == "attn":
            self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
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
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        if self.config["use_gnn"]:
            self.sentinel_fusion = HeadSentinelFusion(
                encoder_dim + self.config["encoder_output_dim"], encoder_dim
            )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self.apply(self._init_weights)

        self.tag_representation_dim = tag_representation_dim
        self.n_edge_labels = n_edge_labels

    def forward(
        self,
        encoded_text_input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        gnn_pooled_vector: torch.Tensor = None,
        step_indices: torch.Tensor = None,
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
            encoded_text_input, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=encoded_text_input.size(1)
            )

        encoded_text = self._input_dropout(encoded_text_input)
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
            ones_limit = max(torch.where(mask[0, -1] == 1)[0])
            mask_ones = mask.new_ones(batch_size, mask.shape[1], 1)
            mask = torch.cat([mask_ones, mask], dim=2)
            mask_ones = mask.new_ones(batch_size, 1, mask.shape[2])
            mask_ones[:, :, ones_limit + 2 :] = 0
            mask = torch.cat([mask_ones, mask], dim=1)
            mask[:, ones_limit + 2 :, :] = 0

        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )

        encoded_text = self._dropout(encoded_text)

        if self.config['laplacian_pe'] == 'parser':
            encoded_text = self.lap_pe(input=encoded_text,
                                       graph_laplacian=graph_laplacian,
                                       step_indices=step_indices if self.config['procedural'] else None,
                                       )

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text)))

        if self.config["arc_pred"] == "attn":
            # shape (batch_size, sequence_length, arc_representation_dim)
            dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text)))   

            # shape (batch_size, sequence_length, sequence_length)
            attended_arcs = self.arc_pred(head_arc, dept_arc)
            
            if self.config['use_parser_gnn']:
                arc_edge_index = []
                head_arc_reps = []
                dept_arc_reps = []
                for i, b in enumerate(attended_arcs):
                    arc_edge_index = b.nonzero(as_tuple=False).t()
                    head_arc_reps.append(self.gnn(head_arc[i], arc_edge_index))
                    dept_arc_reps.append(self.gnn(dept_arc[i], arc_edge_index))
                head_arc = torch.stack(head_arc_reps, dim = 0)
                dept_arc = torch.stack(dept_arc_reps, dim = 0)

                attended_arcs = self.arc_pred(
                    head_arc, dept_arc
                )
        elif self.config["arc_pred"] == "dgm":
            attended_arcs = self.arc_pred(x=head_arc, A=None)["adj"]
        else:
            raise ValueError("arc_pred can either be `attn` or `dgmc`")

        if self.config['step_bilinear_attn']:
            step_reps = get_step_reps(encoded_text, step_indices)
            step_matrix_1 = F.elu(self.step_mlp_1(step_reps))
            step_matrix_2 = F.elu(self.step_mlp_2(step_reps))
            self.step_bilinear_attn(step_matrix_1, step_matrix_2)
            pass

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,
            'mask': mask,
            'metadata': metadata,
        }

        return output

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

