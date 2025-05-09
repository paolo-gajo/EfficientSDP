from typing import Dict, Optional, Tuple, Any, List, Set
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
from model.parser.parser_nn import *
from model.decoder import masked_log_softmax
import math
from debug import save_heatmap
import warnings
import copy

class TriParser(nn.Module):
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
    ) -> None:
        super().__init__()
        raise NotImplementedError('this is wip for a parser with triaffine scoring')
        self.config = config
        if self.config["use_parser_rnn"]:
            self.encoder_h = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(0.2)
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type='simple',
                                    arc_norm=self.config['arc_norm'],
                                    )


        self._dropout = nn.Dropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
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
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
        encoder_attentions: torch.FloatTensor = None,
    ) -> Dict[str, torch.Tensor]:
        
        pos_tags = pos_tags['pos_tags_one_hot'] if self.config['tag_embedding_type'] == 'linear' else pos_tags['pos_tags_labels']

        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.config["use_parser_rnn"]:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.encoder_h(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)

        mask_ones = mask.new_ones(batch_size, 1)
        mask = torch.cat([mask_ones, mask], dim = 1)
        
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        
        encoded_text_input = self._dropout(encoded_text_input)
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input)))


        attended_arcs = self.arc_bilinear(head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,# if encoder_attentions is None else encoder_attentions,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': [],
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
        if config["tag_embedding_type"] != 'none':
            embedding_dim = (
                config["encoder_output_dim"] + config["tag_representation_dim"]
            )
            if config['tag_embedding_type'] == 'linear':
                tag_embedder = nn.Linear(config["n_tags"], config["tag_representation_dim"])
            else:
                tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
        else:
            embedding_dim = config["encoder_output_dim"]
            tag_embedder = None
        n_edge_labels = config["n_edge_labels"]
        parser_rnn_type = config['parser_rnn_type']
        if config['use_parser_rnn']:
            if parser_rnn_type == 'lstm':
                encoder = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif parser_rnn_type == 'gru':
                encoder = nn.GRU(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            else:
                warnings.warn(f"Parser type `{parser_rnn_type}` is neither `gru` nor `lstm`. Setting it to None.")
                encoder = None
        else:
            encoder = None
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=config['arc_representation_dim'],
            tag_representation_dim=config['tag_representation_dim'],
            dropout=0.3,
            use_mst_decoding_for_validation = config['use_mst_decoding_for_validation']
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj
    
class DualEncParser(nn.Module):
    
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
    ) -> None:
        super().__init__()
        self.config = config

        if self.config["use_parser_rnn"]:
            self.encoder_h = encoder
            self.encoder_d = copy.deepcopy(encoder)
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(0.2)
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type='simple',
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        if self.config['gnn_enc_layers'] > 0:
            self.head_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim, use_residual=self.config['parser_residual'])
            self.dept_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim, use_residual=self.config['parser_residual'])
            self.head_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim, use_residual=self.config['parser_residual'])
            self.dept_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim, use_residual=self.config['parser_residual'])

        self._dropout = nn.Dropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
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
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags['pos_tags_labels'])))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.config["use_parser_rnn"]:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output_h, _ = self.encoder_h(packed_input)
            packed_output_d, _ = self.encoder_d(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoded_text_input_h, _ = pad_packed_sequence(packed_output_h,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))
            encoded_text_input_d, _ = pad_packed_sequence(packed_output_d,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input_h.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)

        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input_h = torch.cat([head_sentinel, encoded_text_input_h], dim=1)
        encoded_text_input_d = torch.cat([head_sentinel, encoded_text_input_d], dim=1)

        mask_ones = mask.new_ones(batch_size, 1)
        mask = torch.cat([mask_ones, mask], dim = 1)
        
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        
        encoded_text_input = self._dropout(encoded_text_input)
        
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input_h)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input_d)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input_h)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input_d)))

        for k in range(self.config['gnn_enc_layers']):
            fx = (head_arc + dept_arc) / 2

            head_arc = self.head_gnn(head_arc, dept_arc)
            fx_intermediate = (head_arc + dept_arc) / 2
            dept_arc = self.dept_gnn(head_arc, dept_arc)

            fr = (head_tag + dept_tag) / 2
            
            head_tag = self.head_rel_gnn(head_tag, dept_tag)
            # fr_intermediate = (head_tag + dept_tag) / 2
            dept_tag = self.dept_rel_gnn(head_tag, dept_tag)

        attended_arcs = self.arc_bilinear(head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': [],
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
        if config["tag_embedding_type"] != 'none':
            embedding_dim = (
                config["encoder_output_dim"] + config["tag_representation_dim"]
            )
            # tag_embedder = nn.Linear(config["n_tags"], config["tag_representation_dim"])
            tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
        else:
            embedding_dim = config["encoder_output_dim"]
            tag_embedder = None
        n_edge_labels = config["n_edge_labels"]
        parser_rnn_type = config['parser_rnn_type']
        if config['use_parser_rnn']:
            if parser_rnn_type == 'lstm':
                encoder = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif parser_rnn_type == 'gru':
                encoder = nn.GRU(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            else:
                warnings.warn(f"Parser type `{parser_rnn_type}` is neither `gru` nor `lstm`. Setting it to None.")
                encoder = None
        else:
            encoder = None
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=config['arc_representation_dim'],
            tag_representation_dim=config['tag_representation_dim'],
            dropout=0.3,
            use_mst_decoding_for_validation = config['use_mst_decoding_for_validation']
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj
    
class MultiParser(nn.Module):
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
    ) -> None:
        super().__init__()
        self.config = config
        if self.config["use_parser_rnn"]:
            self.encoder_h = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(0.2)
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type='simple',
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        if self.config['gnn_enc_layers'] > 0:
            self.head_gnn = nn.ModuleList([GraphNNUnit(arc_representation_dim,
                                                        arc_representation_dim,
                                                        use_residual=self.config['parser_residual']) \
                                                        for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
            self.dept_gnn = nn.ModuleList([GraphNNUnit(arc_representation_dim,
                                                        arc_representation_dim,
                                                        use_residual=self.config['parser_residual']) \
                                                        for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
            self.head_rel_gnn = nn.ModuleList([GraphNNUnit(tag_representation_dim,
                                                        tag_representation_dim,
                                                        use_residual=self.config['parser_residual']) \
                                                        for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
            self.dept_rel_gnn = nn.ModuleList([GraphNNUnit(tag_representation_dim,
                                                        tag_representation_dim,
                                                        use_residual=self.config['parser_residual']) \
                                                        for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])

        self._dropout = nn.Dropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
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
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags['pos_tags_labels'])))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.config["use_parser_rnn"]:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.encoder_h(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)

        mask_ones = mask.new_ones(batch_size, 1)
        mask = torch.cat([mask_ones, mask], dim = 1)
        
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        
        encoded_text_input = self._dropout(encoded_text_input)
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input)))

        for k in range(self.config['gnn_enc_layers']):
            head_arc = self.head_gnn[k](head_arc, dept_arc)
            dept_arc = self.dept_gnn[k](head_arc, dept_arc)
            head_tag = self.head_rel_gnn[k](head_tag, dept_tag)
            dept_tag = self.dept_rel_gnn[k](head_tag, dept_tag)

        attended_arcs = self.arc_bilinear(head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': [],
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
        if config["tag_embedding_type"] != 'none':
            embedding_dim = (
                config["encoder_output_dim"] + config["tag_representation_dim"]
            )
            # tag_embedder = nn.Linear(config["n_tags"], config["tag_representation_dim"])
            tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
        else:
            embedding_dim = config["encoder_output_dim"]
            tag_embedder = None
        n_edge_labels = config["n_edge_labels"]
        parser_rnn_type = config['parser_rnn_type']
        if config['use_parser_rnn']:
            if parser_rnn_type == 'lstm':
                encoder = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif parser_rnn_type == 'gru':
                encoder = nn.GRU(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            else:
                warnings.warn(f"Parser type `{parser_rnn_type}` is neither `gru` nor `lstm`. Setting it to None.")
                encoder = None
        else:
            encoder = None
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=config['arc_representation_dim'],
            tag_representation_dim=config['tag_representation_dim'],
            dropout=0.3,
            use_mst_decoding_for_validation = config['use_mst_decoding_for_validation']
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj