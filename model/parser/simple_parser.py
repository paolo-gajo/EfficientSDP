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
import numpy as np

class SimpleParser(nn.Module):
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
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            self.encoder_h = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2 if self.config['parser_rnn_type'] != 'transformer' else embedding_dim
        else:
            self.encoder_h = None
            encoder_dim = embedding_dim
        self.lstm_input_size = embedding_dim
        self.lstm_hidden_size = encoder_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(0.2)
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type=self.config['bias_type'],
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        self._dropout = nn.Dropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation
        if self.config['parser_init'] == 'xu':
            self.apply(self._init_weights_xavier_uniform)
        elif self.config['parser_init'] == 'norm':
            self.apply(self._init_norm)
        elif self.config['parser_init'] == 'xu+norm':
            self.apply(self._init_weights_xavier_uniform)
            torch.nn.init.normal_(self.head_arc_feedforward.weight, std=np.sqrt(2 / (encoder_dim + arc_representation_dim)))
            torch.nn.init.normal_(self.dept_arc_feedforward.weight, std=np.sqrt(2 / (encoder_dim + arc_representation_dim)))
        if self.config['bma_init'] == 'norm':
            torch.nn.init.normal_(self.arc_bilinear._weight_matrix, std=np.sqrt(2 / (encoder_dim + arc_representation_dim)))
            torch.nn.init.normal_(self.arc_bilinear._bias, std=np.sqrt(2 / (encoder_dim + arc_representation_dim)))
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
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.encoder_h is not None:
            if self.config["parser_rnn_type"] != 'transformer':
                # existing LSTM/RNN handling
                lengths = mask.sum(dim=1).cpu()
                packed_input = pack_padded_sequence(
                    encoded_text_input, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.encoder_h(packed_input)
                encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                            batch_first=True,
                                                            total_length=encoded_text_input.size(1))
            else:
                # Transformer encoding
                # Transformer expects (seq_len, batch, embed_dim)
                # Create padding mask: True for positions that should be masked
                src = encoded_text_input.permute(1, 0, 2)
                # mask: (batch, seq) -> src_key_padding_mask: (batch, seq)
                src_key_padding_mask = mask == 0
                # Apply transformer encoder
                encoded = self.encoder_h(src, src_key_padding_mask=src_key_padding_mask)
                # Back to (batch, seq, embed)
                encoded_text_input = encoded.permute(1, 0, 2)

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
            'attended_arcs': attended_arcs,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': [],
        }

        return output

    def _init_norm(self, module):
        """
        Initialize module parameters using a normal distribution.
        Applies nn.init.normal_ to weight and bias tensors with mean=0.0 and std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size).
        For 1D tensors (e.g., biases or projection vectors), temporarily unsqueeze to 2D.
        """
        # Weights
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight
            if w.dim() < 2:
                # Temporarily make it 2D
                w_unsq = w.unsqueeze(0)
                nn.init.normal_(w_unsq, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))
                module.weight.data = w_unsq.squeeze(0)
            else:
                nn.init.normal_(w, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))

        # Biases
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            b = module.bias
            if b.dim() < 2:
                b_unsq = b.unsqueeze(0)
                nn.init.normal_(b_unsq, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))
                module.bias.data = b_unsq.squeeze(0)
            else:
                nn.init.normal_(b, mean=0.0, std=np.sqrt(2/(self.lstm_input_size + self.lstm_hidden_size)))

    def _init_weights_xavier_uniform(self, module):
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
        # Determine embedding_dim and tag_embedder
        if config['tag_embedding_type'] == 'linear':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
            tag_embedder = nn.Linear(config["n_tags"], config["tag_representation_dim"])
            print('Using nn.Linear for tag embeddings!')
        elif config['tag_embedding_type'] == 'embedding':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
            tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
            print('Using nn.Embedding for tag embeddings!')
        elif config['tag_embedding_type'] == 'none':
            embedding_dim = config["encoder_output_dim"] # 768
            tag_embedder = None
        else:
            raise ValueError('Parameter `tag_embedding_type` can only be == `linear` or `embedding` or `none`!')            
        n_edge_labels = config["n_edge_labels"]

        # Build encoder
        encoder = None
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            typ = config['parser_rnn_type']
            if typ == 'lstm':
                print(f'Using {nn.LSTM} as parser encoder!')
                encoder = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif typ == 'normlstm':
                print(f'Using {LayerNormLSTM} as parser encoder!')
                encoder = LayerNormLSTM(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif typ == 'rnn':
                print(f'Using {nn.RNN} as parser encoder!')
                encoder = nn.RNN(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif typ == 'normrnn':
                print(f'Using {LayerNormRNN} as parser encoder!')
                encoder = LayerNormRNN(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif typ == 'gru':
                print(f'Using {nn.GRU} as parser encoder!')
                encoder = nn.GRU(
                    input_size=embedding_dim,
                    hidden_size=config["parser_rnn_hidden_size"],
                    num_layers=config['parser_rnn_layers'],
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.3,
                )
            elif typ == 'transformer':
                print(f'Using {nn.TransformerEncoderLayer} as parser encoder!')
                layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=config.get('transformer_n_heads', 8),
                    dim_feedforward=config.get('transformer_ff_dim', 4 * embedding_dim),
                    dropout=0.1,
                )
                encoder = nn.TransformerEncoder(
                    layer,
                    num_layers=config['parser_rnn_layers'],
                )
            else:
                print(f'Using {None} as parser encoder!')
                warnings.warn(f"Unknown parser_rnn_type {typ}, setting encoder to None.")
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
            use_mst_decoding_for_validation=config['use_mst_decoding_for_validation'],
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj