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
        self.config = config
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            self.encoder_h = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            self.encoder_h = None
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(config['tag_dropout'])

        self.sib_head_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.sib_dep_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.cop_head_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.cop_dep_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.gp_head_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.gp_dep_ff = nn.Linear(encoder_dim, arc_representation_dim)
        self.gp_head_dep_ff = nn.Linear(encoder_dim, arc_representation_dim)

        self.tri_sib = TrilinearMatrixAttention(
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    mode = 'sib',
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    arc_norm=self.config['arc_norm'],
                                    )
        num_params = sum(p.numel() for p in self.tri_sib.parameters() if p.requires_grad)
        print(f"Number of parameters self.tri_sib: {num_params}")
        
        self.tri_cop = TrilinearMatrixAttention(
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    mode = 'cop',
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    arc_norm=self.config['arc_norm'],
                                    )
        num_params = sum(p.numel() for p in self.tri_cop.parameters() if p.requires_grad)
        print(f"Number of parameters self.tri_cop: {num_params}")
        
        self.tri_gp = TrilinearMatrixAttention(
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    arc_representation_dim,
                                    mode = '',
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    arc_norm=self.config['arc_norm'],
                                    )
        
        num_params = sum(p.numel() for p in self.tri_gp.parameters() if p.requires_grad)
        print(f"Number of parameters self.tri_gp: {num_params}")

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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.encoder_h is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.encoder_h(packed_input)
            encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
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
            
        sib_head = self._dropout(F.elu(self.sib_head_ff(encoded_text_input)))
        sib_dep = self._dropout(F.elu(self.sib_dep_ff(encoded_text_input)))
        cop_head = self._dropout(F.elu(self.cop_head_ff(encoded_text_input)))
        cop_dep = self._dropout(F.elu(self.cop_dep_ff(encoded_text_input)))
        gp_head = self._dropout(F.elu(self.gp_head_ff(encoded_text_input)))
        gp_dep = self._dropout(F.elu(self.gp_dep_ff(encoded_text_input)))
        gp_head_dep = self._dropout(F.elu(self.gp_head_dep_ff(encoded_text_input)))


        sib = self.tri_sib(sib_head, sib_dep, sib_dep)
        cop = self.tri_cop(cop_head, cop_dep, cop_head)
        gp = self.tri_gp(gp_head, gp_head_dep, gp_dep)
        
        output = {
            'head_tag': head_tag,
            'dep_tag': dep_tag,
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
        parser_rnn_type = config['parser_rnn_type']
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
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
