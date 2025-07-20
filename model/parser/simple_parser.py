from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.parser.parser_nn import *
from model.utils.nn_utils import prepend_ones
from debug import save_heatmap
import warnings
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
            self.tag_dropout = nn.Dropout(config['tag_dropout'])
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type=self.config['bias_type'],
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        self._dropout = nn.Dropout(config['mlp_dropout'])
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
        tag_embeddings: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        if self.config["tag_embedding_type"] != 'none':
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
                src_key_padding_mask = mask == 0
                encoded_text_input = self.encoder_h(encoded_text_input, src_key_padding_mask=src_key_padding_mask)

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)

        mask, head_indices, head_tags = prepend_ones(batch_size, mask, head_indices, head_tags)
        
        encoded_text_input = self._dropout(encoded_text_input)
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dep_tag = self._dropout(F.elu(self.dep_tag_feedforward(encoded_text_input)))

        arc_logits = self.arc_bilinear(head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dep_tag': dep_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'arc_logits': arc_logits,
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
        elif config['tag_embedding_type'] == 'embedding':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
            tag_embedder = nn.Embedding(config["n_tags"], config["tag_representation_dim"])
        elif config['tag_embedding_type'] == 'none':
            embedding_dim = config["encoder_output_dim"] # 768
            tag_embedder = None
        else:
            raise ValueError('Parameter `tag_embedding_type` can only be == `linear` or `embedding` or `none`!')            
        print(f'Using {tag_embedder.__class__} for tag embeddings!')
        n_edge_labels = config["n_edge_labels"]

        encoder = get_encoder(config, embedding_dim)
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=config['arc_representation_dim'],
            tag_representation_dim=config['tag_representation_dim'],
            use_mst_decoding_for_validation=config['use_mst_decoding_for_validation'],
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj

def get_encoder(config, embedding_dim):
    kwargs = {
        'input_size': embedding_dim,
        'hidden_size': config["parser_rnn_hidden_size"],
        'num_layers': config['parser_rnn_layers'],
        'batch_first': True,
        'bidirectional': True,
        'dropout': config['rnn_dropout'],
    }
    custom_kwargs = {
        'rnn_residual': config['rnn_residual'],
    }
    if config['use_parser_rnn'] \
    and config['parser_rnn_layers'] > 0 \
    and config['parser_rnn_hidden_size'] > 0:
        parser_rnn_type = config['parser_rnn_type']
        if parser_rnn_type == 'lstm':
            encoder = nn.LSTM(**kwargs)
        elif parser_rnn_type == 'normlstm':
            encoder = LayerNormLSTM(**kwargs, **custom_kwargs)
        elif parser_rnn_type == 'rnn':
            encoder = nn.RNN(**kwargs)
        elif parser_rnn_type == 'normrnn':
            encoder = LayerNormRNN(**kwargs, **custom_kwargs)
        elif parser_rnn_type == 'gru':
            encoder = nn.GRU(**kwargs)
        elif parser_rnn_type == 'transformer':
            encoder = TransformerParserEncoder(
                input_dim=embedding_dim,
                num_layers=config['parser_rnn_layers'],
            )
        else:
            warnings.warn(f"Unknown parser_rnn_type {parser_rnn_type}, setting encoder to None.")
            encoder = None
    print(f'Using {encoder.__class__} as parser encoder!')
    return encoder

class TransformerParserEncoder(nn.Module):
    def __init__(self, input_dim, num_layers=12):
        super().__init__()

        self.hidden_size   = 768          # d_model
        self.num_heads     = 12
        self.intermediate  = 3072         # dim_feedforward

        # Project raw features to the BERT hidden size
        self.lin_in = nn.Linear(input_dim, self.hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model         = self.hidden_size,
            nhead           = self.num_heads,
            dim_feedforward = self.intermediate,
            dropout         = 0.1,
            activation      = "gelu",
            layer_norm_eps  = 1e-12,
            batch_first     = True          # (batch, seq, hidden)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer = layer,
            num_layers    = num_layers
        )

        self.lin_out = nn.Linear(self.hidden_size, input_dim)

    def forward(self, x, src_key_padding_mask):
        x = self.lin_in(x)          # (B, L, 768)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)      # (B, L, 768)
        x = self.lin_out(x)
        return x