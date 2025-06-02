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
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DGMParser(nn.Module):
    def __init__(
        self,
        config: Dict,
        encoder: nn.LSTM,
        embedding_dim: int,
        n_edge_labels: int,
        tag_embedder: nn.Linear,
        arc_representation_dim: int,
        tag_representation_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config

        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            self.seq_encoder = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            self.encoder_h = None
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder

        self.tag_dropout = nn.Dropout(0.2)
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        # assert self.config['gnn_layers'] > 0, 'If using GNNParser, must have `gnn_layers` > 0.'
        self.arc_bilinear = nn.ModuleList([
            BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    use_input_biases=True)
            for _ in range(1 + self.config['gnn_layers'])]).to(self.config['device'])
        # self.arc_bilinear_t = nn.ModuleList([
        #     BilinearMatrixAttention(arc_representation_dim,
        #                             arc_representation_dim,
        #                             use_input_biases=True)
        #     for _ in range(1 + self.config['gnn_layers'])]).to(self.config['device'])

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        if self.config['gnn_layers'] > 0:
            self.head_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim)
            self.dept_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim)
            self.head_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim)
            self.dept_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim)

        self._dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        
        self.apply(self._init_weights)

        self.arc_representation_dim = arc_representation_dim
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

        # encoded_text_input, mask, head_tags, head_indices = self.remove_extra_padding(encoded_text_input, mask, head_tags, head_indices)
        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags['pos_tags_labels'])))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.encoder_h is not None:
            # Compute lengths from the binary mask.
            lengths = mask.sum(dim=1).cpu()
            # Pack the padded sequence using the lengths.
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.seq_encoder(packed_input)
            # Unpack the sequence, ensuring the output has the original sequence length.
            encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long).to(self.config['device']), mask], dim = 1)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)
        
        if self.config['procedural']:
            sentinel_step_index = torch.zeros(step_indices.shape[0], dtype=torch.long).unsqueeze(1)
            step_indices = torch.cat([sentinel_step_index.to(self.config['device']), step_indices], dim = 1)
        
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        
        encoded_text_input = self._dropout(encoded_text_input)
        
        if self.config['laplacian_pe'] == 'parser':
            encoded_text = self.lap_pe(encoded_text_input=encoded_text,
                                       graph_laplacian=graph_laplacian,
                                       step_indices=step_indices if self.config['procedural'] else None,
                                       )
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input)))
        
        # the following is based on 'Graph-based Dependency Parsing with Graph Neural Networks'
        # https://aclanthology.org/P19-1237/

        _, seq_len, _ = encoded_text_input.size()
        gnn_losses = []
        valid_positions = mask.sum() - batch_size
        float_mask = mask.float()

        for k in range(self.config['gnn_layers']):
            attended_arcs = self.arc_bilinear[k](head_arc, dept_arc)
            # attended_arcs_t = self.arc_bilinear_t[k](head_arc, dept_arc)
            arc_probs = torch.nn.functional.softmax(attended_arcs, dim = -1)
            # arc_probs_t = torch.nn.functional.softmax(attended_arcs_t, dim = -1)
            # save_heatmap(arc_probs, 'arc_probs.pdf')
            arc_probs_masked = masked_log_softmax(attended_arcs, mask) * float_mask.unsqueeze(1)
            # save_heatmap(arc_probs_masked, 'arc_probs_masked.pdf')
            
            # range_tensor = torch.arange(batch_size).unsqueeze(1)
            # length_tensor = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            # arc_loss = arc_probs_masked[range_tensor, length_tensor, head_indices]
            # arc_loss = arc_loss[:, 1:]
            # arc_nll = -arc_loss.sum() / valid_positions.float()
            # gnn_losses.append(arc_nll)

            # this is just a way of getting both H and D into the same feature matrix
            # and have them automatically multiplied by the weights of the soft adjacency matrix
            hx = torch.matmul(arc_probs, head_arc)
            # save_heatmap(hx, 'hx.pdf')
            # this is transposed because the indices in equation 9 of the paper
            # are switched for h and arc_representation_dim
            dx = torch.matmul(arc_probs.transpose(1, 2), dept_arc)
            # save_heatmap(dx, 'dx.pdf')
            fx = hx + dx
            # save_heatmap(fx, 'fx.pdf')
            # TODO: calculate losses for each layer during training and then use them in the final loss

            # adj_m = edge_index_to_adj(batch_arc.edge_index)
            # save_heatmap(adj_m, 'adj_m.pdf')
            head_arc = self.head_gnn(fx, head_arc)
            fx_intermediate = torch.matmul(arc_probs, head_arc) + dx
            dept_arc = self.dept_gnn(fx_intermediate, dept_arc)

            hr = torch.matmul(arc_probs, head_tag)
            dr = torch.matmul(arc_probs.transpose(1, 2), dept_tag)
            fr = hr + dr
            
            head_tag = self.head_rel_gnn(fr, head_tag)
            fr_intermediate = torch.matmul(arc_probs, head_tag) + dr
            dept_tag = self.dept_rel_gnn(fr_intermediate, dept_tag)

        attended_arcs = self.arc_bilinear[-1](head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': gnn_losses
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
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=config["parser_rnn_hidden_size"],
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=0.3,
            )
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
            input_dropout=0.3,
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj