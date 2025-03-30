from typing import Dict, Optional, Tuple, Any, List, Set
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
from model.parser.parser_utils import *
from model.decoder import masked_log_softmax
from debug import save_heatmap
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

class GCNLayer(GCNConv):
    def __init__(self, in_channels, out_channels, improved = False, cached = False, add_self_loops = None, normalize = True, bias = True, **kwargs):
        super().__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias, **kwargs)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, edge_index, edge_weight = None):
        out = super().forward(x, edge_index, edge_weight)
        return self.norm(out)

class GraphNNUnit(nn.Module):
    def __init__(self, h_dim, d_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = nn.Parameter(torch.Tensor(h_dim, d_dim))
        self.B = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.B)

    def forward(self, H, D):
        H_new = torch.matmul(H, self.W)
        D_new = torch.matmul(D, self.B)
        out = torch.nn.functional.tanh(H_new + D_new)
        return out # self.W * H + self.B * D

class GNNParser(nn.Module):
    def __init__(
        self,
        config: Dict,
        embedding_dim: int,
        n_edge_labels: int,
        tag_embedder: nn.Linear,
        arc_representation_dim: int,
        tag_representation_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config

        if self.config["use_tag_embeddings_in_parser"]:
            self.tag_embedder = tag_embedder

        self.tag_dropout = nn.Dropout(0.2)
        self.head_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        self.arc_pred = nn.ModuleList([
            BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    use_input_biases=True)
            for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])

        self.head_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        
        self.head_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim)
        self.dept_gnn = GraphNNUnit(arc_representation_dim, arc_representation_dim)
        self.head_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim)
        self.dept_rel_gnn = GraphNNUnit(tag_representation_dim, tag_representation_dim)        

        self._dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(embedding_dim))
        
        self.apply(self._init_weights)

        self.arc_representation_dim = arc_representation_dim
        self.tag_representation_dim = tag_representation_dim
        self.n_edge_labels = n_edge_labels

    def forward(
        self,
        input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        # input, mask, head_tags, head_indices = self.remove_extra_padding(input, mask, head_tags, head_indices)
        if self.config["use_tag_embeddings_in_parser"]:
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            input = torch.cat([input, tag_embeddings], dim=-1)
        
        batch_size, _, encoding_dim = input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long).to(self.config['device']), mask], dim = 1)
        
        # Concatenate the head sentinel onto the sentence representation.
        input = torch.cat([head_sentinel, input], dim=1)
        
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
        
        input = self._dropout(input)
        
        if self.config['laplacian_pe'] == 'parser':
            encoded_text = self.lap_pe(input=encoded_text,
                                       graph_laplacian=graph_laplacian,
                                       step_indices=step_indices if self.config['procedural'] else None,
                                       )
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(input)))

        # the following is based on 'Graph-based Dependency Parsing with Graph Neural Networks'
        # https://aclanthology.org/P19-1237/

        _, seq_len, _ = input.size()
        gnn_losses = []
        valid_positions = mask.sum() - batch_size
        float_mask = mask.float()
        for k in range(self.config['gnn_enc_layers']):
            arc_s = self.arc_pred[k](head_arc, dept_arc)
            # save_heatmap(arc_s, 'arc_s.pdf')
            arc_p = torch.nn.functional.softmax(arc_s, dim = -1)
            # save_heatmap(arc_p, 'arc_p.pdf')
            arc_p_masked = masked_log_softmax(arc_s, mask) * float_mask.unsqueeze(1)
            # save_heatmap(arc_p_masked, 'arc_p_masked.pdf')
            
            range_tensor = torch.arange(batch_size).unsqueeze(1)
            length_tensor = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            arc_loss = arc_p_masked[range_tensor, length_tensor, head_indices]
            arc_loss = arc_loss[:, 1:]
            arc_nll = -arc_loss.sum() / valid_positions.float()
            gnn_losses.append(arc_nll)

            # this is just a way of getting both H and D into the same feature matrix
            # and have them automatically multiplied by the weights of the soft adjacency matrix
            hx = torch.matmul(arc_p, head_arc)
            # save_heatmap(hx, 'hx.pdf')
            # this is transposed because the indices in equation 9 of the paper
            # are switched for h and arc_representation_dim
            dx = torch.matmul(arc_p.transpose(1, 2), dept_arc)
            # save_heatmap(dx, 'dx.pdf')
            fx = hx + dx
            # save_heatmap(fx, 'fx.pdf')
            # TODO: calculate losses for each layer during training and then use them in the final loss

            # adj_m = edge_index_to_adj(batch_arc.edge_index)
            # save_heatmap(adj_m, 'adj_m.pdf')
            head_arc = self.head_gnn(fx, head_arc)
            fx_intermediate = torch.matmul(arc_p, head_arc) + dx
            dept_arc = self.dept_gnn(fx_intermediate, dept_arc)

            hr = torch.matmul(arc_p, head_tag)
            dr = torch.matmul(arc_p.transpose(1, 2), dept_tag)
            fr = hr + dr
            
            head_tag = self.head_rel_gnn(fr, head_tag)
            fr_intermediate = torch.matmul(arc_p, head_tag) + dr
            dept_tag = self.dept_rel_gnn(fr_intermediate, dept_tag)

        attended_arcs = self.arc_pred[-1](head_arc, dept_arc)

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

    def batch_samples(self, x: torch.Tensor, adj: torch.Tensor, batch_size: int):
        data_list = []
        for b in range(batch_size):
            N = adj[b].shape[-1]
            edge_index = torch.stack([torch.arange(N).to(self.config['device']), adj[b]], dim = 0)
            data_list.append(Data(x=x[b], edge_index=edge_index))
        batch = Batch.from_data_list(data_list)
        return batch

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
            tag_embedder = nn.Linear(config["n_tags"], config["tag_embedding_dimension"])
        else:
            embedding_dim = config["encoder_output_dim"]
            tag_embedder = None
        
        n_edge_labels = config["n_edge_labels"]
        
        model_obj = cls(
            config=config,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=500,
            tag_representation_dim=100,
            input_dropout=0.3,
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj

class TorchGNNParser(nn.Module):
    def __init__(
        self,
        config: Dict,
        embedding_dim: int,
        n_edge_labels: int,
        tag_embedder: nn.Linear,
        arc_representation_dim: int,
        tag_representation_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config

        if self.config["use_tag_embeddings_in_parser"]:
            self.tag_embedder = tag_embedder

        self.tag_dropout = nn.Dropout(0.2)
        self.head_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        self.arc_pred = [BilinearMatrixAttention(arc_representation_dim,
                                                arc_representation_dim,
                                                use_input_biases=True).to(self.config['device'])\
                                                for _ in range(self.config['gnn_enc_layers'])]
        self.head_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        
        self.head_gnn = GCNLayer(arc_representation_dim, arc_representation_dim)
        self.dept_gnn = GCNLayer(arc_representation_dim, arc_representation_dim)
        self.head_rel_gnn = GCNLayer(tag_representation_dim, tag_representation_dim)
        self.dept_rel_gnn = GCNLayer(tag_representation_dim, tag_representation_dim)        

        self._dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(embedding_dim))
        
        self.apply(self._init_weights)

        self.arc_representation_dim = arc_representation_dim
        self.tag_representation_dim = tag_representation_dim
        self.n_edge_labels = n_edge_labels

    def forward(
        self,
        input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        # input, mask, head_tags, head_indices = self.remove_extra_padding(input, mask, head_tags, head_indices)
        if self.config["use_tag_embeddings_in_parser"]:
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            input = torch.cat([input, tag_embeddings], dim=-1)
        
        batch_size, _, encoding_dim = input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long).to(self.config['device']), mask], dim = 1)
        
        # Concatenate the head sentinel onto the sentence representation.
        input = torch.cat([head_sentinel, input], dim=1)
        
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
        
        input = self._dropout(input)
        
        if self.config['laplacian_pe'] == 'parser':
            encoded_text = self.lap_pe(input=encoded_text,
                                       graph_laplacian=graph_laplacian,
                                       step_indices=step_indices if self.config['procedural'] else None,
                                       )
            
        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(input)))
        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(input)))

        # the following is based on 'Graph-based Dependency Parsing with Graph Neural Networks'
        # https://aclanthology.org/P19-1237/

        _, seq_len, _ = input.size()
        gnn_losses = []
        valid_positions = mask.sum() - batch_size
        float_mask = mask.float()
        for k in range(self.config['gnn_enc_layers']):
            arc_s = self.arc_pred[k](head_arc, dept_arc) # NOTE: THIS IS MAKING HUGE SCORES, WTF?
            # save_heatmap(arc_s, 'arc_s.pdf')
            arc_p = torch.nn.functional.softmax(arc_s)
            # save_heatmap(arc_p, 'arc_p.pdf')
            arc_p_masked = masked_log_softmax(arc_s, mask) * float_mask.unsqueeze(1)
            # save_heatmap(arc_p_masked, 'arc_p_masked.pdf')
            
            range_tensor = torch.arange(batch_size).unsqueeze(1)
            length_tensor = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            arc_loss = arc_p_masked[range_tensor, length_tensor, head_indices]
            arc_loss = arc_loss[:, 1:]
            arc_nll = -arc_loss.sum() / valid_positions.float()
            gnn_losses.append(arc_nll)

            # save_heatmap(arc_p[0], 'arc_p.pdf')
            adj_m = torch.argmax(arc_s, dim = -1)
            # save_heatmap(head_arc, 'head_arc.pdf')

            # this is just a way of getting both H and D into the same feature matrix
            # and have them automatically multiplied by the weights of the soft adjacency matrix
            hx = torch.matmul(arc_p, head_arc)
            # save_heatmap(hx, 'hx.pdf')
            # this is transposed because the indices in equation 9 of the paper
            # are switched for h and arc_representation_dim
            dx = torch.matmul(arc_p.transpose(1, 2), dept_arc)
            # save_heatmap(dx, 'dx.pdf')
            fx = hx + dx
            # save_heatmap(fx, 'fx.pdf')
            # TODO: calculate losses for each layer during training and then use them in the final loss

            batch_arc = self.batch_samples(x=fx, adj=adj_m, batch_size=batch_size)

            # adj_m = edge_index_to_adj(batch_arc.edge_index)
            # save_heatmap(adj_m, 'adj_m.pdf')
            head_arc = self.head_gnn(x=batch_arc.x, edge_index=batch_arc.edge_index)
            head_arc = head_arc.reshape(batch_size, -1, self.arc_representation_dim)
            fx_intermediate = torch.matmul(arc_p, head_arc) + dx
            batch_arc = self.batch_samples(x=fx_intermediate, adj=adj_m, batch_size=batch_size)
            dept_arc = self.dept_gnn(batch_arc.x, batch_arc.edge_index)
            dept_arc = dept_arc.reshape(batch_size, -1, self.arc_representation_dim)

            hr = torch.matmul(arc_p, head_tag)
            dr = torch.matmul(arc_p.transpose(1, 2), dept_tag)
            fr = hr + dr
            
            batch_rel = self.batch_samples(x=fr, adj=adj_m, batch_size=batch_size)
            head_tag = self.head_rel_gnn(x=batch_rel.x, edge_index=batch_rel.edge_index)
            head_tag = head_tag.reshape(batch_size, -1, self.tag_representation_dim)
            fr_intermediate = torch.matmul(arc_p, head_tag) + dr
            batch_rel = self.batch_samples(x=fr_intermediate, adj=adj_m, batch_size=batch_size)
            dept_tag = self.dept_rel_gnn(batch_rel.x, batch_rel.edge_index)
            dept_tag = dept_tag.reshape(batch_size, -1, self.tag_representation_dim)

        attended_arcs = self.arc_pred[-1](head_arc, dept_arc)

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

    def batch_samples(self, x: torch.Tensor, adj: torch.Tensor, batch_size: int):
        data_list = []
        for b in range(batch_size):
            N = adj[b].shape[-1]
            edge_index = torch.stack([torch.arange(N).to(self.config['device']), adj[b]], dim = 0)
            data_list.append(Data(x=x[b], edge_index=edge_index))
        batch = Batch.from_data_list(data_list)
        return batch

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
            tag_embedder = nn.Linear(config["n_tags"], config["tag_embedding_dimension"])
        else:
            embedding_dim = config["encoder_output_dim"]
        
        n_edge_labels = config["n_edge_labels"]
        
        model_obj = cls(
            config=config,
            embedding_dim=embedding_dim,
            n_edge_labels=n_edge_labels,
            tag_embedder=tag_embedder,
            arc_representation_dim=500,
            tag_representation_dim=100,
            input_dropout=0.3,
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj

def edge_index_to_adj(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Converts a [2, N] edge_index to an NxN adjacency matrix.

    Args:
        edge_index (torch.Tensor): A [2, num_edges] tensor representing the edge index.

    Returns:
        torch.Tensor: An [N, N] adjacency matrix.
    """
    num_nodes = edge_index.max().item() + 1  # Determine the number of nodes
    adj_matrix = torch.zeros((num_nodes, num_nodes)).to(edge_index.device)  # Initialize NxN matrix

    # Set edges to 1 (assuming unweighted graph)
    x = edge_index[0].long()
    y = edge_index[1].long()
    adj_matrix[x, y] = 1
    # save_heatmap(adj_matrix, 'adj_matrix.pdf')
    print(torch.max(adj_matrix))
    return adj_matrix

