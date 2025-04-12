import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from model.parser.parser_nn import *
from typing import Dict, List, Any
import math

class GCNParser(nn.Module):
    def __init__(
        self,
        config: dict,
        embedding_dim: int,
        arc_representation_dim: int,
        tag_representation_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config

        self._dropout = nn.Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(embedding_dim))

        self.head_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(embedding_dim, arc_representation_dim)
        assert self.config['gnn_enc_layers'] > 0, 'If using GCNParser, must have `gnn_enc_layers` > 0.'
        self.arc_bilinear = nn.ModuleList([
            BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    use_input_biases=True)
            for _ in range(1 + self.config['gnn_enc_layers'])]).to(self.config['device'])

        self.head_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)

        # Two-layer GCNs for updating arc representations.
        self.conv1_arc = GCNConv(arc_representation_dim, arc_representation_dim)
        self.conv2_arc = GCNConv(arc_representation_dim, arc_representation_dim)
        self.dropout_arc = nn.Dropout(input_dropout)

        # Two-layer GCNs for updating relation (tag) representations.
        self.conv1_rel = GCNConv(tag_representation_dim, tag_representation_dim)
        self.conv2_rel = GCNConv(tag_representation_dim, tag_representation_dim)
        self.dropout_rel = nn.Dropout(input_dropout)

        self.tag_representation_dim = tag_representation_dim
        self.n_edge_labels = self.config['n_edge_labels']
        
    def forward(
        self,
        encoded_text_input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: list = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
    ) -> dict:
        
        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long, device=self.config['device']), mask], dim=1)
        
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)
        
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], dim=1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], dim=1)
        
        encoded_text_input = self._dropout(encoded_text_input)
        
        # Compute initial representations.
        # (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        # (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input)))

        _, seq_len, _ = encoded_text_input.size()
        gnn_losses = []
        valid_positions = mask.sum() - batch_size
        float_mask = mask.float()

        # Loop over the number of GNN encoder layers.
        for k in range(self.config['gnn_enc_layers']):
            # Compute a soft adjacency (attention) matrix.
            attended_arcs = self.arc_bilinear[k](head_arc, dept_arc)
            arc_probs = F.softmax(attended_arcs, dim=-1)
            
            # # Compute loss as in the original implementation.
            # arc_probs_masked = masked_log_softmax(attended_arcs, mask) * float_mask.unsqueeze(1)
            # range_tensor = torch.arange(batch_size, device=self.config['device']).unsqueeze(1)
            # length_tensor = torch.arange(seq_len, device=self.config['device']).unsqueeze(0).expand(batch_size, -1)
            # arc_loss = arc_probs_masked[range_tensor, length_tensor, head_indices]
            # arc_loss = arc_loss[:, 1:]
            # arc_nll = -arc_loss.sum() / valid_positions.float()
            # gnn_losses.append(arc_nll)
            
            # Convert the dense soft adjacency matrix to a sparse representation.
            # dense_to_sparse can handle batched inputs and will adjust node indices.
            edge_index, edge_attr = batch_top_k(arc_probs, k = self.config['top_k'])
            
            # ----- Update arc representations using GCNConv -----
            # Flatten the batch: (B, N, F) -> (B*N, F)
            head_arc_flat = head_arc.reshape(batch_size * seq_len, -1)
            dept_arc_flat = dept_arc.reshape(batch_size * seq_len, -1)
            
            # Apply two GCN layers on head_arc.
            head_arc_updated = self.conv1_arc(head_arc_flat, edge_index, edge_attr)
            head_arc_updated = F.elu(head_arc_updated)
            head_arc_updated = self.dropout_arc(head_arc_updated)
            head_arc_updated = self.conv2_arc(head_arc_updated, edge_index, edge_attr)
            
            # Apply two GCN layers on dept_arc.
            dept_arc_updated = self.conv1_arc(dept_arc_flat, edge_index, edge_attr)
            dept_arc_updated = F.elu(dept_arc_updated)
            dept_arc_updated = self.dropout_arc(dept_arc_updated)
            dept_arc_updated = self.conv2_arc(dept_arc_updated, edge_index, edge_attr)
            
            # Reshape back to (batch_size, sequence_length, F)
            head_arc = head_arc_updated.reshape(batch_size, seq_len, -1)
            dept_arc = dept_arc_updated.reshape(batch_size, seq_len, -1)
            
            # ----- Update relation (tag) representations using GCNConv -----
            head_tag_flat = head_tag.reshape(batch_size * seq_len, -1)
            dept_tag_flat = dept_tag.reshape(batch_size * seq_len, -1)
            
            head_tag_updated = self.conv1_rel(head_tag_flat, edge_index, edge_attr)
            head_tag_updated = F.elu(head_tag_updated)
            head_tag_updated = self.dropout_rel(head_tag_updated)
            head_tag_updated = self.conv2_rel(head_tag_updated, edge_index, edge_attr)
            
            dept_tag_updated = self.conv1_rel(dept_tag_flat, edge_index, edge_attr)
            dept_tag_updated = F.elu(dept_tag_updated)
            dept_tag_updated = self.dropout_rel(dept_tag_updated)
            dept_tag_updated = self.conv2_rel(dept_tag_updated, edge_index, edge_attr)
            
            head_tag = head_tag_updated.reshape(batch_size, seq_len, -1)
            dept_tag = dept_tag_updated.reshape(batch_size, seq_len, -1)
            
        # Compute final attended arcs.
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

    def batch_samples(self, x: torch.Tensor, adj: torch.Tensor, batch_size: int):
        data_list = []
        for b in range(batch_size):
            N = adj[b].shape[-1]
            edge_index = torch.stack([torch.arange(N).to(self.config['device']), adj[b]], dim = 0)
            data_list.append(Data(x=x[b], edge_index=edge_index))
        batch = Batch.from_data_list(data_list)
        return batch
    
    @classmethod
    def get_model(cls, config):
        embedding_dim = config["encoder_output_dim"]
        model_obj = cls(
            config=config,
            embedding_dim=embedding_dim,
            arc_representation_dim=500,
            tag_representation_dim=100,
            input_dropout=0.3,
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj