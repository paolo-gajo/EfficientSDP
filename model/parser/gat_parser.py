import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from model.parser.parser_nn import *
import math

class GATParser(nn.Module):
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
        self.arc_bilinear = nn.ModuleList([
            BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type='simple',
                                    arc_norm=self.config['arc_norm'],
                                    )
            for _ in range(1 + self.config['gnn_enc_layers'])]).to(self.config['device'])

        self.head_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(embedding_dim, tag_representation_dim)

        # Two-layer GATs for updating arc representations.
        self.conv1_arc = nn.ModuleList([GATv2Conv(arc_representation_dim,
                                    arc_representation_dim,
                                    heads=self.config['num_attn_heads'],
                                    concat=False,
                                    edge_dim=1,
                                    residual=True) for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
        self.conv2_arc = nn.ModuleList([GATv2Conv(arc_representation_dim,
                                    arc_representation_dim,
                                    heads=self.config['num_attn_heads'],
                                    concat=False,
                                    edge_dim=1,
                                    residual=True) for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
        self.dropout_arc = nn.Dropout(input_dropout)

        # Two-layer GATs for updating relation (tag) representations.
        self.conv1_rel = nn.ModuleList([GATv2Conv(tag_representation_dim,
                                    tag_representation_dim,
                                    heads=self.config['num_attn_heads'],
                                    concat=False,
                                    edge_dim=1,
                                    residual=True) for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
        self.conv2_rel = nn.ModuleList([GATv2Conv(tag_representation_dim,
                                    tag_representation_dim,
                                    heads=self.config['num_attn_heads'],
                                    concat=False,
                                    edge_dim=1,
                                    residual=True) for _ in range(self.config['gnn_enc_layers'])]).to(self.config['device'])
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
        if self.current_step > self.config['use_gnn_steps'] and self.config['gnn_enc_layers'] > 0:
            for k in range(self.config['gnn_enc_layers']):
                # Compute a soft adjacency (attention) matrix.
                attended_arcs = self.arc_bilinear[k](head_arc, dept_arc)
                arc_probs = F.softmax(attended_arcs, dim=-1)
                
                # Compute loss as in the original implementation.
                arc_probs_masked = masked_log_softmax(attended_arcs, mask) * float_mask.unsqueeze(1)
                range_tensor = torch.arange(batch_size, device=self.config['device']).unsqueeze(1)
                length_tensor = torch.arange(seq_len, device=self.config['device']).unsqueeze(0).expand(batch_size, -1)
                arc_loss = arc_probs_masked[range_tensor, length_tensor, head_indices]
                arc_loss = arc_loss[:, 1:]
                arc_nll = -arc_loss.sum() / valid_positions.float()
                gnn_losses.append(arc_nll)
                
                # Convert the dense soft adjacency matrix to a sparse representation.
                # dense_to_sparse can handle batched inputs and will adjust node indices.
                edge_attr, edge_index = torch.topk(arc_probs, self.config['top_k'], dim=-1)
                edge_attr_T, edge_index_T = torch.topk(arc_probs.transpose(1, 2), self.config['top_k'], dim=-1)
                edge_index = edge_index.reshape(edge_index.shape[0], edge_index.shape[1] * self.config['top_k'])
                edge_index_T = edge_index_T.reshape(edge_index_T.shape[0], edge_index_T.shape[1] * self.config['top_k'])
                edge_attr = edge_attr.reshape(edge_attr.shape[0], edge_attr.shape[1] * self.config['top_k'])
                edge_attr_T = edge_attr_T.reshape(edge_attr_T.shape[0], edge_attr_T.shape[1] * self.config['top_k'])
                
                batch_head_arc = self.batch_samples(head_arc, edge_index, edge_attr)
                batch_dept_arc = self.batch_samples(dept_arc, edge_index_T, edge_attr_T)
                batch_head_tag = self.batch_samples(head_tag, edge_index, edge_attr)
                batch_dept_tag = self.batch_samples(dept_tag, edge_index_T, edge_attr_T)
                
                # Apply two GCN layers on head_arc.
                head_arc = self.conv1_arc[k](batch_head_arc.x, batch_head_arc.edge_index, batch_head_arc.edge_attr)
                head_arc = self.unbatch_samples(head_arc, batch_head_arc.batch)
                # head_arc_updated = F.tanh(head_arc_updated)
                # head_arc_updated = self.dropout_arc(head_arc_updated)
                # head_arc_updated = self.conv2_arc[k](head_arc_updated, edge_index, edge_attr)
                
                # dept_arc_flat = (dept_arc_flat + head_arc_updated) / 2
                
                # Apply two GCN layers on dept_arc.
                dept_arc = self.conv2_arc[k](batch_dept_arc.x, batch_dept_arc.edge_index, batch_dept_arc.edge_attr)
                dept_arc = self.unbatch_samples(dept_arc, batch_dept_arc.batch)
                # dept_arc_updated = F.tanh(dept_arc_updated)
                # dept_arc_updated = self.dropout_arc(dept_arc_updated)
                # dept_arc_updated = self.conv2_arc[k](dept_arc_updated, edge_index, edge_attr)
                
                head_tag = self.conv1_rel[k](batch_head_tag.x, batch_head_tag.edge_index, batch_head_tag.edge_attr)
                head_tag = self.unbatch_samples(head_tag, batch_head_tag.batch)
                # head_tag = F.tanh(head_tag)
                # head_tag = self.dropout_rel(head_tag)
                # head_tag = self.conv2_rel[k](head_tag, edge_index, edge_attr)
                
                # dept_tag_flat = (dept_tag_flat + head_tag) / 2
                
                dept_tag = self.conv2_rel[k](batch_dept_tag.x, batch_dept_tag.edge_index, batch_dept_tag.edge_attr)
                dept_tag = self.unbatch_samples(dept_tag, batch_dept_tag.batch)
                # dept_tag = F.tanh(dept_tag)
                # dept_tag = self.dropout_rel(dept_tag)
                # dept_tag = self.conv2_rel[k](dept_tag, edge_index, edge_attr)
            
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

    def unbatch_samples(self, x: torch.Tensor, batch_index: torch.Tensor):
        batch_size = torch.max(batch_index) + 1
        x_unbatched = []
        for b in range(batch_size):
            index_mask = torch.where(batch_index==b)
            x_filtered = x[index_mask]
            x_unbatched.append(x_filtered)
        x_unbatched = torch.stack(x_unbatched)
        return x_unbatched

    def batch_samples(self, x: torch.Tensor, edge_i: torch.Tensor, edge_a: torch.Tensor = None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        k = edge_i.shape[1] // seq_len  # infer K from shape (B, N*K)

        data_list = []
        for b in range(batch_size):
            src = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(seq_len, k).flatten()
            tgt = edge_i[b]  # already flattened to shape [N*K]
            edge_index = torch.stack([src, tgt], dim=0)
            if edge_a is not None:
                edge_attr = edge_a[b].reshape(-1, 1)  # ensure shape [N*K, 1] for edge_dim=1
                data_list.append(Data(x=x[b], edge_index=edge_index, edge_attr=edge_attr))
            else:
                data_list.append(Data(x=x[b], edge_index=edge_index))
        return Batch.from_data_list(data_list)
    
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