import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from model.utils.nn import *
from model.utils.nn import get_encoder
import numpy as np
from typing import Dict, Any, List

"""
TODO: it would be much better to subclass SimpleParser
for all the custom parsers
""" 

class GATParser(nn.Module):
    def __init__(
        self,
        config: dict,
        encoder: nn.LSTM,
        embedding_dim: int,
        arc_representation_dim: int,
        tag_representation_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.gnn_activation = getattr(F, config['gnn_activation'], lambda x: x)

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

        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        
        self.arc_bilinear = nn.ModuleList([
            BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type=self.config['bias_type'],
                                    arc_norm=self.config['arc_norm'],
                                    )
            for _ in range(1 + self.config['gnn_layers'])]).to(self.config['device'])

        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        if self.config['gnn_layers'] > 0:
            # Two-layer GATs for updating arc representations.
            self.conv1_arc = nn.ModuleList([GATv2Conv(arc_representation_dim,
                                        arc_representation_dim,
                                        heads=self.config['num_attn_heads'],
                                        concat=False,
                                        edge_dim=1,
                                        residual=True) for _ in range(self.config['gnn_layers'])]).to(self.config['device'])
            self.conv2_arc = nn.ModuleList([GATv2Conv(arc_representation_dim,
                                        arc_representation_dim,
                                        heads=self.config['num_attn_heads'],
                                        concat=False,
                                        edge_dim=1,
                                        residual=True) for _ in range(self.config['gnn_layers'])]).to(self.config['device'])
            
            # Two-layer GATs for updating relation (tag) representations.
            self.conv1_rel = nn.ModuleList([GATv2Conv(tag_representation_dim,
                                        tag_representation_dim,
                                        heads=self.config['num_attn_heads'],
                                        concat=False,
                                        edge_dim=1,
                                        residual=True) for _ in range(self.config['gnn_layers'])]).to(self.config['device'])
            self.conv2_rel = nn.ModuleList([GATv2Conv(tag_representation_dim,
                                        tag_representation_dim,
                                        heads=self.config['num_attn_heads'],
                                        concat=False,
                                        edge_dim=1,
                                        residual=True) for _ in range(self.config['gnn_layers'])]).to(self.config['device'])
        
        self._gnn_dropout = nn.Dropout(config['gnn_dropout'])
        self._mlp_dropout = nn.Dropout(config['mlp_dropout'])
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        self._run_inits(encoder_dim, arc_representation_dim)
        self.tag_representation_dim = tag_representation_dim
        
    def forward(
        self,
        input: torch.FloatTensor,
        tag_embeddings: torch.Tensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        step_indices: torch.LongTensor = None,
        graph_laplacian: torch.LongTensor = None,
        mode: str = None,
    ) -> Dict:
        
        if tag_embeddings is not None:
            input = torch.cat([input, tag_embeddings], dim=-1)

        if self.encoder_h is not None:
            if self.config["parser_rnn_type"] != 'transformer':
                # existing LSTM/RNN handling
                lengths = mask.sum(dim=1).cpu()
                packed_input = pack_padded_sequence(
                    input, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.encoder_h(packed_input)
                input, _ = pad_packed_sequence(packed_output,
                                                            batch_first=True,
                                                            total_length=input.size(1))
            else:
                # Transformer encoding
                src_key_padding_mask = mask == 0
                input = self.encoder_h(input, src_key_padding_mask=src_key_padding_mask)

        batch_size, _, encoding_dim = input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long, device=self.config['device']), mask], dim=1)
        
        # Concatenate the head sentinel onto the sentence representation.
        input = torch.cat([head_sentinel, input], dim=1)
        
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], dim=1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], dim=1)
        
        input = self._mlp_dropout(input)
        
        # Compute initial representations.
        # (batch_size, sequence_length, arc_representation_dim)
        head_arc = self._mlp_dropout(F.elu(self.head_arc_feedforward(input)))
        dept_arc = self._mlp_dropout(F.elu(self.dept_arc_feedforward(input)))
        # (batch_size, sequence_length, tag_representation_dim)
        head_tag = self._mlp_dropout(F.elu(self.head_tag_feedforward(input)))
        dep_tag = self._mlp_dropout(F.elu(self.dep_tag_feedforward(input)))

        _, seq_len, _ = input.size()
        gnn_losses = []
        valid_positions = mask.sum() - batch_size
        float_mask = mask.float()

        # loop over the number of GNN encoder layers
        if self.current_step > self.config['use_gnn_steps'] and self.config['gnn_layers'] > 0:
            for k in range(self.config['gnn_layers']):
                # compute a soft adjacency attention matrix
                arc_logits = self.arc_bilinear[k](head_arc, dept_arc)
                arc_probs = F.softmax(arc_logits, dim=-1)
                
                # compute loss as in the original implementation.
                arc_probs_masked = masked_log_softmax(arc_logits, mask) * float_mask.unsqueeze(1)
                range_tensor = torch.arange(batch_size, device=self.config['device']).unsqueeze(1)
                length_tensor = torch.arange(seq_len, device=self.config['device']).unsqueeze(0).expand(batch_size, -1)
                arc_loss = arc_probs_masked[range_tensor, length_tensor, head_indices]
                arc_loss = arc_loss[:, 1:]
                arc_nll = -arc_loss.sum() / valid_positions.float()
                gnn_losses.append(arc_nll)
                
                # convert the dense soft adjacency matrix to a sparse representation
                # dense_to_sparse can handle batched inputs and will adjust node indices
                edge_attr, edge_index = torch.topk(arc_probs, self.config['top_k'], dim=-1)
                edge_index = edge_index.reshape(edge_index.shape[0], edge_index.shape[1] * self.config['top_k'])
                edge_attr = edge_attr.reshape(edge_attr.shape[0], edge_attr.shape[1] * self.config['top_k'])
                edge_attr_T, edge_index_T = torch.topk(arc_probs.transpose(1, 2), self.config['top_k'], dim=-1)
                edge_index_T = edge_index_T.reshape(edge_index_T.shape[0], edge_index_T.shape[1] * self.config['top_k'])
                edge_attr_T = edge_attr_T.reshape(edge_attr_T.shape[0], edge_attr_T.shape[1] * self.config['top_k'])
                
                batch_head_arc = self.batch_samples(head_arc, edge_index, edge_attr)
                batch_dept_arc = self.batch_samples(dept_arc, edge_index_T, edge_attr_T)
                batch_head_tag = self.batch_samples(head_tag, edge_index, edge_attr)
                batch_dep_tag = self.batch_samples(dep_tag, edge_index_T, edge_attr_T)
                
                # update edges
                head_arc = self.conv1_arc[k](batch_head_arc.x, batch_head_arc.edge_index, batch_head_arc.edge_attr)
                head_arc = self.unbatch_samples(head_arc, batch_head_arc.batch)
                head_arc = self.gnn_activation(head_arc)
                head_arc = self._gnn_dropout(head_arc)
                
                dept_arc = self.conv2_arc[k](batch_dept_arc.x, batch_dept_arc.edge_index, batch_dept_arc.edge_attr)
                dept_arc = self.unbatch_samples(dept_arc, batch_dept_arc.batch)
                dept_arc = self.gnn_activation(dept_arc)
                dept_arc = self._gnn_dropout(dept_arc)
                
                # update relations
                head_tag = self.conv1_rel[k](batch_head_tag.x, batch_head_tag.edge_index, batch_head_tag.edge_attr)
                head_tag = self.unbatch_samples(head_tag, batch_head_tag.batch)
                head_tag = self.gnn_activation(head_tag)
                head_tag = self._gnn_dropout(head_tag)

                dep_tag = self.conv2_rel[k](batch_dep_tag.x, batch_dep_tag.edge_index, batch_dep_tag.edge_attr)
                dep_tag = self.unbatch_samples(dep_tag, batch_dep_tag.batch)
                dep_tag = self.gnn_activation(dep_tag)
                dep_tag = self._gnn_dropout(dep_tag)
                
        # compute final attended arcs
        arc_logits = self.arc_bilinear[-1](head_arc, dept_arc)

        output = {
            'head_tag': head_tag,
            'dep_tag': dep_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'arc_logits': arc_logits,
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': gnn_losses
        }
        return output
    
    def _run_inits(self, encoder_dim, arc_representation_dim):
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
        # Determine embedding_dim and tag_embedder
        if config['tag_embedding_type'] == 'linear':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
        elif config['tag_embedding_type'] == 'embedding':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
        elif config['tag_embedding_type'] == 'none':
            embedding_dim = config["encoder_output_dim"] # 768
        else:
            raise ValueError('Parameter `tag_embedding_type` can only be == `linear` or `embedding` or `none`!')            
        encoder = get_encoder(config, embedding_dim)
        model_obj = cls(
            config=config,
            encoder=encoder,
            embedding_dim=embedding_dim,
            arc_representation_dim=config['arc_representation_dim'],
            tag_representation_dim=config['tag_representation_dim'],
        )
        model_obj.softmax_multiplier = config["softmax_scaling_coeff"]
        return model_obj