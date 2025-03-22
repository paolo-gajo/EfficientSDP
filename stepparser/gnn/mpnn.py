import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import List, Set, Tuple
from stepparser.gnn.utils import get_step_reps

class CustomMPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.2, aggr='mean'):
        """
        Custom MPNN layer that applies:
          - psi: a linear layer with non-linearity on the concatenation of x_i and x_j,
          - aggregation (avg) over the messages,
          - phi: another linear layer with non-linearity on the concatenation of x_i and aggregated message.
        Residual connections are added.
        
        Args:
            in_channels (int): Dimensionality of the input node features.
            out_channels (int): Dimensionality of the output node features.
            dropout (float): Dropout rate applied after the update.
            aggr (str): Aggregation method, default 'mean' (averaging).
        """
        super(CustomMPNNLayer, self).__init__(aggr=aggr)
        
        # psi: transforms the concatenated (x_i, x_j) into a message
        self.psi = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU()
        )
        
        # phi: updates x_i based on its original value and the aggregated message
        self.phi = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # If dimensions differ, project x_i to out_channels for the residual connection.
        self.residual_lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x, edge_index):
        """
        Forward pass for the custom MPNN layer.
        
        Args:
            x (Tensor): Node features, shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in COO format.
            
        Returns:
            Tensor: Updated node features, shape [num_nodes, out_channels].
        """
        # Compute aggregated messages using the custom message function.
        agg_message = self.propagate(edge_index, x=x)
        
        # Compute residual (project if necessary)
        res = self.residual_lin(x) if self.residual_lin is not None else x
        
        # Concatenate the original features with the aggregated message and update.
        out = self.phi(torch.cat([x, agg_message], dim=-1))
        
        # Add residual connection and apply dropout.
        out = out + res
        return self.dropout(out)
    
    def message(self, x_i, x_j):
        """
        Message function using psi.
        
        Args:
            x_i (Tensor): Central node features.
            x_j (Tensor): Neighbor node features.
            
        Returns:
            Tensor: Computed message for each edge.
        """
        return self.psi(torch.cat([x_i, x_j], dim=-1))


class MPNNNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout=0.2, aggr='mean'):
        """
        MPNN network that stacks multiple custom MPNN layers and mimics the architecture of GATNet.
        
        Args:
            input_dim (int): Dimensionality of input node features.
            output_dim (int): Dimensionality of output node features.
            num_layers (int): Number of MPNN layers.
            dropout (float): Dropout rate.
            aggr (str): Aggregation method to be used in each layer.
        """
        super(MPNNNet, self).__init__()
        
        # Stack multiple MPNN layers.
        self.mpnn_layers = nn.ModuleList([
            CustomMPNNLayer(input_dim if i == 0 else output_dim, output_dim, dropout=dropout, aggr=aggr)
            for i in range(num_layers)
        ])
        
        # Down-projection layer, analogous to GATNet.
        self.down_proj = nn.Linear(output_dim, output_dim)
        
        # Merge layer to combine encoder and MPNN outputs in process_step_representations.
        self.merge = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the stacked MPNN layers.
        
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge index in COO format.
            
        Returns:
            Tensor: Updated node features.
        """
        for layer in self.mpnn_layers:
            x = layer(x, edge_index)
        x = self.down_proj(x)
        return x
    
    def process_step_representations(self, encoder_output, step_indices, edge_index_batch):
        """
        Process step representations through the MPNN and merge with encoder output.
        
        Args:
            encoder_output (Tensor): Encoder outputs for the batch.
            step_indices (Tensor): Step indices for each token.
            step_graphs (List[Set[Tuple]]): Edge sets (as binary edges) for each sample.
            
        Returns:
            Tuple[Tensor, Tensor]: Updated encoder output and pooled MPNN output.
        """
        # Compute step-level representations.
        step_representations = get_step_reps(h=encoder_output, step_indices=step_indices)
        
        gnn_outputs = []
        for x, edge_index in zip(step_representations, edge_index_batch):
            if edge_index.numel() > 0:
                gnn_out = self(x, edge_index.to(x.device))
            else:
                gnn_out = torch.zeros(x.shape).to(x.device)
            gnn_outputs.append(gnn_out)
        
        # Update encoder outputs with the corresponding MPNN outputs.
        for enc_out, gnn_out, step_idx in zip(encoder_output, gnn_outputs, step_indices):
            enc_out = enc_out.clone()
            step_idx = step_idx - 1  # Adjust for 1-indexing.
            step_num = int(torch.max(step_idx).item())
            for i in range(step_num + 1):
                step_i_indices = torch.where(step_idx == i)[0]
                enc_out_step = enc_out.index_select(0, step_i_indices)
                # Expand the MPNN output corresponding to the current step.
                gnn_out_step = gnn_out[i].unsqueeze(0).expand(enc_out_step.shape[0], -1)
                updated = F.relu(self.merge(torch.cat([enc_out_step, gnn_out_step], dim=-1)))
                enc_out = enc_out.index_copy(0, step_i_indices, updated)
                
        # Pool the MPNN output (mean pooling).
        gnn_out_pooled = gnn_out.mean(dim=0)
        return encoder_output, gnn_out_pooled