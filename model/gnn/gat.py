import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import List, Set, Tuple
from model.gnn.utils import get_step_reps

class GATNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, heads, dropout=0.2):
        super(GATNet, self).__init__()
        
        # Define the list of GATv2Conv layers
        self.gat_layers = nn.ModuleList(
            [GATv2Conv(input_dim if i == 0 else output_dim * heads, output_dim, heads, dropout=dropout) 
             for i in range(num_layers)]
        )
        
        # Down projection layer
        self.down_proj = nn.Linear(output_dim * heads, output_dim)
        
        # Merge layer
        self.merge = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x, edge_index):
        # Pass through each GATv2Conv layer
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
        
        # Apply down projection
        x = self.down_proj(x)
        
        return x
    
    def process_step_representations(self, encoder_output, step_indices, edge_index_batch):
        """
        Process step representations through GNN and merge with encoder output
        
        Args:
            encoder_output (torch.Tensor): Encoder outputs for batch
            step_indices (torch.Tensor): Step indices for each token
            step_graphs (List[Set[Tuple]]): Step graphs for each sample
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated encoder output and pooled GNN output
        """
        # Get step-level representations
        step_representations = get_step_reps(h=encoder_output, step_indices=step_indices)
        
        # Process each sample in the batch through GNN
        gnn_outputs = []
        for x, edge_index in zip(step_representations, edge_index_batch):
            if edge_index.numel() > 0:
                gnn_out = self(x, edge_index.to(x.device))
            else:
                gnn_out = torch.zeros(x.shape).to(x.device)
            gnn_outputs.append(gnn_out)
            
        # Update encoder outputs with GNN outputs
        for enc_out, gnn_out, step_idx in zip(encoder_output, gnn_outputs, step_indices):
            enc_out = enc_out.clone()
            step_idx = step_idx - 1  # Adjust for 1-indexing
            step_num = int(torch.max(step_idx).item())
            for i in range(step_num + 1):
                step_i_indices = torch.where(step_idx == i)[0]
                enc_out_step = enc_out.index_select(0, step_i_indices)
                gnn_out_step = gnn_out[i].unsqueeze(0).expand(enc_out_step.shape[0], -1)
                updated = F.relu(self.merge(torch.cat([enc_out_step, gnn_out_step], dim=-1)))
                enc_out = enc_out.index_copy(0, step_i_indices, updated)
                
        # Return pooled GNN output for downstream tasks
        gnn_out_pooled = gnn_out.mean(dim=0)
        
        return encoder_output, gnn_out_pooled