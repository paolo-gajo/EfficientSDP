import torch
import torch.nn as nn
import sys
from torch_geometric.nn import GCNConv, GATConv
sys.path.append('./debug')
from viz import save_heatmap

class GraphNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, conv_type="gcn", heads=1):
        super().__init__()
        
        # Define the list of graph convolution layers based on the selected type
        self.layers = nn.ModuleList()
        
        if conv_type.lower() == "gcn":
            self.layers = nn.ModuleList(
                [GCNConv(input_dim if i == 0 else output_dim, output_dim) for i in range(num_layers)]
            )
        elif conv_type.lower() == "gat":
            # For GAT, we need to consider the heads
            # First layer
            self.layers.append(GATConv(input_dim, output_dim // heads, heads=heads))
            
            # Middle layers
            for _ in range(1, num_layers):
                self.layers.append(GATConv(output_dim, output_dim // heads, heads=heads))
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}. Choose 'gcn' or 'gat'.")
            
        self.conv_type = conv_type

    def forward(self, x, edge_index):
        # Handle the case where edge_index is empty
        if edge_index.numel() == 0:
            # Create a self-loop for each node
            num_nodes = x.size(0)
            edge_index = torch.arange(0, num_nodes, device=x.device)
            edge_index = torch.stack([edge_index, edge_index], dim=0)
        
        # Pass through each layer
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        return x
    
class DGM_Layer(nn.Module):
    def __init__(self, config, input_dim, output_dim=None, distance="euclidean", conv_type="gcn", heads=1, num_gnn_layers=1):
        super(DGM_Layer, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Use the flexible GraphNet for both embed_f and diffusion
        self.embed_f = GraphNet(input_dim, self.output_dim, num_layers=self.num_gnn_layers, conv_type=conv_type, heads=heads)
        self.diffusion = GraphNet(self.output_dim, self.output_dim, num_layers=self.num_gnn_layers, conv_type=conv_type, heads=heads)
        
        self.distance = distance
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.conv_type = conv_type
        self.apply_diffusion = False  # Flag to control whether to apply diffusion

    def forward(self, x, A=None):
        device = x.device
        batch_size, num_nodes = x.shape[0], x.shape[1]
        
        if A is None:
            A = torch.zeros((batch_size, num_nodes, num_nodes), device=device)
        
        # Convert adjacency matrix to edge index
        edge_index, batch = self.adj_to_edge_index(A)
        edge_index = edge_index.to(device)
        
        # Reshape x for processing
        # From [batch_size, num_nodes, feature_dim] to [batch_size * num_nodes, feature_dim]
        x_flat = x.reshape(-1, x.size(-1))
        
        # Estimate normalization parameters
        centroid = x_flat.mean(0, keepdim=True).detach()
        scale = (0.9 / (x_flat - centroid).abs().max()).detach()
        
        # Apply first graph network
        x_flat = self.embed_f(x_flat, edge_index)
        
        # Reshape back to [batch_size, num_nodes, feature_dim]
        x_reshaped = x_flat.reshape(batch_size, num_nodes, -1)
        
        # Calculate distances
        if self.distance == "hyperbolic":
            D, _x = pairwise_poincare_distances(x_reshaped)
        else:
            x_normalized = scale * (x_flat - centroid)
            x_normalized = x_normalized.reshape(batch_size, num_nodes, -1)
            D, _x = pairwise_euclidean_distances(x_normalized)
        
        # Calculate adjacency matrix
        A = torch.sigmoid(self.temperature * (self.threshold - D))
        
        # Mask self-connections
        mask_values = torch.full((num_nodes,), float('-inf'), device=device)
        mask = torch.diag_embed(mask_values).unsqueeze(0).expand(batch_size, -1, -1)
        A = A + mask
        A = torch.clamp(A, min=0)
        
        # Convert to edge index format
        new_edge_index, batch_indices = self.adj_to_edge_index(A)
        new_edge_index = new_edge_index.to(device)
        
        # Apply diffusion if enabled
        if self.apply_diffusion and new_edge_index.numel() > 0:
            x_flat = self.diffusion(x_flat, new_edge_index)
            x_reshaped = x_flat.reshape(batch_size, num_nodes, -1)
        
        return {
            "x": x_reshaped,  # Shape: [batch_size, num_nodes, feature_dim]
            "adj": A,         # Shape: [batch_size, num_nodes, num_nodes]
            "edge_index": new_edge_index
        }

    def adj_to_edge_index(self, adj):
        """Convert adjacency matrix to edge index format with batch support
        
        Args:
            adj: Adjacency matrix [B, N, N] or [N, N]
                
        Returns:
            edge_index: Edge index [2, E] with proper batch handling
            batch: Batch indices for each edge [E]
        """
        device = adj.device
        
        # Check if the adjacency matrix is batched
        if adj.dim() == 3:  # [B, N, N]
            batch_size, num_nodes = adj.size(0), adj.size(1)
            
            # Initialize edge indices and batch indices
            edge_indices_list = []
            batch_indices_list = []
            
            # Process each batch
            for b in range(batch_size):
                # Find edges in this batch
                edges = torch.nonzero(adj[b] > 0.5, as_tuple=True)
                
                if len(edges[0]) > 0:
                    # Get source and destination nodes
                    src, dst = edges
                    
                    # Adjust node indices for this batch
                    src_adjusted = src + b * num_nodes
                    dst_adjusted = dst + b * num_nodes
                    
                    # Stack source and destination nodes
                    batch_edge_index = torch.stack([src_adjusted, dst_adjusted], dim=0)
                    edge_indices_list.append(batch_edge_index)
                    
                    # Store batch indices for these edges
                    batch_indices_list.append(torch.full((len(src),), b, dtype=torch.long, device=device))
            
            # If no edges found in any batch
            if len(edge_indices_list) == 0:
                # Return empty tensors with correct shape
                return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
            
            # Concatenate all edge indices and batch indices
            edge_index = torch.cat(edge_indices_list, dim=1)
            batch_indices = torch.cat(batch_indices_list)
            
            return edge_index, batch_indices
        else:  # [N, N] (single adjacency matrix)
            # Get indices where adj > 0
            edges = torch.nonzero(adj > 0.5, as_tuple=True)
            
            # If no edges
            if len(edges[0]) == 0:
                # Return empty tensor with correct shape
                return torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
            
            # Create edge index
            edge_index = torch.stack(edges, dim=0)
            # Single batch, all zeros
            batch_indices = torch.zeros((edge_index.size(1),), dtype=torch.long, device=device)
            
            return edge_index, batch_indices

class DGM_c(nn.Module):
    def __init__(self, config, input_dim, hidden_dims=None, num_layers=1, num_gnn_layers=1, distance="euclidean", 
                 conv_type="gcn", heads=1, apply_diffusion=False):
        super(DGM_c, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_gnn_layers = num_gnn_layers
        
        # Set up dimensions for each layer
        if hidden_dims is None:
            hidden_dims = [input_dim] * num_layers
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers
        
        # Ensure hidden_dims has the right length
        assert len(hidden_dims) == num_layers, "hidden_dims must have length equal to num_layers"
        
        # Create multiple DGM_Layer instances
        self.layers = nn.ModuleList()
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dims[i]
            layer = DGM_Layer(
                config=config,
                input_dim=in_dim,
                output_dim=out_dim,
                distance=distance,
                conv_type=conv_type,
                heads=heads,
                num_gnn_layers=num_gnn_layers,
            )
            layer.apply_diffusion = apply_diffusion
            self.layers.append(layer)
            in_dim = out_dim
    
    def forward(self, x, A=None):
        out_list = []
        adj_list = []
        
        # Initialize with input
        current_x = x
        current_A = A
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            out_dict = layer(current_x, current_A)
            current_x = out_dict["x"]
            current_A = out_dict["adj"]
            
            out_list.append(out_dict["x"])
            adj_list.append(out_dict["adj"])
        
        # Return results from all layers and the final layer
        return {
            "x": current_x,  # Final output
            "adj": current_A,  # Final adjacency
            "x_all": out_list,  # All intermediate outputs
            "adj_all": adj_list  # All intermediate adjacencies
        }

# Euclidean distance
def pairwise_euclidean_distances(x):
    """
    Calculate pairwise Euclidean distances between nodes in each batch
    
    Args:
        x: Node features [B, N, D]
        
    Returns:
        dist: Distance matrix [B, N, N]
    """
    batch_size, num_nodes, dim = x.shape
    
    # Reshape for batch processing
    x_expanded_1 = x.unsqueeze(2)  # [B, N, 1, D]
    x_expanded_2 = x.unsqueeze(1)  # [B, 1, N, D]
    
    # Calculate squared differences
    diff = x_expanded_1 - x_expanded_2  # [B, N, N, D]
    dist = torch.sum(diff ** 2, dim=-1)  # [B, N, N]
    
    return dist, x

# Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    """
    Calculate pairwise Poincaré distances between nodes in each batch
    
    Args:
        x: Node features [B, N, D]
        dim: Dimension to sum over (-1 for feature dimension)
        
    Returns:
        dist: Distance matrix [B, N, N]
    """
    batch_size, num_nodes, feature_dim = x.shape
    
    # Compute norms
    x_norm = torch.sum(x ** 2, dim=dim, keepdim=True)  # [B, N, 1]
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    
    # Normalize features
    x = x / (x_norm * (1 + 1e-2))
    
    # Recompute norms after normalization
    x_norm = torch.sum(x ** 2, dim=dim, keepdim=True)  # [B, N, 1]
    
    # Expand dimensions for broadcasting
    x_expanded_1 = x.unsqueeze(2)  # [B, N, 1, D]
    x_expanded_2 = x.unsqueeze(1)  # [B, 1, N, D]
    
    # Calculate squared differences
    diff = x_expanded_1 - x_expanded_2  # [B, N, N, D]
    pq = torch.sum(diff ** 2, dim=-1)  # [B, N, N]
    
    # Prepare norm products for denominator
    norm_1 = x_norm  # [B, N, 1]
    norm_2 = x_norm.transpose(1, 2)  # [B, 1, N]
    
    # Calculate distance using hyperbolic formula
    denominator = (1 - norm_1) * (1 - norm_2)  # [B, N, N]
    dist = torch.arccosh(1e-6 + 1 + 2 * pq / denominator) ** 2
    
    return dist, x