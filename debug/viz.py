import matplotlib.pyplot as plt
import torch
import seaborn as sns

def save_heatmap(matrix, filename="heatmap.pdf", cmap="viridis"):
    """
    Saves a heatmap of a square matrix as a PDF.

    Parameters:
    - matrix (torch.Tensor or np.ndarray): Square matrix to visualize.
    - filename (str): Output filename (default: "heatmap.pdf").
    - cmap (str): Matplotlib colormap (default: "viridis").
    """

    if isinstance(matrix, torch.Tensor):
        if hasattr(matrix, 'layout'):
            if matrix.layout == torch.sparse_coo:
                matrix = matrix.to_dense()
        matrix = matrix.clone().detach().cpu().numpy()  # Convert PyTorch tensor to NumPy

    # if matrix.shape[0] != matrix.shape[1]:
    #     raise ValueError("Input matrix must be square.")
    if len(matrix.shape) > 2:
        matrix = matrix[0]

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.title("Heatmap")
    
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def show_attentions(model_out):
    # Extract attention matrices (Shape: num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = model_out.attentions

    # Get attention maps from the last layer
    last_layer_attentions = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
    num_heads = last_layer_attentions.shape[0]

    # Plot heatmaps for each attention head in the last layer
    fig, ax = plt.subplots(figsize=(20, 15))  # 12 heads (4 rows × 3 cols)

    sns.heatmap(last_layer_attentions[0].detach().cpu().numpy(), ax=ax, cmap="Blues")

    plt.tight_layout()
    plt.savefig('model_attentions.pdf', format='pdf')

import torch
import numpy as np
import matplotlib.pyplot as plt

def save_batch_heatmap(matrices, filename="batch_heatmap.pdf", cmap="viridis", titles=None):
    """
    Saves heatmaps of a batch of matrices as subplots in a single PDF.
    
    Parameters:
    - matrices (list of torch.Tensor/np.ndarray or single tensor with batch dim): 
      Batch of matrices to visualize. Can be a list of 2D tensors or a 3D tensor 
      with shape (N, H, W) where N is batch size.
    - filename (str): Output filename (default: "batch_heatmap.pdf").
    - cmap (str): Matplotlib colormap (default: "viridis").
    - titles (list of str, optional): List of titles for each subplot. If None,
      uses "Matrix 0", "Matrix 1", etc.
    """
    
    # Handle input format - convert to list of 2D arrays
    processed_matrices = []
    
    if isinstance(matrices, (list, tuple)):
        # Already a list/tuple of matrices
        for i, matrix in enumerate(matrices):
            processed_matrices.append(_process_single_matrix(matrix))
    else:
        # Single tensor with batch dimension
        if isinstance(matrices, torch.Tensor):
            if hasattr(matrices, 'layout') and matrices.layout == torch.sparse_coo:
                matrices = matrices.to_dense()
            matrices = matrices.clone().detach().cpu().numpy()
        
        # Handle different input shapes
        if len(matrices.shape) == 3:
            # Shape (N, H, W) - batch of 2D matrices
            for i in range(matrices.shape[0]):
                processed_matrices.append(matrices[i])
        elif len(matrices.shape) == 4:
            # Shape (N, C, H, W) or similar - take first channel/element
            for i in range(matrices.shape[0]):
                processed_matrices.append(matrices[i, 0])
        else:
            raise ValueError(f"Unsupported input shape: {matrices.shape}")
    
    N = len(processed_matrices)
    if N == 0:
        raise ValueError("No matrices to plot")
    
    # Calculate subplot grid dimensions
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle case where we only have one subplot
    if N == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each matrix
    for i in range(N):
        im = axes[i].imshow(processed_matrices[i], cmap=cmap, aspect="auto")
        
        # Set title
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        else:
            axes[i].set_title(f"Matrix {i}")
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=axes[i])
    
    # Hide empty subplots
    for i in range(N, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()

def _process_single_matrix(matrix):
    """Helper function to process a single matrix to numpy array format."""
    if isinstance(matrix, torch.Tensor):
        if hasattr(matrix, 'layout') and matrix.layout == torch.sparse_coo:
            matrix = matrix.to_dense()
        matrix = matrix.clone().detach().cpu().numpy()
    
    # Handle extra dimensions
    if len(matrix.shape) > 2:
        matrix = matrix[0]  # Take first element if there are extra dims
    
    return matrix

def indices_to_adjacency_matrices(indices_batch, max_node=None, symmetric=False, self_loops=False):
    """
    Converts a batch of index sequences to adjacency matrices.
    
    Parameters:
    - indices_batch (torch.Tensor): Shape (batch_size, sequence_length)
      Each row contains a sequence of node indices representing edges or paths
    - max_node (int, optional): Maximum node index + 1 (size of adjacency matrix).
      If None, inferred from the maximum value in indices_batch
    - symmetric (bool): If True, makes adjacency matrices symmetric (undirected graph)
    - self_loops (bool): If True, allows self-loops (diagonal entries)
    
    Returns:
    - torch.Tensor: Shape (batch_size, max_node, max_node) - batch of adjacency matrices
    """
    
    batch_size, seq_len = indices_batch.shape
    device = indices_batch.device
    
    # Determine matrix size
    if max_node is None:
        max_node = int(indices_batch.max().item()) + 1
    
    # Initialize adjacency matrices (all zeros)
    adj_matrices = torch.zeros(batch_size, max_node, max_node, 
                              dtype=torch.float32, device=device)
    
    # Process each sequence in the batch
    for batch_idx in range(batch_size):
        sequence = indices_batch[batch_idx]
        
        # Remove padding zeros from the end (assuming 0 is padding)
        # Find last non-zero element
        non_zero_mask = sequence != 0
        if non_zero_mask.any():
            last_valid_idx = non_zero_mask.nonzero(as_tuple=True)[0][-1].item() + 1
            valid_sequence = sequence[:last_valid_idx]
        else:
            # Handle case where sequence starts with non-zero
            # Keep all elements or find actual sequence end
            valid_sequence = sequence
        
        # Create edges between consecutive nodes in the sequence
        for i in range(len(valid_sequence) - 1):
            node1 = valid_sequence[i].item()
            node2 = valid_sequence[i + 1].item()
            
            # Skip if either node exceeds max_node
            if node1 >= max_node or node2 >= max_node:
                continue
                
            # Add edge
            adj_matrices[batch_idx, node1, node2] = 1.0
            
            # Add reverse edge if symmetric
            if symmetric:
                adj_matrices[batch_idx, node2, node1] = 1.0
        
        # Add self-loops if requested
        if self_loops:
            unique_nodes = torch.unique(valid_sequence)
            for node in unique_nodes:
                node = node.item()
                if node < max_node:
                    adj_matrices[batch_idx, node, node] = 1.0
    
    return adj_matrices

def indices_to_adjacency_matrices_advanced(indices_batch, max_node=None, 
                                         edge_weights=None, padding_value=0):
    """
    Advanced version with more options for edge creation.
    
    Parameters:
    - indices_batch (torch.Tensor): Shape (batch_size, sequence_length)
    - max_node (int, optional): Maximum node index + 1
    - edge_weights (torch.Tensor, optional): Shape (batch_size, sequence_length-1)
      Weights for edges between consecutive nodes
    - padding_value (int): Value used for padding (default: 0)
    
    Returns:
    - torch.Tensor: Shape (batch_size, max_node, max_node) - weighted adjacency matrices
    """
    
    batch_size, seq_len = indices_batch.shape
    device = indices_batch.device
    
    if max_node is None:
        # Filter out padding values when finding max
        non_padding_mask = indices_batch != padding_value
        if non_padding_mask.any():
            max_node = int(indices_batch[non_padding_mask].max().item()) + 1
        else:
            max_node = 1  # Default minimum size
    
    adj_matrices = torch.zeros(batch_size, max_node, max_node, 
                              dtype=torch.float32, device=device)
    
    for batch_idx in range(batch_size):
        sequence = indices_batch[batch_idx]
        
        # Find valid sequence length (exclude padding)
        non_padding_mask = sequence != padding_value
        if non_padding_mask.any():
            valid_indices = non_padding_mask.nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                # Handle case where padding might be in the middle
                last_valid_idx = valid_indices[-1].item() + 1
                valid_sequence = sequence[:last_valid_idx]
                
                # Remove internal padding if any
                valid_sequence = valid_sequence[valid_sequence != padding_value]
            else:
                continue
        else:
            continue
        
        # Create edges
        for i in range(len(valid_sequence) - 1):
            node1 = valid_sequence[i].item()
            node2 = valid_sequence[i + 1].item()
            
            if node1 >= max_node or node2 >= max_node:
                continue
            
            # Determine edge weight
            if edge_weights is not None and i < edge_weights.shape[1]:
                weight = edge_weights[batch_idx, i].item()
            else:
                weight = 1.0
            
            adj_matrices[batch_idx, node1, node2] = weight
    
    return adj_matrices

# Example usage:
if __name__ == "__main__":
    # Example with list of tensors
    batch_tensors = [torch.randn(10, 10) for _ in range(6)]
    save_batch_heatmap(batch_tensors, "example_batch.pdf", titles=[f"Random Matrix {i+1}" for i in range(6)])
    
    # Example with single batched tensor
    batched_tensor = torch.randn(4, 8, 8)
    save_batch_heatmap(batched_tensor, "example_batched.pdf")

