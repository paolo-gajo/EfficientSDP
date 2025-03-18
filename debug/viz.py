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
        matrix = matrix.clone().detach().cpu().numpy()  # Convert PyTorch tensor to NumPy

    # if matrix.shape[0] != matrix.shape[1]:
    #     raise ValueError("Input matrix must be square.")

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
