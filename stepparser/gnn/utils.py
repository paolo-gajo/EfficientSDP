import torch
from typing import List, Set, Tuple

def get_step_reps(
        h: torch.Tensor, step_indices: torch.Tensor,
    ):
        """
        Compute step-level representations by averaging token representations for each step.

        Args:
            h (torch.Tensor): Tensor of shape [batch, seqlen, dim] containing token-level representations.
            step_indices (torch.Tensor): Tensor of shape [batch, seqlen] with integer step indices for each token.

        Returns:
            list[torch.Tensor]: List of length batch where each element is a tensor of shape [num_steps, dim].
        """
        batch_step_reps = []
        # Iterate over each sample in the batch
        for sample_reps, sample_steps in zip(h, step_indices):
            # Get unique step indices in sorted order.
            # (If you want to ignore a particular step (e.g. index 0), you can filter it out here.)
            unique_steps = torch.unique(
                sample_steps[torch.where(sample_steps != 0)[0]], sorted=True
            )

            x = []
            for step in unique_steps:
                # Create a boolean mask for tokens corresponding to this step.
                mask = sample_steps == step
                # Average over the tokens for this step along the sequence length dimension.
                # This produces a representation with shape [dim].
                rep = sample_reps[mask].mean(dim=0)
                x.append(rep)

            # Stack step representations to obtain a tensor of shape [num_steps, dim] for this sample.
            x = torch.stack(x, dim=0)
            batch_step_reps.append(x)
        return batch_step_reps

def pad_square_matrix(matrix: torch.Tensor, k: int):
    """Pads a square matrix to size (k, k) with zeros."""
    n = matrix.shape[0]
    if n >= k:
        return matrix[:k, :k]  # Trim if larger
    
    # Create a zero matrix of size (k, k)
    padded_matrix = torch.zeros((k, k), dtype=matrix.dtype, device=matrix.device)
    
    # Copy the original matrix into the top-left corner
    padded_matrix[:n, :n] = matrix
    return padded_matrix