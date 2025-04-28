import torch
import torch.nn as nn
from model.gnn import pad_square_matrix
from typing import List, Dict
# we want to take the eigenvectors of the graph laplacian
# and pass them through a linear layer [k, d]
# and sum that with each representation of the index
# corresponding to that node index i \in 1, ..., k

class LaplacePE(torch.nn.Module):
    def __init__(self, embedding_dim: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.lap_proj = nn.Linear(self.max_steps, embedding_dim)
        self.lap_norm = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, input: torch.Tensor, graph_laplacian: List[torch.Tensor], step_indices: torch.Tensor):
        for i, (L, step_idx) in enumerate(zip(graph_laplacian, step_indices)):
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            eigenvectors = pad_square_matrix(eigenvectors, self.max_steps)#[:L.shape[0], :]
            eigen_pad = torch.zeros(self.max_steps).unsqueeze(0).to(eigenvectors.device)
            # we need `eigen_pad` because we need to be able to index the 0 vector for the indices that are 0
            eigenvectors = torch.cat([eigen_pad, eigenvectors], dim = 0)
            laplacian_pe = self.lap_proj(eigenvectors)
            step_idx_monitor = step_idx.clone().detach().cpu()
            laplacian_pe_extended = laplacian_pe[step_idx, :]
            # save_heatmap(laplacian_pe_extended, 'lap_pe.pdf')
            input[i] = input[i] + laplacian_pe_extended
            ...
        return self.lap_norm(input.transpose(1, 2)).transpose(1, 2)
        ...

