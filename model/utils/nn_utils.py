import torch
from typing import Set, Tuple
from torch_geometric.utils import to_dense_adj

def prepend_ones(batch_size, mask, head_indices, head_tags):
    mask_ones = mask.new_ones(batch_size, 1)
    mask = torch.cat([mask_ones, mask], dim = 1)
    
    if head_indices is not None:
        head_indices = torch.cat(
            [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
        )
    if head_tags is not None:
        head_tags = torch.cat(
            [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
        )
    return mask, head_indices, head_tags

def adj_indices_to_adj_matrix(targets: torch.LongTensor):
    B, S = targets.shape
    indices = torch.arange(S).repeat(B, 1).to(targets.device)
    edge_index = torch.stack([indices, targets])
    adj_batch_list = []
    for b in range(B):
        adj = to_dense_adj(edge_index[:, b, :]).squeeze()
        adj_batch_list.append(adj)
    adj_batch = torch.stack(adj_batch_list)
    return adj_batch

def graph_to_edge_index(graph: Set[Tuple]):
    if not (len(graph)) > 0:
        return torch.empty(0)
    return torch.tensor(list(graph), dtype=torch.long).T

def edge_index_to_adj_matrix(edge_index: torch.Tensor):
    if not edge_index.numel() > 0:
        return edge_index
    edge_index = edge_index
    k = torch.max(edge_index) + 1
    adj = torch.zeros((k, k), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    return adj

def get_deg_matrix(adj_matrix: torch.Tensor):
    N = adj_matrix.shape[0]
    degs = []
    for i in range(N):
        degs.append(sum(adj_matrix[i]))
    deg_matrix = torch.diag(torch.tensor(degs))
    return deg_matrix

def get_graph_laplacian(deg_m: torch.Tensor, adj_m: torch.Tensor):
    L = deg_m - adj_m
    return L