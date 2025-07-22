import torch
from model.utils.io_utils import load_json
from model.utils.nn import adjust_for_sentinel, adj_indices_to_adj_matrix
import matplotlib.pyplot as plt

def make_adj_sequence(adj_square: torch.LongTensor, M: int = 20):
    B, S, S = adj_square.shape
    L = min(S, M)
    adj_seq = torch.zeros(B, S, L+1, 1)
    adj_seq[:, :, 0, 0] = 1

    for i in range(S):
        start = max(0, i-M)
        row = adj_square[:, i, start:i]
        length = row.shape[-1]
        adj_seq[:, i, 1:1+length, 0] = row
    adj_seq = adj_seq.reshape(B*S, L+1, 1)
    return adj_seq

def reshape_adj(A: torch.Tensor, M: int):
    # slice out the BOS scalars
    A = A[:, 1:, :]
    # A_seq already has the same number of elements
    # equal to the side of the full adj matrix
    B, S, _ = A.shape
    A_reshaped = torch.zeros((B, S, S)).to(A.device)
    for i in range(S):
        start = max(0, i-M)
        length  = i - start
        # here we don't slice [1:1+length]
        # like in `make_adj_sequence`
        # because we removed the BOS scalars 
        row = A[:, i, :length, 0]
        A_reshaped[:, i, start:i] = row
    # A_reshaped = A_reshaped.transpose(1, 2)
    return A_reshaped

def main():
    dataset = load_json('./data/ade/train.json')
    head_list = [el['head_indices'] for el in dataset[5:6]]
    head_indices = torch.tensor(head_list)
    mask, head_indices, head_tags = adjust_for_sentinel(None, head_indices, None)
    print(head_indices, head_indices.shape)
    adj_square = adj_indices_to_adj_matrix(head_indices).long()
    print(adj_square, adj_square.shape)
    # adj_square = adj_square[:1, :, :]
    print(adj_square[0, 4, 1])
    adj_seq = make_adj_sequence(adj_square)  # shape (B*S, M+1, 1)
    adj_seq = adj_seq.squeeze(-1)
    print(adj_seq.long(), adj_seq.shape)

if __name__ == "__main__":
    main()

