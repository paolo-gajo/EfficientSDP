import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from debug.viz import save_heatmap

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         # input_size: dim of input vector x_t
#         # hidden_size: dim of hidden state h_t
#         self.W = nn.Parameter(torch.randn(hidden_size, input_size))   # input-to-hidden
#         self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))  # hidden-to-hidden
#         self.b = nn.Parameter(torch.zeros(hidden_size))               # bias

#     def forward(self, input: torch.Tensor, last_hidden: torch.Tensor):
#         # input: (input_size,)
#         # last_hidden: (hidden_size,)
#         Wx = self.W @ input             # shape: (hidden_size,)
#         Uh = self.U @ last_hidden  # shape: (hidden_size,)
#         return F.tanh(Wx + Uh + self.b)

class GraphRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.g_in_size = config['encoder_output_dim']
        self.M = config['graph_rnn_m']
        self.hidden_graph = config['graph_rnn_hidden_graph']
        self.hidden_edge = config['graph_rnn_hidden_edge']
        self.graph_l = config['graph_rnn_node_layers']
        self.edge_l = config['graph_rnn_edge_layers']
        self.bidirectional = False
        self.graph_rnn = nn.RNN(self.g_in_size,
                                self.hidden_graph,
                                num_layers=self.graph_l,
                                batch_first=True,
                                bidirectional=self.bidirectional,
                                dropout=0,
                )
        self.edge_rnn = nn.RNN(self.M,
                                self.hidden_edge,
                                num_layers=self.edge_l,
                                batch_first=True,
                                bidirectional=self.bidirectional,
                                dropout=0,
                )
        self.edge_cls = nn.Linear(self.hidden_edge, 1)

    def graph_pass(self, seq: torch.Tensor, batch_size: int):
        h_prev = torch.zeros((self.graph_l * (2 if self.bidirectional else 1),
                            batch_size,
                            self.hidden_graph)).to(self.config['device'])
        # H_out = []
        # for i in range(seq.shape[0]):
        #     x = seq[i]
        #     lhs = []
        #     for l, cell in enumerate(self.graph_rnn):
        #         lhs.append(x)
        #     h_prev = torch.stack(lhs)
        #     H_out.append(x) 
        # H_out = torch.stack(H_out)
        y, lhs = self.graph_rnn(seq, h_prev)
        return y, lhs

    def edge_pass(self, graph_state: torch.Tensor, index: int, lengths: torch.LongTensor):
        h_prev = [graph_state] + [torch.zeros_like(graph_state) for _ in range(self.edge_l - 1)]
        h_prev = torch.stack(h_prev).to(self.config['device'])
        A = torch.zeros(self.M).to(self.config['device'])
        m = min(index, self.M)
        for j in range(m):
            x = A
            lhs = []
            for l, cell in enumerate(self.edge_rnn):
                x = cell(x, h_prev[l])
                lhs.append(x)
            s = self.edge_cls(x)
            A[j] = F.sigmoid(s)
            h_prev = torch.stack(lhs)
        return A

    def reshape_adj(self, A):
        # A_seq already has the same number of elements
        # equal to the side of the full adj matrix
        S = len(A)
        A_reshaped = torch.zeros((S, S)).to(self.config['device'])
        for i in range(S):
            shift = max(0, i-self.M)
            A_reshaped[i][shift:i] = A[i][:i]
        A_reshaped = A_reshaped.transpose(0, 1)
        return A_reshaped

    def forward(self, input: torch.Tensor, mask: torch.LongTensor):
        # input dim [B, S, D]
        H_batch = []
        A_batch = []
        lengths = mask.sum(dim=-1).cpu()
        packed_input = pack_padded_sequence(input,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)
        packed_output, _ = self.graph_pass(packed_input, input.shape[0])
        Y, _ = pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=input.size(1))

        for b, batch in enumerate(Y):
            A_seq = []
            for i in range(Y.shape[0]):
                A = self.edge_pass(Y[i], i, lengths[b])
                A_seq.append(A)
            A_reshaped = self.reshape_adj(A_seq)
            H_batch.append(Y)
            A_batch.append(A_reshaped)
        H_batch = torch.stack(H_batch)
        A_batch = torch.stack(A_batch)
        return {'lhs': H_batch,
                'adj': A_batch,
                }