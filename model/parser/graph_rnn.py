import torch
import torch.nn as nn
import torch.nn.functional as F
from debug.viz import save_heatmap
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # input_size: dim of input vector x_t
        # hidden_size: dim of hidden state h_t
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))   # input-to-hidden
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))  # hidden-to-hidden
        self.b = nn.Parameter(torch.zeros(hidden_size))               # bias

    def forward(self, input: torch.Tensor, last_hidden: torch.Tensor):
        # input: (input_size,)
        # last_hidden: (hidden_size,)
        Wx = self.W @ input             # shape: (hidden_size,)
        Uh = self.U @ last_hidden  # shape: (hidden_size,)
        return F.tanh(Wx + Uh + self.b)

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

        self.graph_cell = nn.ModuleList(
            [RNNCell(self.g_in_size, self.hidden_graph)] +\
            [RNNCell(self.hidden_graph, self.hidden_graph) for _ in range(self.graph_l - 1)]
            )
        self.edge_cell = nn.ModuleList(
            [RNNCell(self.M, self.hidden_edge)] +\
            [RNNCell(self.hidden_edge, self.hidden_edge) for _ in range(self.edge_l - 1)]
            )
        self.edge_cls = nn.Linear(self.hidden_edge, 1)

    def graph_pass(self, seq: torch.Tensor):
        h_prev = torch.zeros((self.graph_l, self.hidden_graph)).to(self.config['device'])
        H_out = []
        for i in range(seq.shape[0]):
            x = seq[i]
            lhs = []
            for l, cell in enumerate(self.graph_cell):
                x = cell(x, h_prev[l])
                lhs.append(x)
            h_prev = torch.stack(lhs)
            H_out.append(x) 
        H_out = torch.stack(H_out)
        return H_out

    def edge_pass(self, graph_state: torch.Tensor, index: int, lengths: torch.LongTensor):
        h_prev = [graph_state] + [torch.zeros_like(graph_state) for _ in range(self.edge_l - 1)]
        h_prev = torch.stack(h_prev).to(self.config['device'])
        A = torch.zeros(self.M).to(self.config['device'])
        m = min(index, self.M)
        for j in range(m):
            x = A
            lhs = []
            for l, cell in enumerate(self.edge_cell):
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
        lengths = mask.sum(dim=-1)
        for b, batch in enumerate(input):
            H = self.graph_pass(batch)
            A_seq = []
            for i in range(H.shape[0]):
                A = self.edge_pass(H[i], i, lengths[b])
                A_seq.append(A)
            A_reshaped = self.reshape_adj(A_seq)
            H_batch.append(H)
            A_batch.append(A_reshaped)
        H_batch = torch.stack(H_batch)
        A_batch = torch.stack(A_batch)
        return {'lhs': H_batch,
                'adj': A_batch,
                }