import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from debug.viz import save_heatmap, save_batch_heatmap
from typing import List, Dict, Any

class GraphRNNBilinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config['encoder_output_dim']
        self.M = config['graph_rnn_m']
        self.hidden_graph = config['graph_rnn_hidden_graph']
        self.hidden_edge = config['graph_rnn_hidden_edge']
        self.graph_l = config['graph_rnn_node_layers']
        self.edge_l = config['graph_rnn_edge_layers']
        self.tag_representation_dim = config['tag_representation_dim']
        self.bidirectional = False
        self.graph_rnn = nn.RNN(self.input_size,
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

        self.head_tag_feedforward = nn.Linear(self.input_size, self.tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(self.input_size, self.tag_representation_dim)
        self._dropout = nn.Dropout(config['tag_dropout'])

    def graph_pass(self, seq: torch.Tensor, batch_size: int):
        h_prev = torch.zeros((self.graph_l * (2 if self.bidirectional else 1),
                            batch_size,
                            self.hidden_graph)).to(self.config['device'])
        y, lhs = self.graph_rnn(seq, h_prev)
        return y, lhs

    def edge_pass(self, graph_state: torch.Tensor):
        B, S, D = graph_state.shape
        A = torch.zeros(B, S, self.M, device = self.config['device'])
        for i in range(S):
            h_prev = graph_state[:, i, :].unsqueeze(0)
            h_prev = h_prev.expand(self.edge_l, -1, -1).contiguous()
            h_prev = h_prev.to(self.config['device'])
            # self.M is simply how many edges i wanna generate
            # so that means how far back across S
            # i.e. i'm gonna produce edges to the nodes
            a = torch.zeros(self.M, device = self.config['device'])
            for j in range(self.M):
                x = A[:, i, :] # we need x to be a single node
                x = x.unsqueeze(1)
                out, h_prev = self.edge_rnn(x, h_prev)
                edge = self.edge_cls(out)
                edge = F.sigmoid(edge.view(-1))
                A[:, i, j] = edge 
        return A

    def reshape_adj(self, A):
        # A_seq already has the same number of elements
        # equal to the side of the full adj matrix
        B, S, M = A.shape
        A_reshaped = torch.zeros((B, S, S)).to(self.config['device'])
        for b in range(B):
            for i in range(S):
                shift = max(0, i-self.M)
                A_reshaped[b, i, shift:i] = A[b, i, :i]
        A_reshaped = A_reshaped.transpose(1, 2)
        return A_reshaped

    def forward(self, input: torch.Tensor,
                mask: torch.LongTensor,
                tag_embeddings: torch.Tensor,
                head_indices: torch.Tensor,
                head_tags: torch.Tensor,
                metadata: List[Dict[str, Any]] = [],
                ):
        # if self.config["tag_embedding_type"] != 'none':
        #     # tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
        #     input = torch.cat([input, tag_embeddings], dim=-1)

        # input dim [B, S, D]
        lengths = mask.sum(dim=-1).cpu()
        packed_input = pack_padded_sequence(input,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)
        packed_output, _ = self.graph_pass(packed_input, input.shape[0])
        graph_states, _ = pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=input.size(1))

        # during training the input into the edgeRNN
        # needs to be [B, S, M]
        A_batch = self.edge_pass(graph_states)
        # NOTE: maybe we don't need it because masking should already be handled in the decoder 
        # but here we could use lengths to zero out any adjacency predictions made for the padding
        arc_logits = self.reshape_adj(A_batch)

        head_tag = self._dropout(F.elu(self.head_tag_feedforward(graph_states)))
        dep_tag = self._dropout(F.elu(self.dep_tag_feedforward(graph_states)))

        output = {
            'head_tag': head_tag,
            'dep_tag': dep_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'arc_logits': arc_logits,
            'mask': mask,
            'metadata': metadata,
        }
        
        return output
    
class GraphRNNSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config['encoder_output_dim']
        self.M = config['graph_rnn_m']
        self.hidden_graph = config['graph_rnn_hidden_graph']
        self.hidden_edge = config['graph_rnn_hidden_edge']
        self.graph_l = config['graph_rnn_node_layers']
        self.edge_l = config['graph_rnn_edge_layers']
        self.bidirectional = False
        self.graph_rnn = nn.RNN(self.input_size,
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
        self.n_edge_labels = config['n_edge_labels']
        self.edge_cls = nn.Linear(self.hidden_edge, 1)

    def graph_pass(self, seq: torch.Tensor, batch_size: int):
        h_prev = torch.zeros((self.graph_l * (2 if self.bidirectional else 1),
                            batch_size,
                            self.hidden_graph)).to(self.config['device'])
        y, lhs = self.graph_rnn(seq, h_prev)
        return y, lhs

    def edge_pass(self, graph_state: torch.Tensor):
        B, S, D = graph_state.shape
        A = torch.zeros(B, S, self.M, device = self.config['device'])
        for i in range(S):
            h_prev = graph_state[:, i, :].unsqueeze(0)
            h_prev = h_prev.expand(self.edge_l, -1, -1).contiguous()
            h_prev = h_prev.to(self.config['device'])
            # self.M is simply how many edges i wanna generate
            # so that means how far back across S
            # i.e. i'm gonna produce edges to the nodes
            a = torch.zeros(self.M, device = self.config['device'])
            for j in range(self.M):
                x = A[:, i, :] # x needs to be a single node
                x = x.unsqueeze(1) # x needs to be considered a sequence of 1 element
                out, h_prev = self.edge_rnn(x, h_prev)
                edge = self.edge_cls(out)
                edge = F.sigmoid(edge.view(-1)) # assign the batch of predictions
                A[:, i, j] = edge 
        return A

    def reshape_adj(self, A):
        # A_seq already has the same number of elements
        # equal to the side of the full adj matrix
        B, S, M = A.shape
        A_reshaped = torch.zeros((B, S, S)).to(self.config['device'])
        for b in range(B):
            for i in range(S):
                shift = max(0, i-self.M)
                A_reshaped[b, i, shift:i] = A[b, i, :i]
        A_reshaped = A_reshaped.transpose(1, 2)
        return A_reshaped

    def forward(self, input: torch.Tensor, mask: torch.LongTensor):
        # input dim [B, S, D]
        lengths = mask.sum(dim=-1).cpu()
        packed_input = pack_padded_sequence(input,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)
        packed_output, _ = self.graph_pass(packed_input, input.shape[0])
        graph_states, _ = pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=input.size(1))

        # during training the input into the edgeRNN
        # needs to be [B, S, M]
        A_batch = self.edge_pass(graph_states)
        # NOTE: maybe we don't need it because masking should already be handled in the decoder 
        # but here we could use lengths to zero out any adjacency predictions made for the padding
        A_reshaped = self.reshape_adj(A_batch)
        return {'lhs': graph_states,
                'adj': A_reshaped,
                }