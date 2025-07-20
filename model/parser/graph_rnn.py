import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from debug.viz import save_heatmap, save_batch_heatmap
from model.utils.nn_utils import adj_indices_to_adj_matrix, prepend_ones
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
        self.edge_rnn = nn.RNN(1,
                                self.hidden_edge,
                                num_layers=self.edge_l,
                                batch_first=True,
                                bidirectional=self.bidirectional,
                                dropout=0,
                )
        self.edge_cls = nn.Linear(self.hidden_edge, 1)

        self._head_sentinel = torch.nn.Parameter(torch.randn(self.input_size))

        self.head_tag_feedforward = nn.Linear(self.hidden_graph, self.tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(self.hidden_graph, self.tag_representation_dim)
        self._dropout = nn.Dropout(config['tag_dropout'])

    def forward(self, input: torch.Tensor,
                mask: torch.LongTensor,
                tag_embeddings: torch.Tensor,
                head_indices: torch.Tensor,
                head_tags: torch.Tensor,
                mode: str,
                metadata: List[Dict[str, Any]] = [],
                ):
        # if self.config["tag_embedding_type"] != 'none':
        #     # tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
        #     input = torch.cat([input, tag_embeddings], dim=-1)
        
        
        # input dim [B, S, D]
        B, _, D = input.shape
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(B, 1, D)
        input = torch.cat([head_sentinel, input], dim=1)
        mask, head_indices, head_tags = prepend_ones(B, mask, head_indices, head_tags)
        _, S, _ = input.shape
        lengths = mask.sum(dim=-1).cpu()
        packed_input = pack_padded_sequence(input,
                                            lengths,
                                            batch_first=True,
                                            enforce_sorted=False)
        packed_output, _ = self.graph_pass(packed_input, input.shape[0])
        graph_state, _ = pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=input.size(1))

        # during training the input into the edgeRNN
        # needs to be [B, S, M]
        if mode == 'train':
            A_batch = self.edge_pass_train(graph_state=graph_state, head_indices=head_indices)
        else:
            A_batch = self.edge_pass_test(graph_state=graph_state)

        # NOTE: maybe we don't need it because masking should already be handled in the decoder 
        # but here we could use lengths to zero out any adjacency predictions made for the padding
        A_batch = A_batch.reshape(B, S, self.M+1, 1)
        arc_logits = self.reshape_adj(A_batch)

        head_tag = self._dropout(F.elu(self.head_tag_feedforward(graph_state)))
        dep_tag = self._dropout(F.elu(self.dep_tag_feedforward(graph_state)))

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

    def graph_pass(self, seq: torch.Tensor, batch_size: int):
        h_prev = torch.zeros((self.graph_l * (2 if self.bidirectional else 1),
                            batch_size,
                            self.hidden_graph)).to(self.config['device'])
        y, lhs = self.graph_rnn(seq, h_prev)
        return y, lhs

    def edge_pass_train(self, graph_state: torch.Tensor, head_indices: torch.LongTensor):
        B, S, D = graph_state.shape
        graph_state_tall = graph_state.reshape(B*S, D)
        # A needs to be shape [B, S, M] (like in `edge_pass_test`)
        A_gold = adj_indices_to_adj_matrix(head_indices)
        A = self.make_adj_sequence(A_gold)
        A = A.to(self.config['device'])
        h_prev = torch.zeros(self.edge_l, B*S, self.hidden_edge, device=A.device)
        h_prev[0, :, :] = graph_state_tall
        out, h_prev = self.edge_rnn(A, h_prev)
        A_out = F.sigmoid(self.edge_cls(out))
        return A_out
    
    def edge_pass_test(self, graph_state: torch.Tensor):
        B, S, D = graph_state.shape # [B, S, D]
        graph_state_tall = graph_state.reshape(B*S, D) # [BS, D]
        x = torch.ones(B*S, 1, 1, device = self.config['device'])
        h_prev = torch.zeros(self.edge_l, B*S, self.hidden_edge, device=x.device)
        h_prev[0, :, :] = graph_state_tall 
        A_out = [x]
        for _ in range(self.M): # produce M edges
            out, h_prev = self.edge_rnn(x, h_prev)
            logits = self.edge_cls(out)
            probs = F.sigmoid(logits)
            x = probs
            A_out.append(probs)
        A_out = torch.cat(A_out, dim = 1)
        return A_out

    def make_adj_sequence(self, adj_square: torch.LongTensor):
        """
            Given a batch of square matrices [B, S, S]
            produce a sequence with B*S elements
            where each element is a sequence of M elements
            each with a scalar,
            where the final shape is then [B*S, M, 1]
        """
        B, S, S = adj_square.shape
        adj_seq = torch.zeros(B, S, self.M + 1, 1)
        adj_seq[:, :, 0, 0] = 1
        """
            I need to put the values of `adj_square` into `adj_seq`
            which means that for each batch (dim0)
            and for each element (nodes, dim1) of `adj_square` then i have a sequence
            of S elements, e.g. 157 head indices for each node.

            I need to put M head indices
            in the last two dimensions [self.M, 1] of `adj_seq`.

            At the i-th step/node, i have to gather head indices
            starting from the earliest possible position of the row,
            i.e. max(0, i-M) and then gather up to that plus min(i, M),
            (excluding the last position, so slice [:min(i, M)])

            Each single head index needs to be put in the last dimension of `adj_seq`
            
            Assuming M=20:
            At i=0 I don't gather any nodes []
            At i=15 I gather [0:15], so [0, ..., 14] --> in this case min(i, M) == 15
            At i=20 I gather [0, ..., 19]
            At i=27 I gather [7, ..., 26]
            and for each of the gathered values i need to go and fill the m-th value
            of `adj_seq`
        """
        for i in range(S):
            start = max(0, i-self.M)
            row = adj_square[:, i, start:i]
            length = row.shape[-1]
            adj_seq[:, i, 1:1+length, 0] = row
        adj_seq = adj_seq.reshape(B*S, self.M + 1, 1)
        return adj_seq

    def reshape_adj(self, A):
        # slice out the BOS scalars
        A = A[:, :, 1:, :]
        # A_seq already has the same number of elements
        # equal to the side of the full adj matrix
        B, S, Mp1, _ = A.shape
        A_reshaped = torch.zeros((B, S, S)).to(self.config['device'])
        for i in range(S):
            start = max(0, i-self.M)
            length  = i - start
            # here we don't slice [1:1+length]
            # like in `make_adj_sequence`
            # because we removed the BOS scalars 
            row = A[:, i, :length, 0]
            A_reshaped[:, i, start:i] = row
        A_reshaped = A_reshaped.transpose(1, 2)
        return A_reshaped
    
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