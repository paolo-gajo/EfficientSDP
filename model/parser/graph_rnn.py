import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from debug.viz import save_heatmap, save_batch_heatmap
from model.utils.nn import adjust_for_sentinel, BilinearMatrixAttention, adj_indices_to_adj_matrix
from model.utils.rnn_utils import make_adj_sequence, reshape_adj
from typing import List, Dict, Any

class GraphRNNBilinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['tag_embedding_type'] == 'linear':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
        elif config['tag_embedding_type'] == 'embedding':
            embedding_dim = config["encoder_output_dim"] + config["tag_representation_dim"] # 768 + 100 = 868
        elif config['tag_embedding_type'] == 'none':
            embedding_dim = config["encoder_output_dim"] # 768
        else:
            raise ValueError('Parameter `tag_embedding_type` can only be == `linear` or `embedding` or `none`!')    
        self.input_size = embedding_dim
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
        
        self.head_arc_feedforward = nn.Linear(self.hidden_graph * 2 if self.bidirectional else self.hidden_graph,
                                        config['arc_representation_dim'])
        self.dept_arc_feedforward = nn.Linear(self.hidden_graph * 2 if self.bidirectional else self.hidden_graph,
                                        config['arc_representation_dim'])

        self.arc_bilinear = BilinearMatrixAttention(config['arc_representation_dim'],
                                    config['arc_representation_dim'],
                                    activation = nn.ReLU() if self.config['biaffine_activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type=self.config['bias_type'],
                                    arc_norm=self.config['arc_norm'],
                                    )

        self.head_tag_feedforward = nn.Linear(self.hidden_graph, self.tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(self.hidden_graph, self.tag_representation_dim)
        self._dropout = nn.Dropout(config['tag_dropout'])

    def forward(self,
                input: torch.Tensor,
                tag_embeddings: torch.Tensor,
                mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = [],
                head_tags: torch.Tensor = None,
                head_indices: torch.Tensor = None,
                step_indices: torch.LongTensor = None,
                graph_laplacian: torch.LongTensor = None,
                mode: str = None,
                ):
        
        if tag_embeddings is not None:
            input = torch.cat([input, tag_embeddings], dim=-1)
        
        # input dim [B, S, D]
        B, _, D = input.shape
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(B, 1, D)
        input = torch.cat([head_sentinel, input], dim=1)
        mask, head_indices, head_tags = adjust_for_sentinel(mask, head_indices, head_tags)
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
        arc_logits = reshape_adj(A=A_batch, M=self.M)

        # head_arc = self._dropout(F.elu(self.head_arc_feedforward(graph_state)))
        # dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(graph_state)))

        # arc_logits = self.arc_bilinear(head_arc, dept_arc)

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
        A_square = adj_indices_to_adj_matrix(head_indices)
        A = make_adj_sequence(A_square, M=self.M)
        A = A.to(self.config['device'])
        h_prev = torch.zeros(self.edge_l, B*S, self.hidden_edge, device=A.device)
        h_prev[0, :, :] = graph_state_tall
        out, h_prev = self.edge_rnn(A, h_prev)
        pred = self.edge_cls(out)
        # pred = F.sigmoid(pred)
        A_out = pred
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
            pred = self.edge_cls(out)
            # pred = F.sigmoid(pred)
            x = pred
            A_out.append(pred)
        A_out = torch.cat(A_out, dim = 1)
        return A_out
    
class GraphRNNSimple(nn.Module):
    ...