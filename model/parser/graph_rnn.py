import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from debug.viz import save_heatmap, save_batch_heatmap
from model.utils.nn import adjust_for_sentinel, BilinearMatrixAttention, adj_indices_to_adj_matrix
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
        self.bidirectional = True
        self.graph_rnn = nn.RNN(self.input_size,
                                self.hidden_graph,
                                num_layers=self.graph_l,
                                batch_first=True,
                                bidirectional=self.bidirectional,
                                dropout=0.3,
                )
        self.edge_rnn_past = nn.RNN(1,
                                self.hidden_edge,
                                num_layers=self.edge_l,
                                batch_first=True,
                                bidirectional=False,
                                dropout=0.3,
                )
        self.edge_rnn_future = nn.RNN(1,
                                self.hidden_edge,
                                num_layers=self.edge_l,
                                batch_first=True,
                                bidirectional=False,
                                dropout=0.3,
                )
        
        self.edge_cls_past = nn.Linear(self.hidden_edge, 1)
        self.edge_cls_future = nn.Linear(self.hidden_edge, 1)
        graph_rnn_output_dim = self.hidden_graph * 2 if self.bidirectional else self.hidden_graph
        self.graph_to_edge = nn.Linear(graph_rnn_output_dim, self.hidden_edge)

        self.bos_past = nn.Parameter(torch.tensor([0.0]))
        self.bos_future = nn.Parameter(torch.tensor([0.0]))
        
        self._head_sentinel = torch.nn.Parameter(torch.randn(self.input_size))

        self.head_tag_feedforward = nn.Linear(graph_rnn_output_dim, self.tag_representation_dim)
        self.dep_tag_feedforward = nn.Linear(graph_rnn_output_dim, self.tag_representation_dim)
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
            A_past, A_future = self.edge_pass_train(graph_state=graph_state, head_indices=head_indices)
        else:
            A_past = self.edge_pass_test(graph_state=graph_state, future=False)
            A_future = self.edge_pass_test(graph_state=graph_state, future=True)
            ...
        neg_inf = torch.finfo(graph_state.dtype).min
        Ap = A_past.reshape(B, S, -1)
        Af = A_future.reshape(B, S, -1)
        arc_logits = torch.full((B, S, S), 0, device=graph_state.device, dtype=graph_state.dtype)

        for i in range(S):
            start = max(0, i - self.M)
            length = i - start
            if length > 0:
                arc_logits[:, i, start:i] = Ap[:, i, :length]
        # for i in range(S):
        #     end = min(S, i + self.M)
        #     length = end - i
        #     if length > 0:
        #         arc_logits[:, i, i:end] = Af[:, i, :length]

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
        num_dirs = 2 if self.bidirectional else 1
        if isinstance(self.graph_rnn, nn.LSTM):
            h0 = torch.zeros(self.graph_l * num_dirs, batch_size, self.hidden_graph, device=self.config['device'])
            c0 = torch.zeros_like(h0)
            y, (hn, cn) = self.graph_rnn(seq, (h0, c0))
            return y, (hn, cn)
        else:  # nn.RNN or nn.GRU
            h0 = torch.zeros(self.graph_l * num_dirs, batch_size, self.hidden_graph, device=self.config['device'])
            y, hn = self.graph_rnn(seq, h0)
            return y, (hn, None)

    def edge_pass_train(self, graph_state, head_indices):
        B, S, D = graph_state.shape
        proj = self.graph_to_edge(graph_state.reshape(B*S, D))

        def run_seq(A_in, future: bool):
            rnn  = self.edge_rnn_future if future else self.edge_rnn_past
            head = self.edge_cls_future if future else self.edge_cls_past
            L, H = self.edge_l, self.hidden_edge

            # init hidden (and cell if LSTM)
            if isinstance(rnn, nn.LSTM):
                h0 = torch.zeros(L, B*S, H, device=A_in.device, dtype=proj.dtype)
                c0 = torch.zeros_like(h0)
                h0[0, :, :] = proj
                out, _ = rnn(A_in, (h0, c0))            # [B*S, L+1, H]
            else:  # nn.RNN / nn.GRU
                h0 = torch.zeros(L, B*S, H, device=A_in.device, dtype=proj.dtype)
                h0[0, :, :] = proj
                out, _ = rnn(A_in, h0)                  # [B*S, L+1, H]

            logits = head(out)[:, :, :]               # drop BOS-shifted last
            return logits                                # [B*S, L, 1]

        A_square     = adj_indices_to_adj_matrix(head_indices)
        A_in_past, A_in_future = self.make_adj_sequence(A_square, M=self.M)
        A_out_past   = run_seq(A_in_past,   future=False)
        A_out_future = run_seq(A_in_future, future=True)

        return A_out_past, A_out_future

    def edge_pass_test(self, graph_state: torch.Tensor, future: bool = False):
        B, S, D = graph_state.shape
        proj = self.graph_to_edge(graph_state.reshape(B*S, D))
        L, H = self.edge_l, self.hidden_edge

        rnn  = self.edge_rnn_future if future else self.edge_rnn_past
        head = self.edge_cls_future if future else self.edge_cls_past

        # init hidden (and cell if LSTM)
        if isinstance(rnn, nn.LSTM):
            h = torch.zeros(L, B*S, H, device=self.config['device'], dtype=proj.dtype)
            c = torch.zeros_like(h)
            h[0, :, :] = proj
        else:
            h = torch.zeros(L, B*S, H, device=self.config['device'], dtype=proj.dtype)
            h[0, :, :] = proj
            c = None  # unused

        x = (self.bos_future if future else self.bos_past).view(1, 1).expand(B*S, 1, 1).to(self.config['device'])
        preds = []
        steps = min(self.M, S)
        for _ in range(steps):
            if isinstance(rnn, nn.LSTM):
                out, (h, c) = rnn(x, (h, c))         # (B*S, 1, H)
            else:
                out, h = rnn(x, h)                   # (B*S, 1, H)
            logit = head(out)                        # (B*S, 1, 1)
            preds.append(logit)
            x = logit
        return torch.cat(preds, dim=1)               # (B*S, M, 1)

    def make_adj_sequence(self, adj_square: torch.LongTensor, M: int = 20, future: bool = False):
        '''
        return:
            A matrix where each row is a sequence that the RNN will have to learn.
        '''
        B, S, _ = adj_square.shape
        L = min(M, S)
        device = adj_square.device
        out_past = torch.zeros(B, S, L, 1, device=device, dtype=torch.float32)
        out_future = torch.zeros(B, S, L, 1, device=device, dtype=torch.float32)
        for i in range(S):
            end = min(S, i + 1 + M)
            row_future = adj_square[:, i, i+1:end]
            start = max(0, i - M)
            row_past = adj_square[:, i, start:i]
            length_past = row_past.shape[-1]
            length_future = row_future.shape[-1]
            assert length_past <= L
            assert length_future <= L
            if length_past > 0:
                out_past[:, i, :length_past, 0] = row_past.to(out_past.dtype)          # shape: (B, Lp)
            if length_future > 0:
                out_future[:, i, :length_future, 0] = row_future.to(out_future.dtype)      # shape: (B, Lf)
            ...

        bos = (self.bos_future if future else self.bos_past).view(1, 1, 1, 1).expand(B, S, 1, 1).to(device)
        # out = torch.cat([bos, out], dim=2)
        out_reshaped_past = out_past.reshape(B * S, L, 1)
        out_reshaped_past = out_reshaped_past.to(self.config['device'])
        out_reshaped_future = out_future.reshape(B * S, L, 1)
        out_reshaped_future = out_reshaped_future.to(self.config['device'])
        return out_reshaped_past, out_reshaped_future

    def reshape_adj(self, A: torch.Tensor, B: int, S: int):
        '''
        Turns a [B*S, D] matrix [B, S, D] 
        '''
        L = A.shape[1]
        A_reshaped = A.reshape(B, S, L)
        A_new = torch.zeros((B, S, S)).to(A.device)
        for b, el in enumerate(A_reshaped):
            for i in range(el.shape[0]):
                start = max(0, i-self.M)
                length  = i - start
                A_new[b, i, start:i] = el[i, :length]
        return A_new