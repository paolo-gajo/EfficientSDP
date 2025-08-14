import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.parser import GraphBiaffineAttention
from torch_geometric.nn import GATv2Conv
from model.gnn.custom_gat import GATv2ConvNormalized
from torch_geometric.utils import unbatch, to_dense_adj
from model.decoder import BilinearDecoder, masked_log_softmax, GraphDecoder
from model.utils.nn import pad_inputs, square_pad_3d, square_pad_4d
import numpy as np
from typing import Set, Tuple, List
from debug.model_debugging import nan_checker, check_param_norm, check_grad_norm
from debug.viz import save_batch_heatmap, indices_to_adjacency_matrices
import copy

GAT_DICT = {
    'base': GATv2Conv,
    'norm': GATv2ConvNormalized,
}

class LGI(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['num_embeddings']:
            self.embedder = nn.Embedding(config['num_embeddings'], self.config['encoder_output_dim'] // self.config['num_node_feats'])
        layers = []
        for i in range(self.config['lgi_enc_layers']):
            if config['num_embeddings']:
                in_dim = self.embedder.embedding_dim * self.config['num_node_feats']
            else:
                in_dim = self.config['feat_dim'] if i == 0 else self.config['encoder_output_dim']
            kwargs = {
                "in_channels": in_dim,
                "out_channels": self.config['encoder_output_dim'],
                "heads": self.config['num_attn_heads'],
                "concat": False,
                "edge_dim": config['edge_dim'] if not config['use_fc'] else None,
                "residual": True,
            }
            if self.config['lgi_gat_type'] == 'norm':
                kwargs.update({"score_norm": self.config['gat_norm']})
            layers.append(GAT_DICT[self.config['lgi_gat_type']](**kwargs))
        self.encoder = nn.ModuleList(layers)

        self.parser = GraphBiaffineAttention(config)
        self.decoder = GraphDecoder(config=config,
                                    tag_representation_dim=config['tag_representation_dim'],
                                    )
        self.mode = "train"
        self.parser.current_step, self._current_step = 0, 0        

    def forward(self, model_input):
        graphs = model_input.to_data_list()

        if self.config['use_fc']:
            batch = model_input.batch.to(self.config['device'])
            # `edge_index` is fully connected with no self loops
            # confirmed with matplotlib (see fc_edge_index.pdf)
            edge_index = build_fc_edge_index(batch, device=self.config['device'])
            edge_attr  = None
            # save_batch_heatmap((to_dense_adj(edge_index, batch=batch) > 0).int(), 'fc_edge_index.pdf')
        else:
            # here `edge_index` is exactly the same as `adj_m_gold` below
            edge_index, edge_attr, batch = _build_proposal_from_graphs(graphs, self.config['device'])
            # save_batch_heatmap((to_dense_adj(edge_index, batch=batch) > 0).int(), 'not_fc_edge_index.pdf')

        # Geometric edge features (distance); requires edge_dim == 1 in GATv2Conv
        # pos = model_input.pos.to(self.config['device'])
        # row, col = edge_index
        # edge_attr = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)  # (E, 1), float
        x = model_input.x.to(self.config['device'])
        if self.config['num_embeddings']:
            x = self.embedder(x)
            x = x.reshape(-1, self.embedder.embedding_dim * self.config['num_node_feats'], 1).squeeze(-1)
        for i, layer in enumerate(self.encoder):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if i < len(self.encoder) - 1:
                x = F.relu(x)

        x = list(unbatch(x, model_input.batch))
        x, mask = pad_inputs(x)

        # get square adjacency matrices from edge indices
        adj_m_labels_gold = [to_dense_adj(el.edge_index,
                                          edge_attr=el.edge_attr if hasattr(el, 'edge_attr') else None,
                                          max_num_nodes=el.num_nodes) for el in graphs]
        adj_m_labels_gold = square_pad_4d(adj_m_labels_gold) # pad them to uniform dimensions

        # get binary adjacency by binaryzing edge features (which are one-hot classes)
        adj_m_gold = (adj_m_labels_gold > 0).any(dim=-1).to(torch.int).to(x.device)
        # save_batch_heatmap(adj_m_gold, 'adj_m_gold.pdf')
        parser_output = self.parser(input=x, mask=mask)

        if self.mode in ["train", "validation"]:
            adj_m_decoder = adj_m_gold
            adj_m_labels_decoder = adj_m_labels_gold
        elif self.mode == "test":
            adj_m_decoder = None
            adj_m_labels_decoder = None

        if not self.config['use_fc']:
            prop_dense = (to_dense_adj(edge_index, batch=batch) > 0)

            # gold dense (bool) per-graph, then pad
            gold_list = []
            for g in graphs:
                gd = to_dense_adj(g.edge_index, edge_attr=getattr(g, 'edge_attr', None),
                                max_num_nodes=g.num_nodes) > 0
                gd = gd.any(dim=-1).squeeze(0)
                gold_list.append(gd)

            B, Nmax, _ = prop_dense.shape
            gold_dense = torch.zeros((B, Nmax, Nmax), dtype=torch.bool, device=prop_dense.device)
            for i, gd in enumerate(gold_list):
                n = gd.size(0); gold_dense[i, :n, :n] = gd

            assert torch.equal(prop_dense, gold_dense), "Encoder proposal != gold (edge set mismatch)"

        decoder_output = self.decoder(
            arc_logits = parser_output['arc_logits'],
            mask = parser_output['mask'],
            head_tag = parser_output['head_tag'],
            dep_tag = parser_output['dep_tag'],
            adj_m = adj_m_decoder,
            adj_m_labels = adj_m_labels_decoder,
        )

        gnn_losses = parser_output.get('gnn_losses', [])

        if self.mode in ["train", "validation"]:
            loss = decoder_output["loss"] * self.config["parser_lambda"]
            if len(gnn_losses) > 0:
                loss += sum(gnn_losses)/len(gnn_losses) * self.config["parser_lambda"]
            return loss
        elif self.mode == "test":
            out_dict = {
                'adj_m_gold': adj_m_gold,
                'adj_m_labels_gold': adj_m_labels_gold,
                'adj_m_pred': decoder_output['adj_m'],
                'adj_m_labels_pred': decoder_output['adj_m_labels'],
                'mask': mask,
                }
            return out_dict

    def set_mode(self, mode="train"):
        """
        This function will determine if loss should be computed or evaluation metrics
        """
        assert mode in [
            "train",
            "test",
            "validation",
        ], f"Mode {mode} is not valid. Mode should be among ['train', 'test', 'validation'] "
        self.mode = mode

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, val):
        self._current_step = val
        if hasattr(self, 'parser'):
            self.parser.current_step = val

    def freeze_gnn(self):
        assert self.config['model_type'] == 'attn'
        if self.config['gnn_layers'] > 0:
            for param in self.parser.conv1_arc.parameters():
                param.requires_grad = False
            for param in self.parser.conv2_arc.parameters():
                param.requires_grad = False
            for param in self.parser.conv1_rel.parameters():
                param.requires_grad = False
            for param in self.parser.conv2_rel.parameters():
                param.requires_grad = False
            for layer in self.parser.arc_bilinear[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
            print(f"Parser GNN conv layers FROZEN at step {self.current_step}!")
            print(f"Parser tail bilinear layers FROZEN at step {self.current_step}!")

    def unfreeze_gnn(self):
        assert self.config['model_type'] == 'attn'
        if self.config['gnn_layers'] > 0:
            for param in self.parser.conv1_arc.parameters():
                param.requires_grad = True
            for param in self.parser.conv2_arc.parameters():
                param.requires_grad = True
            for param in self.parser.conv1_rel.parameters():
                param.requires_grad = True
            for param in self.parser.conv2_rel.parameters():
                param.requires_grad = True
            for layer in self.parser.arc_bilinear[:-1]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Parser GNN conv layers UNFROZEN at step {self.current_step}!")
            print(f"Parser tail bilinear layers UNFROZEN at step {self.current_step}!")
        return True
    
    def init_gnn_biaffines(self, optimizer):
        original_arc_bilinear: nn.ModuleList = self.parser.arc_bilinear
        params_to_remove = set(original_arc_bilinear.parameters())
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p not in params_to_remove]
        trained_arc_bilinear = self.parser.arc_bilinear[-1]
        new_layers = [copy.deepcopy(trained_arc_bilinear) for _ in range(self.config['gnn_layers'])]
        new_arc_bilinear: nn.ModuleList = nn.ModuleList(new_layers + [trained_arc_bilinear])
        self.parser.arc_bilinear = new_arc_bilinear
        optimizer.add_param_group({"params": self.parser.arc_bilinear.parameters()})

    def freeze_tagger(self):
        """Freeze tagger if asked for!"""
        for param in self.tagger.parameters():
            param.requires_grad = False

    def freeze_parser(self):
        """Freeze parser if asked for!"""
        for param in self.parser.parameters():
            param.requires_grad = False

def build_fc_edge_index(batch, device=None):
    device = device or batch.device
    sizes = torch.bincount(batch).tolist()                   # nodes per graph
    offsets = torch.tensor([0] + sizes[:-1], device=device).cumsum(0)

    chunks = []
    for n, off in zip(sizes, offsets):
        idx  = torch.arange(off, off + n, device=device)
        row  = idx.repeat_interleave(n)
        col  = idx.repeat(n)
        mask = row != col                                    # drop diagonal
        chunks.append(torch.stack((row[mask], col[mask]), 0))
    if chunks:
        ei = torch.cat(chunks, dim=1)
    else:
        ei = torch.empty(2, 0, dtype=torch.long, device=device)
    return ei.long()

def _canon_filter(ei, ea):
    # ei: (2, E) local indices 0..n-1, ea: (E, d) or None
    if ea is None:
        return ei, None
    if ea.dim() == 1:
        ea = ea.unsqueeze(-1)
    keep = (ea > 0).any(dim=-1).bool()
    return ei[:, keep], ea[keep]

def _build_proposal_from_graphs(graphs, device):
    edge_indices, edge_attrs = [], []
    sizes = []
    off = 0
    for el in graphs:
        ei = el.edge_index.to(device)
        ea = getattr(el, 'edge_attr', None)
        if ea is not None: ea = ea.to(device)
        ei, ea = _canon_filter(ei, ea)           # <-- SAME rule as gold
        n = el.num_nodes
        sizes.append(n)
        edge_indices.append(ei + off)
        if ea is not None: edge_attrs.append(ea)
        off += n

    if len(edge_indices) == 0:
        E = torch.empty(2, 0, dtype=torch.long, device=device)
        A = None
    else:
        E = torch.cat(edge_indices, dim=1)
        A = torch.cat(edge_attrs, dim=0) if len(edge_attrs) == len(edge_indices) else None

    # batch vector consistent with graphs
    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(sizes)],
        dim=0
    )
    return E, A, batch
