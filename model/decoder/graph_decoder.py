import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Set
import numpy as np

class GraphDecoder(nn.Module):
    def __init__(self,
                 config: Dict,
                 tag_representation_dim: int,
                 ):
        super().__init__()
        self.config = config
        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim,
                                                      tag_representation_dim,
                                                      config['edge_dim'] + 1,
                                                      )
        self.softmax_multiplier = config["softmax_scaling_coeff"]

    def forward(self,
                arc_logits: torch.Tensor,
                mask: torch.Tensor,
                head_tag: torch.Tensor,
                dep_tag: torch.Tensor,
                adj_m: torch.Tensor,
                adj_m_labels: torch.Tensor,
                ):
        
        # mask scores before decoding
        float_mask = mask.float()
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf

        arc_logits = arc_logits + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # training
        if adj_m is not None and adj_m is not None:
            adj_m=adj_m
            adj_m_labels=adj_m_labels
            arc_loss, tag_loss = self._construct_loss(
                head_tag=head_tag,
                dep_tag=dep_tag,
                arc_logits=arc_logits,
                adj_m=adj_m,
                adj_m_labels=adj_m_labels,
                mask=mask,
            )
            loss = arc_loss #+ tag_loss
        
        # evaluation
        else:
            probs=torch.sigmoid(arc_logits)
            adj_m = (probs >= 0.5).to(torch.int)
            adj_m_labels = None
            loss = None
            arc_loss = None
            tag_loss = None

        output_dict = {
            "adj_m": adj_m,
            "adj_m_labels": adj_m_labels,
            "arc_loss": arc_loss,
            "tag_loss": tag_loss,
            "loss": loss,
            "mask": mask,
        }

        return output_dict

    def _construct_loss(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        arc_logits: torch.Tensor,
        adj_m: torch.Tensor,
        adj_m_labels: torch.Tensor,
        mask: torch.Tensor,
    ):
        # arc_logits: (B, N, N), adj_m: (B, N, N), mask: (B, N)

        # Pairwise mask for valid node pairs
        pair_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).float()  # (B,N,N)

        eye = torch.eye(arc_logits.size(1), device=arc_logits.device).unsqueeze(0)
        pair_mask = pair_mask * (1 - eye)

        # Compute per-edge BCE with logits and then mask
        per_edge_loss = F.binary_cross_entropy_with_logits(
            arc_logits, adj_m.float(), reduction='none'
        )  # (B,N,N)

        per_edge_loss = per_edge_loss * pair_mask

        # Safe average over valid entries
        denom = pair_mask.sum().clamp_min(1.0)
        arc_loss = per_edge_loss.sum() / denom
        # i need to select the representations
        # either based on the gold adjacency matrix
        # or the (binary) predicted edges
        selected_head_tag = self.select_heads(head_tag, dep_tag, edges=adj_m)
        edge_label_logits = self.tag_bilinear(selected_head_tag, dep_tag)

        probs=torch.sigmoid(arc_logits)
        adj_m_preds = (probs >= 0.5).to(torch.int)

        # TODO: implement tag loss if you supervise labels
        tag_loss = torch.tensor(0.0, device=arc_logits.device)

        return arc_loss, tag_loss


    def select_heads(self, head_tag, dep_tag, edges):
        head_tag_logits = ...
        return head_tag_logits