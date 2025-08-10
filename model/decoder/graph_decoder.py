import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Set
import numpy as np

class GraphDecoder(nn.Module):
    def __init__(self,
                 config: Dict,
                 tag_representation_dim: int,
                 n_edge_labels: int,
                 use_mst_decoding_for_validation: bool = True,
                 ):
        super().__init__()
        self.config = config
        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation
        self.tag_bilinear = torch.nn.modules.Bilinear(tag_representation_dim, tag_representation_dim, n_edge_labels)
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
            probs=F.sigmoid(arc_logits)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, sequence_length, _ = arc_logits.size()
        mask = (mask.float() + tiny_value_of_dtype(arc_logits.dtype)).log()
        probs = torch.sigmoid(arc_logits)
        arc_loss = F.binary_cross_entropy(probs, adj_m.float())
        tag_loss = ...
        return arc_loss, tag_loss

    def _greedy_decode(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        arc_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when deptren are attached to their parents).

        Parameters
        ----------
        head_tag : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        dep_tag : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        arc_logits : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        diag_mask = torch.diag(arc_logits.new(mask.size(1)).fill_(-np.inf))
        arc_logits = arc_logits + diag_mask
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).byte().unsqueeze(2)
            arc_logits.masked_fill_(minus_mask.bool(), -np.inf)
        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = arc_logits.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag, dep_tag, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

