from typing import Dict, Optional, Tuple, Any, List, Set
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
from model.parser.parser_nn import *
from model.decoder import masked_log_softmax
import math
from debug import save_heatmap
import warnings
import copy

class TriParser(nn.Module):
    def __init__(
        self,
        config: Dict,
        encoder: nn.LSTM,
        embedding_dim: int,
        n_edge_labels: int,
        tag_embedder: nn.Linear,
        arc_representation_dim: int,
        tag_representation_dim: int,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        n_entity_labels: int = None,
        n_relation_labels: int = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config['use_parser_rnn'] \
        and config['parser_rnn_layers'] > 0 \
        and config['parser_rnn_hidden_size'] > 0:
            self.encoder_h = encoder
            encoder_dim = self.config["parser_rnn_hidden_size"] * 2
        else:
            self.encoder_h = None
            encoder_dim = embedding_dim

        if self.config["tag_embedding_type"] != 'none':
            self.tag_embedder = tag_embedder
            self.tag_dropout = nn.Dropout(0.2)
        
        self.head_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.dept_arc_feedforward = nn.Linear(encoder_dim, arc_representation_dim)
        self.head_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)
        self.dept_tag_feedforward = nn.Linear(encoder_dim, tag_representation_dim)

        self.arc_bilinear = BilinearMatrixAttention(arc_representation_dim,
                                    arc_representation_dim,
                                    activation = nn.ReLU() if self.config['activation'] == 'relu' else None,
                                    use_input_biases=True,
                                    bias_type='simple',
                                    arc_norm=self.config['arc_norm'],
                                    )

        # Higher-Order Modeling components as described in Section D
        # Trilinear functions for sibling, co-parent, and grandparent types
        d = tag_representation_dim  # dimension for trilinear operations
        
        # W3, W4, W5 from equations 12, 13, 14
        self.W_sib = nn.Parameter(torch.Tensor(d, d, d))
        self.W_cop = nn.Parameter(torch.Tensor(d, d, d))
        self.W_gp = nn.Parameter(torch.Tensor(d, d, d))
        
        # Entity and relation label prediction components
        self.n_entity_labels = n_entity_labels
        self.n_relation_labels = n_relation_labels
        
        if n_entity_labels is not None:
            self.entity_classifier = nn.Linear(encoder_dim, n_entity_labels)
        
        if n_relation_labels is not None:
            self.relation_classifier = nn.Linear(2 * encoder_dim, n_relation_labels)
        
        # MFV inference parameters
        self.edge_feature_dim = config.get('edge_feature_dim', 128)
        self.edge_feature_projection = nn.Linear(encoder_dim * 2, self.edge_feature_dim)
        
        # Type-specific edge scores (sib, cop, gp)
        self.edge_type_projections = nn.ModuleDict({
            'sib': nn.Linear(self.edge_feature_dim, 1),
            'cop': nn.Linear(self.edge_feature_dim, 1),
            'gp': nn.Linear(self.edge_feature_dim, 1)
        })

        self._dropout = nn.Dropout(dropout)
        self._head_sentinel = torch.nn.Parameter(torch.randn(encoder_dim))
        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation
        self.apply(self._init_weights)
        self.tag_representation_dim = tag_representation_dim
        self.n_edge_labels = n_edge_labels
        
        # Number of MFV iterations
        self.mfv_iterations = config.get('mfv_iterations', 3)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module.data)

    def trilinear(self, x, y, z, W):
        """
        Compute trilinear function as in equation 11: Trilinear(z1, z2, z3) = z1^T W z2
        """
        # x: [batch, seq_len, dim]
        # y: [batch, seq_len, dim]
        # z: [batch, seq_len, dim]
        # W: [dim, dim, dim]
        
        batch_size, seq_len, dim = x.size()
        
        # Reshape for computation
        x_reshaped = x.view(batch_size, seq_len, 1, dim)
        y_reshaped = y.view(batch_size, 1, seq_len, dim)
        z_reshaped = z.view(batch_size, seq_len, 1, dim)
        
        # Compute x^T W y
        x_W = torch.einsum('bijd,dkl->bijkl', x_reshaped, W)
        x_W_y = torch.einsum('bijkl,bjkd->bikd', x_W, y_reshaped)
        
        # Compute (x^T W y) z
        result = torch.einsum('bikd,bkd->bik', x_W_y, z_reshaped)
        
        return result

    def compute_type_scores(self, head_repr, dep_repr, mask):
        """
        Compute scores for sibling, co-parent, and grandparent types
        as in equations 12, 13, and 14
        """
        batch_size, seq_len, dim = head_repr.size()
        
        # Compute phi scores for each type
        # Equation 12: Sibling score
        phi_sib = self.trilinear(head_repr, dep_repr, dep_repr, self.W_sib)
        
        # Equation 13: Co-parent score
        phi_cop = self.trilinear(head_repr, dep_repr, head_repr, self.W_cop)
        
        # Equation 14: Grandparent score
        # Need to create head_tail representation
        head_tail = torch.cat([head_repr[:, :-1], head_repr[:, 1:]], dim=-1)
        phi_gp = self.trilinear(head_repr, head_tail, dep_repr, self.W_gp)
        
        # Apply mask
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [batch, seq_len, seq_len]
        phi_sib = phi_sib * mask_2d
        phi_cop = phi_cop * mask_2d
        phi_gp = phi_gp * mask_2d
        
        return phi_sib, phi_cop, phi_gp

    def compute_edge_features(self, encoded_text, mask):
        """
        Compute edge features for MFV inference
        """
        batch_size, seq_len, dim = encoded_text.size()
        
        # Create pairwise features
        head_repr = encoded_text.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq_len, seq_len, dim]
        dep_repr = encoded_text.unsqueeze(1).expand(-1, seq_len, -1, -1)   # [batch, seq_len, seq_len, dim]
        
        # Concatenate head and dependent representations
        edge_repr = torch.cat([head_repr, dep_repr], dim=-1)  # [batch, seq_len, seq_len, 2*dim]
        
        # Project to edge feature space
        edge_features = self.edge_feature_projection(edge_repr)  # [batch, seq_len, seq_len, edge_feat_dim]
        
        return edge_features

    def mean_field_variational_inference(self, unary_potentials, edge_features, mask):
        """
        Implement MFV inference as described in Section D
        """
        batch_size, seq_len, _ = unary_potentials.size()
        
        # Initialize distribution G^0_ij(Q_ij) by normalizing unary potential
        edge_dist = torch.sigmoid(unary_potentials)  # Initial G^0_ij(Q_ij) from equation 15
        
        # Perform M iterations of MFV updates (equation 17 and 18)
        for m in range(self.mfv_iterations):
            # Create a copy of current distribution for message passing
            prev_dist = edge_dist.clone()
            
            # Calculate type-specific messages (equation 17 expanded in equation 18)
            # Compute messages for sibling type
            sib_messages = torch.bmm(prev_dist, self.edge_type_projections['sib'](edge_features))
            
            # Compute messages for co-parent type
            cop_messages = torch.bmm(prev_dist.transpose(1, 2), self.edge_type_projections['cop'](edge_features))
            
            # Compute messages for grandparent type
            # This needs special handling for the hierarchical structure
            gp_messages = torch.zeros_like(sib_messages)
            for i in range(1, seq_len):
                gp_messages[:, i, :] = torch.bmm(prev_dist[:, :i, :], self.edge_type_projections['gp'](edge_features[:, :i, :]))
            
            # Sum up all messages (equation 18)
            total_messages = sib_messages + cop_messages + gp_messages
            
            # Update edge distribution (equation 18)
            edge_dist = torch.sigmoid(unary_potentials + total_messages)
            
            # Apply mask
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            edge_dist = edge_dist * mask_2d
        
        return edge_dist

    def forward(
        self,
        encoded_text_input: torch.FloatTensor,
        pos_tags: torch.LongTensor,
        mask: torch.LongTensor,
        metadata: List[Dict[str, Any]] = [],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        entity_labels: torch.LongTensor = None,
        relation_labels: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        
        pos_tags = pos_tags['pos_tags_one_hot'] if self.config['tag_embedding_type'] == 'linear' else pos_tags['pos_tags_labels']

        if self.config["tag_embedding_type"] != 'none':
            tag_embeddings = self.tag_dropout(F.relu(self.tag_embedder(pos_tags)))
            encoded_text_input = torch.cat([encoded_text_input, tag_embeddings], dim=-1)

        if self.encoder_h is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_input = pack_padded_sequence(
                encoded_text_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.encoder_h(packed_input)
            encoded_text_input, _ = pad_packed_sequence(packed_output,
                                                        batch_first=True,
                                                        total_length=encoded_text_input.size(1))

        batch_size, _, encoding_dim = encoded_text_input.size()
        head_sentinel = self._head_sentinel.view(1, 1, -1).expand(batch_size, 1, encoding_dim)
        
        encoded_text_input = torch.cat([head_sentinel, encoded_text_input], dim=1)

        mask_ones = mask.new_ones(batch_size, 1)
        mask = torch.cat([mask_ones, mask], dim = 1)
        
        if head_indices is not None:
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], dim=1
            )
        if head_tags is not None:
            head_tags = torch.cat(
                [head_tags.new_zeros(batch_size, 1), head_tags], dim=1
            )
        
        encoded_text_input = self._dropout(encoded_text_input)
            
        head_arc = self._dropout(F.elu(self.head_arc_feedforward(encoded_text_input)))
        dept_arc = self._dropout(F.elu(self.dept_arc_feedforward(encoded_text_input)))
        head_tag = self._dropout(F.elu(self.head_tag_feedforward(encoded_text_input)))
        dept_tag = self._dropout(F.elu(self.dept_tag_feedforward(encoded_text_input)))

        # Basic dependency parsing
        attended_arcs = self.arc_bilinear(head_arc, dept_arc)
        
        # Higher-order modeling
        # Compute type-specific scores
        phi_sib, phi_cop, phi_gp = self.compute_type_scores(head_tag, dept_tag, mask)
        
        # Compute edge features for MFV inference
        edge_features = self.compute_edge_features(encoded_text_input, mask)
        
        # Perform MFV inference to get edge distributions
        edge_dist = self.mean_field_variational_inference(attended_arcs, edge_features, mask)
        
        # Entity and relation label predictions (equations 20-22)
        entity_predictions = None
        relation_predictions = None
        
        if self.n_entity_labels is not None:
            # Predict entity labels (equation 20)
            entity_predictions = self.entity_classifier(encoded_text_input)
        
        if self.n_relation_labels is not None:
            # Predict relation labels between tokens (equation 22)
            # We need to create pairwise token representations
            batch_size, seq_len, dim = encoded_text_input.size()
            head_repr = encoded_text_input.unsqueeze(2).expand(-1, -1, seq_len, -1)
            dep_repr = encoded_text_input.unsqueeze(1).expand(-1, seq_len, -1, -1)
            
            # Concatenate and classify
            pair_repr = torch.cat([head_repr, dep_repr], dim=-1)
            relation_predictions = self.relation_classifier(pair_repr)
            
            # Apply edge existence mask from MFV inference
            # Only predict relations for edges that exist (where edge_dist > 0.5)
            edge_mask = (edge_dist > 0.5).float()
            relation_predictions = relation_predictions * edge_mask.unsqueeze(-1)

        losses = []
        if head_tags is not None and head_indices is not None:
            # Calculate cross-entropy loss for dependency parsing
            arc_loss = self._get_arc_loss(attended_arcs, head_indices, mask)
            tag_loss = self._get_tag_loss(
                relation_predictions, head_tags, head_indices, mask
            ) if relation_predictions is not None else 0
            losses.append(arc_loss + tag_loss)
        
        if entity_labels is not None and self.n_entity_labels is not None:
            # Calculate cross-entropy loss for entity labels (equation 24)
            entity_loss = F.cross_entropy(
                entity_predictions.view(-1, self.n_entity_labels),
                entity_labels.view(-1),
                ignore_index=-1
            )
            losses.append(entity_loss)
        
        if relation_labels is not None and self.n_relation_labels is not None:
            # Calculate cross-entropy loss for relation labels (equation 23)
            valid_positions = (relation_labels != -1).float()
            relation_loss = F.cross_entropy(
                relation_predictions.view(-1, self.n_relation_labels),
                relation_labels.view(-1),
                ignore_index=-1
            )
            losses.append(relation_loss)

        output = {
            'head_tag': head_tag,
            'dept_tag': dept_tag,
            'head_indices': head_indices,
            'head_tags': head_tags,
            'attended_arcs': attended_arcs,
            'edge_dist': edge_dist,  # G^M_ij(Q_ij) from MFV inference
            'phi_sib': phi_sib,      # Sibling scores
            'phi_cop': phi_cop,      # Co-parent scores
            'phi_gp': phi_gp,        # Grandparent scores
            'mask': mask,
            'metadata': metadata,
            'gnn_losses': losses,
        }
        
        if entity_predictions is not None:
            output['entity_predictions'] = entity_predictions
        
        if relation_predictions is not None:
            output['relation_predictions'] = relation_predictions

        return output
        
    def _get_arc_loss(self, arc_scores, arc_targets, mask):
        """
        Calculate loss for arc prediction
        """
        # Flatten the scores and targets
        batch_size, seq_len, _ = arc_scores.size()
        arc_scores = arc_scores.view(-1)
        arc_targets = arc_targets.view(-1)
        mask = mask.view(-1).float()
        
        # Calculate binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(arc_scores, arc_targets.float(), reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def _get_tag_loss(self, tag_scores, tag_targets, arc_targets, mask):
        """
        Calculate loss for tag prediction
        """
        # Get scores for correct arcs only
        batch_size, seq_len, _, n_labels = tag_scores.size()
        flat_indices = self._get_flat_indices(arc_targets, batch_size, seq_len)
        tag_scores = tag_scores.view(-1, n_labels).index_select(0, flat_indices)
        
        # Get targets and mask
        tag_targets = tag_targets.view(-1).index_select(0, flat_indices)
        mask = mask.view(-1).index_select(0, flat_indices).float()
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(tag_scores, tag_targets, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def _get_flat_indices(self, indices, batch_size, seq_len):
        """
        Convert batched indices to flat indices for selection
        """
        offset = torch.arange(batch_size) * seq_len
        offset = offset.unsqueeze(-1).expand_as(indices)
        flat_indices = (indices + offset).view(-1)
        
        return flat_indices

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert model output to human readable format:
        - Use Chu-Liu-Edmonds algorithm to find the optimal parse tree
        - Convert indices to actual arcs
        - Get relation and entity labels
        """
        arc_edge_indices = output_dict["edge_dist"]
        
        # Find edges with probability > 0.5 (as mentioned in the paper)
        predicted_edges = (arc_edge_indices > 0.5).float()
        
        # Get entity labels if available
        if "entity_predictions" in output_dict:
            entity_probs = F.softmax(output_dict["entity_predictions"], dim=-1)
            entity_preds = entity_probs.argmax(dim=-1)
            output_dict["predicted_entities"] = entity_preds
        
        # Get relation labels if available
        if "relation_predictions" in output_dict:
            # Only predict relations for pred