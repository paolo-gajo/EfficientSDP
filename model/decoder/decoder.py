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
                head_tag: torch.Tensor,
                dep_tag: torch.Tensor,
                head_indices: torch.Tensor,
                head_tags: torch.Tensor,
                attended_arcs: torch.Tensor,
                mask: torch.Tensor,
                metadata: List[Dict[str, Any]] = [],
                ):
        
        # mask scores before decoding
        float_mask = mask.float()
        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf

        if not self.config["use_step_mask"]:
            attended_arcs = (
                attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
            )
        else:
            attended_arcs = attended_arcs + minus_mask

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag,
                dep_tag,
                attended_arcs,
                mask
                )
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag,
                dep_tag,
                attended_arcs,
                mask,
            )

        predicted_heads = predicted_heads.to(head_tag.device)
        predicted_head_tags = predicted_head_tags.to(head_tag.device)

        if head_indices is not None and head_tags is not None:
            head_indices=head_indices
            head_tags=head_tags
        else:
            head_indices=predicted_heads.long()
            head_tags=predicted_head_tags.long()

        arc_nll, tag_nll = self._construct_loss(
            head_tag=head_tag,
            dep_tag=dep_tag,
            attended_arcs=attended_arcs,
            head_indices=head_indices,
            head_tags=head_tags,
            mask=mask,
        )

        loss = arc_nll + tag_nll

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
        }

        return output_dict

    def _construct_loss(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

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
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : ``torch.Tensor``, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        Returns
        -------
        arc_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc loss.
        tag_nll : ``torch.Tensor``, required.
            The negative log likelihood from the arc tag loss.
        """
        float_mask = mask.float()
        batch_size, sequence_length, _ = attended_arcs.size()

        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        if not self.config["use_step_mask"]:
            normalised_arc_logits = masked_log_softmax(attended_arcs, mask)
            normalised_arc_logits = (normalised_arc_logits * float_mask.unsqueeze(2) * float_mask.unsqueeze(1))
        else:
            normalised_arc_logits = masked_log_softmax(attended_arcs, mask)
            normalised_arc_logits = normalised_arc_logits * float_mask

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(head_tag, dep_tag, head_indices)
        normalised_head_tag_logits = masked_log_softmax(head_tag_logits, mask.unsqueeze(-1))
        normalised_head_tag_logits = normalised_head_tag_logits * float_mask.unsqueeze(-1)

        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
        dept_index = timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()

        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, dept_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, dept_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()

        return arc_nll, tag_nll

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = list(instance_tags[1:length])
            # `instance_tags` are the indices of the tags. If the names themselves are needed,
            # you should write a mapping function to do the conversion before the following line
            head_tag_labels.append(instance_tags)
            head_indices.append(instance_heads)

        output_dict["predicted_dependencies"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _greedy_decode(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        attended_arcs: torch.Tensor,
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
        attended_arcs : ``torch.Tensor``, required.
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
        diag_mask = torch.diag(attended_arcs.new(mask.size(1)).fill_(-np.inf))
        # save_heatmap(attended_arcs[0], filename='attended_arcs_diag.pdf')
        attended_arcs = attended_arcs + diag_mask
        # save_heatmap(attended_arcs[0], filename='attended_arcs_nodiag.pdf')
        # save_heatmap(attended_arcs[0], filename='attended_arcs_2.pdf')
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            if not self.config["use_step_mask"]:
                minus_mask = (1 - mask).byte().unsqueeze(2)
            else:
                minus_mask = (1 - mask).byte()
            attended_arcs.masked_fill_(minus_mask.bool(), -np.inf)
        # save_heatmap(attended_arcs[0], filename='attended_arcs_3.pdf')
        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag, dep_tag, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding maximum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

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
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [
            batch_size,
            sequence_length,
            sequence_length,
            tag_representation_dim,
        ]
        head_tag = head_tag.unsqueeze(2)
        head_tag = head_tag.expand(*expanded_shape).contiguous()
        dep_tag = dep_tag.unsqueeze(1)
        dep_tag = dep_tag.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag, dep_tag)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)

        """
            Here, before feeding scores to the MST algorithm, we perform softmax
            with temperature, to make most likely label to have score of 1 and rest
            are squashed to 0. This way, MST will take most likely path to build a tree 
            and we won't run into an issue of low precision and high recall. We do this
            softmax for both, arc labels and edge labels. 
        """
        pairwise_head_logits = self.softmax_multiplier * (
            pairwise_head_logits
            - torch.max(pairwise_head_logits, dim=3)[0].unsqueeze(dim=3)
        )
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(0, 3, 1, 2)

        # Shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.softmax_multiplier * (
            attended_arcs - torch.max(attended_arcs, dim=2)[0].unsqueeze(dim=2)
        )
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their deptren.

        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
        batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):

            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for dept, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, dept].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necesarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)

        return torch.from_numpy(np.stack(heads)), torch.from_numpy(
            np.stack(head_tags)
        )

    def _get_head_tags(
        self,
        head_tag: torch.Tensor,
        dep_tag: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and dept tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

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
        head_indices : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        Returns
        -------
        head_tag_logits : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag)
        ).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.

        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag[
            range_vector, head_indices
        ]
        
        selected_head_tag_representations = (
            selected_head_tag_representations.contiguous()
        )
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, dep_tag
        )
        return head_tag_logits

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset)


def masked_log_softmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1
) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


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


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.arange(size, dtype=torch.long, device=f"cuda:{device}")
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)

def decode_mst(
    energy: np.ndarray, length: int, has_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.

    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arborescences on graphs.

    # Parameters

    energy : `np.ndarray`, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is `False`,
        the tensor should have shape (timesteps, timesteps) instead.
    length : `int`, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : `bool`, optional, (default = `True`)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise Exception("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise Exception("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = np.array(original_score_matrix, copy=True)

    old_input = np.zeros([length, length], dtype=np.int32)
    old_output = np.zeros([length, length], dtype=np.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    chu_liu_edmonds(
        length,
        score_matrix,
        current_nodes,
        final_edges,
        old_input,
        old_output,
        representatives,
    )

    heads = np.zeros([max_length], np.int32)
    if has_labels:
        head_type = np.ones([max_length], np.int32)
    else:
        head_type = None

    for dept, parent in final_edges.items():
        heads[dept] = parent
        if has_labels:
            head_type[dept] = label_id_matrix[parent, dept]

    return heads, head_type


def chu_liu_edmonds(
    length: int,
    score_matrix: np.ndarray,
    current_nodes: List[bool],
    final_edges: Dict[int, int],
    old_input: np.ndarray,
    old_output: np.ndarray,
    representatives: List[Set[int]],
):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.

    Note that this function operates in place, so variables
    will be modified.

    # Parameters

    length : `int`, required.
        The number of nodes.
    score_matrix : `np.ndarray`, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : `List[bool]`, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges : `Dict[int, int]`, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input : `np.ndarray`, required.
    old_output : `np.ndarray`, required.
    representatives : `List[Set[int]]`, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.

    # Returns

    Nothing - all variables are modified in place.

    """
    # Set the initial graph to be the greedy best one.
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            dept = old_output[parents[node], node]
            final_edges[dept] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle

            # Add the new edge score to the cycle weight
            # and subtract the edge we're considering removing.
            score = (
                cycle_weight
                + score_matrix[node, node_in_cycle]
                - score_matrix[parents[node_in_cycle], node_in_cycle]
            )

            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    chu_liu_edmonds(
        length,
        score_matrix,
        current_nodes,
        final_edges,
        old_input,
        old_output,
        representatives,
    )

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    previous = parents[key_node]
    while previous != key_node:
        dept = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[dept] = parent
        previous = parents[previous]


def _find_cycle(
    parents: List[int], length: int, current_nodes: List[bool]
) -> Tuple[bool, List[int]]:

    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, list(cycle)
