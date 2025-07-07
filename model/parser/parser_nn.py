import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from typing import List, Tuple, Optional
import math

EPS = 1e-10

class LayerNormLSTM(nn.Module):
    """
    A multi-layer LSTM with Layer Normalization applied to the outputs of each layer.
    This module mimics nn.LSTM's interface but inserts nn.LayerNorm between layers.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

        # Create per-layer LSTM and LayerNorm modules
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer in range(num_layers):
            # Input size for this layer
            layer_in_size = input_size if layer == 0 else hidden_size * self.num_directions
            # Single-layer LSTM
            lstm_layer = nn.LSTM(
                input_size=layer_in_size,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
            self.layers.append(lstm_layer)
        for layer in range(num_layers):
            # LayerNorm over the full hidden dimension (including directions)
            self.layer_norms.append(nn.LayerNorm(hidden_size * self.num_directions))
        # self.layer_norms.append(Passthrough())

    def forward(self, x, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            x (Tensor or PackedSequence): Input sequence of shape
                (batch, seq_len, input_size), or a PackedSequence.
            hx (tuple): Optional initial states (h0, c0) each of shape
                (num_layers * num_directions, batch, hidden_size).
        Returns:
            out: Output sequence (Tensor or PackedSequence).
            (h_n, c_n): Final hidden and cell states.
        """
        # Handle PackedSequence inputs
        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, lengths = pad_packed_sequence(x, batch_first=self.batch_first)

        batch, seq_len, _ = x.size()
        # Initialize states per layer if not provided
        if hx is None:
            h_prev = [None] * (self.num_layers + 1)
            c_prev = [None] * (self.num_layers + 1)
        else:
            h0_all, c0_all = hx
            # Split combined states into per-layer states
            h_prev = list(h0_all.view(self.num_layers, self.num_directions, batch, self.hidden_size))
            c_prev = list(c0_all.view(self.num_layers, self.num_directions, batch, self.hidden_size))

        layer_input = x
        hidden_states = []
        cell_states = []

        # Forward through each layer
        for layer_idx, (lstm, ln) in enumerate(zip(self.layers, self.layer_norms)):
            init_state = None
            if h_prev[layer_idx - 1] is not None:
                init_state = (
                    h_prev[layer_idx - 1].contiguous(),
                    c_prev[layer_idx - 1].contiguous()
                )
            # LSTM forward
            out, (h_n_layer, c_n_layer) = lstm(layer_input, init_state)
            # Apply LayerNorm
            out = ln(out)
            # Apply dropout (except after last layer)
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                out = self.dropout(out)
            # Save states
            hidden_states.append(h_n_layer)
            h_prev[layer_idx] = h_n_layer
            c_prev[layer_idx] = c_n_layer
            cell_states.append(c_n_layer)
            # Input for next layer
            layer_input = out

        # Stack final states
        h_n = torch.stack(hidden_states, dim=0).view(self.num_layers * self.num_directions, batch, self.hidden_size)
        c_n = torch.stack(cell_states, dim=0).view(self.num_layers * self.num_directions, batch, self.hidden_size)

        if is_packed:
            # Re-pack the sequence
            out = pack_padded_sequence(layer_input, lengths, batch_first=self.batch_first, enforce_sorted=False)
        else:
            out = layer_input

        return out, (h_n, c_n)

class LayerNormRNN(nn.Module):
    """
    A multi-layer RNN with Layer Normalization applied to the outputs of each layer.
    This module mimics nn.RNN's interface but inserts nn.LayerNorm between layers.

    Args:
        input_size:     The number of expected features in the input x.
        hidden_size:    The number of features in the hidden state h.
        num_layers:     Number of recurrent layers.
        nonlinearity:   The nonlinearity to use ('tanh' or 'relu'). Default 'tanh'.
        bidirectional:  If True, becomes a bidirectional RNN.
        dropout:        If non-zero, introduces a Dropout layer on the outputs of each
                        RNN layer except the last layer.
        batch_first:    If True, input and output tensors are provided as (batch, seq, feature).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        super().__init__()
        self.num_layers    = num_layers
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_first   = batch_first

        # Dropout applied between layers (not after last)
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

        # Build per-layer RNNs and LayerNorms
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions
            rnn = nn.RNN(
                input_size=in_size,
                hidden_size=hidden_size,
                num_layers=1,
                nonlinearity=nonlinearity,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
            self.layers.append(rnn)
            # LayerNorm over the hidden feature dimension (times directions)
            self.layer_norms.append(nn.LayerNorm(hidden_size * self.num_directions))

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  Input tensor of shape (batch, seq_len, input_size) if batch_first=True.
            h0: Optional initial hidden state of shape
                (num_layers * num_directions, batch, hidden_size).
        Returns:
            out:  Output tensor of shape (batch, seq_len, num_directions * hidden_size).
            h_n:  Final hidden state of shape (num_layers * num_directions, batch, hidden_size).
        """
        batch_size, seq_len, _ = x.size()

        # Prepare initial hidden states per layer
        if h0 is None:
            # Each layer will initialize its own zeros
            h_prev = [None] * self.num_layers
        else:
            # Split the combined h0 into per-layer chunks
            # shape of h0: (num_layers * num_directions, batch, hidden_size)
            h_prev = list(h0.view(self.num_layers, self.num_directions, batch_size, self.hidden_size))

        layer_input = x
        final_states = []

        # Iterate through layers
        for layer_idx, (rnn, ln) in enumerate(zip(self.layers, self.layer_norms)):
            # pick initial state for this layer, if provided
            init_h = None
            if h_prev[layer_idx] is not None:
                # rnn expects shape (num_directions, batch, hidden_size)
                init_h = h_prev[layer_idx].contiguous()

            # Forward through RNN layer
            out, h_n = rnn(layer_input, init_h)

            # Apply layer normalization on the feature dimension
            out = ln(out)

            # Apply dropout except after last layer
            if self.dropout is not None and layer_idx < self.num_layers - 1:
                out = self.dropout(out)

            # Collect final hidden state
            final_states.append(h_n)

            # The output of this layer is the input to next
            layer_input = out

        # Stack hidden states from all layers
        # final_states is list of length num_layers, each of shape (num_directions, batch, hidden_size)
        h_n = torch.stack(final_states, dim=0).view(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        )

        return out, h_n


class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.
    This unified class supports multiple bias types for flexibility.

    # Parameters

    matrix_1_dim : `int`, required
        The dimension of the matrix `X`. This is `X.size()[-1]` - the length
        of the vector that will go into the similarity computation.
    matrix_2_dim : `int`, required
        The dimension of the matrix `Y`. This is `Y.size()[-1]` - the length
        of the vector that will go into the similarity computation.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the similarity calculation.
    use_input_biases : `bool`, optional (default = `False`)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    out_features : `int`, optional (default = `1`)
        The number of output classes.
    bias_type : `str`, optional (default = "simple")
        Type of bias to use:
        - "simple": A single scalar bias (original BilinearMatrixAttention)
        - "gnn": Separate biases for both matrices (BilinearMatrixAttentionGNN)
        - "dozat": Bias only for matrix_1 (BilinearMatrixAttentionDozatManning)
        - "none": No bias
    """

    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        activation=None,
        use_input_biases: bool = False,
        out_features: int = 1,
        bias_type: str = 'simple',
        arc_norm: bool = True,
    ) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1
        
        if out_features == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(
                torch.Tensor(out_features, matrix_1_dim, matrix_2_dim)
            )
        
        self.arc_norm = arc_norm
        self.scale_norm = math.sqrt((matrix_1_dim + matrix_2_dim) / 2) if arc_norm else 1
        
        # Set up bias parameters based on the bias_type
        self.bias_type = bias_type.lower()
        if self.bias_type == "simple":
            self._bias = nn.Parameter(torch.Tensor(1))
            self._bias_1 = None
            self._bias_2 = None
        elif self.bias_type == "gnn":
            self._bias = None
            self._bias_1 = nn.Parameter(torch.Tensor(matrix_1_dim))
            self._bias_2 = nn.Parameter(torch.Tensor(matrix_2_dim))
        elif self.bias_type == "dozat":
            self._bias = None
            self._bias_1 = nn.Parameter(torch.Tensor(matrix_1_dim))
            self._bias_2 = None
        elif self.bias_type == "none":
            self._bias = None
            self._bias_1 = None
            self._bias_2 = None
        else:
            raise ValueError(f"Unsupported bias_type: {bias_type}."
                           "Choose from 'simple', 'gnn', 'dozat', or 'none'.")

        self.activation = activation or Passthrough()
        self.use_input_biases = use_input_biases
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        
        if self._bias is not None:
            self._bias.data.fill_(0)
        if self._bias_1 is not None:
            self._bias_1.data.fill_(0)
        if self._bias_2 is not None:
            self._bias_2.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        if self.use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], dim=-1)
            matrix_2 = torch.cat([matrix_2, bias2], dim=-1)

        # Handle weight dimensionality
        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
            
        # Compute the core bilinear attention
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        final = final.squeeze(1)
        
        # Apply bias based on the selected bias type
        if self.bias_type == "simple":
            result = final + self._bias
        elif self.bias_type == "gnn":
            bias_1 = torch.matmul(matrix_1, self._bias_1.unsqueeze(1))
            bias_2 = torch.matmul(matrix_2, self._bias_2.unsqueeze(1))
            result = final + bias_1 + bias_2
        elif self.bias_type == "dozat":
            bias = torch.matmul(matrix_1, self._bias_1.unsqueeze(1))
            result = final + bias
        else:  # "none"
            result = final
        
        return self.normalize(self.activation(result))

    def normalize(self, adj):
        if self.arc_norm:
                return adj / self.scale_norm
        else:
            return adj


class TrilinearMatrixAttention(nn.Module):
    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        matrix_3_dim: int,
        mode: str,
        activation=None,
        arc_norm: bool = True,
    ) -> None:
        super().__init__()
        
        self.W = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim, matrix_3_dim))
        
        self.arc_norm = arc_norm
        self.scale_norm = math.sqrt((matrix_1_dim + matrix_2_dim + matrix_3_dim) / 3) if arc_norm else 1
        
        self.mode = mode
        self.activation = activation or Passthrough()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self,
                matrix_1: torch.Tensor,
                matrix_2: torch.Tensor,
                matrix_3: torch.Tensor
                ) -> torch.Tensor:

        B, S, D = matrix_1.shape
        square_mask = torch.ones(S, S, dtype=torch.bool, device=matrix_1.device)
        mask = torch.triu(square_mask, diagonal=1)

        if self.mode == 'sib':
            # j < k so we need the mask to be on the jk dimensions
            # which means the last 2
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif self.mode == 'cop':
            # i < k so we need the mask to be on the ik dimensions
            # meaning 0, [1], 2, [3] i.e. the 1st and 3rd
            mask = mask.unsqueeze(1).unsqueeze(0)
        else:
            mask = square_mask
            mask = mask.unsqueeze(0).unsqueeze(0)

        # matrix_1: head representations
        # matrix_2: dep representations
        # matrix_3: head (cop) / dep (sib) / dep-of-dep (gp)

        # delete p with x
        v1U = torch.einsum('bix,pqr -> biqr', matrix_1, self.W) # shape [B, S, D, D]
        # delete q with x
        v1Uv2 = torch.einsum('biqr,bjx -> bijr', v1U, matrix_2) # shape [B, S, S, D]
        # delete r with x
        v3v1Uv2 = torch.einsum('bkx,bijr -> bijk', matrix_3, v1Uv2) # shape [B, S, S, S]

        scores = v3v1Uv2.masked_fill(~mask, 0.0)
        return self.normalize(self.activation(scores))

    def normalize(self, adj):
        if self.arc_norm:
                return adj / self.scale_norm
        else:
            return adj

class GraphNNUnit(nn.Module):
    def __init__(self,
                h_dim,
                d_dim,
                use_residual=True,
                use_layer_norm=False,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.W = nn.Parameter(torch.Tensor(h_dim, d_dim))
        self.B = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        if self.use_residual:
            self.res = nn.Linear(h_dim, h_dim)
        
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(h_dim)
            
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.B)

    def forward(self, H, D):
        H_new = torch.matmul(H, self.W)
        D_new = torch.matmul(D, self.B)
        out = F.tanh((H_new + D_new) / 2)
        if self.use_residual:
            out = out + (H + D) / 2
            # out = out + H
        if self.use_layer_norm:
            return self.layer_norm(out)
        else:
            return out

class SQRTNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.sqrt_norm = math.sqrt(dim)
    
    def forward(self, input: torch.Tensor):
        return input / self.sqrt_norm

class MHABMA(nn.Module):
    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        num_heads: int,
        activation=None,
        use_input_biases: bool = False,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if out_features == 1:
            self._weight_matrix = nn.Parameter(torch.Tensor(num_heads, matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = nn.Parameter(
                torch.Tensor(num_heads, out_features, matrix_1_dim, matrix_2_dim)
            )

        self._bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation or Passthrough()
        self.use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        if self.use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], dim=-1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            matrix_2 = torch.cat([matrix_2, bias2], dim=-1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        B_m, N_m, L_m, d_m = matrix_1.shape
        matrix_1 = matrix_1.reshape(B_m * N_m, L_m, d_m)
        matrix_2 = matrix_2.reshape(B_m * N_m, L_m, d_m)
        weight = self._weight_matrix.unsqueeze(0).expand(B_m, -1, -1, -1)
        B_w, N_w, L_w, d_w = weight.shape
        weight = weight.reshape(B_w * N_w, L_w, d_w)
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.bmm(matrix_1, weight)
        final = torch.bmm(intermediate, matrix_2.transpose(1, 2))
        final_biased = final.squeeze(1) + self._bias
        out = final_biased.reshape(B_m, N_m, L_m, L_m)
        out = out.mean(dim = 1)
        return self.activation(out)

class Passthrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class InputVariationalDropout(torch.nn.Dropout):
    """
    (from AllenNLP)
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):
        """
        Apply dropout to input tensor.

        # Parameters

        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`

        # Returns

        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(
            ones, self.p, self.training, inplace=False
        )
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class HeadSentinelFusion(nn.Module):
    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        """
        Args:
            input_dim: The dimension of the concatenated vector.
            output_dim: The desired output dimension (same as the original head vector dim).
            use_nonlinearity: Whether to apply a non-linear activation after projection.
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.use_nonlinearity = use_nonlinearity

    def forward(self, head_sentinel, pooled_vector):
        # Concatenate the two vectors along the last dimension.
        fused_vector = torch.cat([head_sentinel, pooled_vector], dim=-1)
        projected = self.projection(fused_vector)
        if self.use_nonlinearity:
            projected = F.relu(
                projected
            )  # or another activation function like GELU/Tanh
        return projected

class Passthrough(torch.nn.Module):
    def forward(self, output):
        return output

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

def batch_top_k(adj_matrices: List[torch.Tensor], k: int):
    """
    Returns the top k edges and their corresponding values from batched adjacency matrices.
    
    Args:
        adj_matrices: Batched adjacency matrices of shape [batch_size, seq_len, seq_len]
        k: Number of top edges to select for each node
        
    Returns:
        edge_index: Tensor of shape [2, num_edges] containing the indices of the edges
        edge_attr: Tensor of shape [num_edges] containing the soft values of the edges
    """
    # Get top k values and indices
    if k < 1 or k is None:
        k = adj_matrices.shape[-1]
    top_k_values, top_k_indices = torch.topk(adj_matrices, k, dim=2)
    
    edge_index_list = []
    edge_attr_list = []
    
    for i, (indices, values) in enumerate(zip(top_k_indices, top_k_values)):
        m_index_list = []
        m_value_list = []
        len_m = indices.shape[-1]
        shift_m = i * len_m
        
        for j, (row_indices, row_values) in enumerate(zip(indices, values)):
            len_r = row_indices.shape[-1]
            # Create edge indices
            tmp = torch.full((len_r,), j).to(adj_matrices.device)
            row = torch.stack([tmp, row_indices], dim=0)
            m_index_list.append(row)
            # Collect corresponding values
            m_value_list.append(row_values)
            ...
        
        m_index = torch.cat(m_index_list, dim=1)
        m_index = m_index + shift_m  # Shift indices for batching
        edge_index_list.append(m_index)
        
        m_values = torch.cat(m_value_list, dim=0)
        edge_attr_list.append(m_values)
    
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    
    return edge_index, edge_attr