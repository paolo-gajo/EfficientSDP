import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

EPS = 1e-10

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

class Passthrough:
    """Simple pass-through activation that returns input unchanged."""
    def __call__(self, x):
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
            row = torch.stack([torch.full((len_r,), j).to(adj_matrices.device), row_indices], dim=0)
            m_index_list.append(row)
            # Collect corresponding values
            m_value_list.append(row_values)
        
        m_index = torch.cat(m_index_list, dim=1)
        m_index = m_index + shift_m  # Shift indices for batching
        edge_index_list.append(m_index)
        
        m_values = torch.cat(m_value_list, dim=0)
        edge_attr_list.append(m_values)
    
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    
    return edge_index, edge_attr