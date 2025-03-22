import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import List, Set, Tuple

class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, heads, dropout=0.2):
        super().__init__()
        ...