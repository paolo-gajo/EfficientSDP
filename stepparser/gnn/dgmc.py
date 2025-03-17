import torch
import torch.nn as nn


class DGM_c(nn.Module):
    def __init__(self, input_dim=None, embed_f=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.input_dim = input_dim
        self.embed_f = embed_f
        self.distance = distance
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.lin = nn.Linear(input_dim, input_dim)
        
        # self.centroid = None
        # self.scale = None
        # self.scale = nn.Parameter(torch.tensor(-1).float(), requires_grad=False)
        # self.centroid = nn.Parameter(
        #     torch.zeros((1, 1, DGM_c.input_dim)).float(), requires_grad=False
        # )

    def forward(self, x, A):
        if A is not None:
            x = self.embed_f(x, A)
        else:
            A = torch.eye(x.shape[0])

        # estimate normalization parameters
        # if self.scale < 0:
        #     self.centroid.data = x.mean(-2, keepdim=True).detach()
        #     self.scale.data = (0.9 / (x - self.centroid).abs().max()).detach()
        x = nn.functional.relu(self.lin(x))
        if self.distance == "hyperbolic":
            D, _x = pairwise_poincare_distances(x)
        else:
            D, _x = pairwise_euclidean_distances(x)

        A = torch.sigmoid(self.temperature * (self.threshold.abs() - D))

        out_dict = {
            "x": x,
            "adj": A,
        }
        return out_dict


# Euclidean distance
def pairwise_euclidean_distances(x):
    dist = torch.cdist(x, x) ** 2
    return dist, x


# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    x = x / (x_norm * (1 + 1e-2))
    x_norm = (x**2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = (
        torch.arccosh(
            1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))
        )
        ** 2
    )
    return dist, x
