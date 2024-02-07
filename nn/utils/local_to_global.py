import torch
import torch.nn as nn

from nn.utils.geometry import rotate


class Globalizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def forward(self, x, R):
        return torch.cat([rotate(xi, R) for xi in x.split(self.num_dims, dim=-1)], -1)
