import numpy as np

import torch
import torch.nn as nn


class FourierFeatureMapper(nn.Module):
    def __init__(self, in_size, out_size, std=1.0):
        super().__init__()
        self._std = std

        self._rng = np.random.default_rng(42)
        B = torch.from_numpy(
            self._rng.normal(0, self._std, size=(in_size, out_size))).float()
        self.register_buffer('B', B)

        self._tau = 2 * np.pi

    def forward(self, x):
        x_proj = (self._tau * x) @ self.B.to(x.device)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
