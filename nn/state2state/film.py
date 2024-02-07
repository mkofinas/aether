import torch
import torch.nn as nn


class FilmedNetwork(nn.Module):
    """Feature-wise linear modulation

    x: inputs
    z: conditioning variable
    F(x, z) = gamma(z) * x + beta(z)
    """
    def __init__(self, x_size, z_size, hidden_size, out_size):
        super().__init__()

        self.linear_1 = nn.Linear(x_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_size)

        self.film_1 = FiLM(hidden_size, z_size, hidden_size)
        self.film_2 = FiLM(hidden_size, z_size, hidden_size)

        self.act_1 = nn.SiLU()
        self.act_2 = nn.SiLU()

    def forward(self, x, z):
        y = self.linear_1(x)
        y = self.film_1(y, z)
        y = self.act_1(y)
        y = self.linear_2(y)
        y = self.film_2(y, z)
        y = self.act_2(y)
        y = self.linear_3(y)
        return y


class FiLM(nn.Module):
    """Feature-wise linear modulation

    x: inputs
    z: conditioning variable
    F(x, z) = gamma(z) * x + beta(z)
    """
    def __init__(self, x_size, z_size, hidden_size):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * x_size),
        )

    def forward(self, x, z):
        """
        x: B x S x D_1
        z: B x 1 x D_2
        """
        modulator = self.modulator(z)
        gamma, beta = torch.chunk(modulator, 2, dim=-1)
        return (1.0 + gamma) * x + beta
