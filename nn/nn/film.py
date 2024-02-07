import torch.nn as nn


class ConcatFiLM(nn.Module):
    """Feature-wise linear modulation (concatentation only)

    x: inputs
    z: conditioning variable
    F(x, z) = x + beta(z)
    """
    def __init__(self, x_size, z_size, hidden_size):
        super().__init__()

        # Additive modulator
        self.beta = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, x_size),
        )

    def forward(self, x, z):
        """
        x: B x S x D_1
        z: B x 1 x D_2
        """
        z_add = self.beta(z)
        return x + z_add


class FiLM(nn.Module):
    """Feature-wise linear modulation

    x: inputs
    z: conditioning variable
    F(x, z) = gamma(z) * x + beta(z)
    """
    def __init__(self, x_size, z_size, hidden_size):
        super().__init__()

        # Multiplicative modulator
        self.gamma = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, x_size),
        )
        # Additive modulator
        self.beta = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, x_size),
        )

    def forward(self, x, z):
        """
        x: B x S x D_1
        z: B x 1 x D_2
        """
        z_mul = self.gamma(z)
        z_add = self.beta(z)
        return (1.0 + z_mul) * x + z_add
