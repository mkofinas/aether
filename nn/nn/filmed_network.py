import torch.nn as nn

from nn.nn.film import FiLM
from nn.nn.film import ConcatFiLM


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


class ConcatFilmedNetwork(nn.Module):
    """Feature-wise linear modulation (concatenation only)

    x: inputs
    z: conditioning variable
    F(x, z) = x + beta(z)
    """
    def __init__(self, x_size, z_size, hidden_size, out_size):
        super().__init__()

        self.linear_1 = nn.Linear(x_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_size)

        self.film_1 = ConcatFiLM(hidden_size, z_size, hidden_size)
        self.film_2 = ConcatFiLM(hidden_size, z_size, hidden_size)

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
