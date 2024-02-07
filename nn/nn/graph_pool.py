import math

import torch
from torch import nn

from torch_geometric.nn.aggr import AttentionalAggregation


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GraphSummary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.summary_net = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            ),
            nn=nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        )
        self.particle_embedding = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.pe = PositionalEncoding(hidden_size+input_size)

    def forward(self, x):
        """
        x shape: B x N x T x D
        """
        batch_size = x.size(0)
        num_objects = x.size(1)
        num_timesteps = x.size(2)

        batch_idx = torch.arange(x.size(0))[:, None, None].to(x.device).expand(*x.shape[:-1])
        batch_idx = batch_idx.flatten()

        y = self.particle_embedding(x).flatten(0, 1)
        particle_embedding = self.rnn(y)[1].squeeze(0).reshape(batch_size, num_objects, -1)
        particle_embedding = particle_embedding.unsqueeze(2).expand(*particle_embedding.shape[:-1], num_timesteps, particle_embedding.shape[-1])

        augmented_flat_positions = torch.cat(
            [x.flatten(0, 1), particle_embedding.flatten(0, 1)], -1)
        augmented_flat_positions = self.pe(augmented_flat_positions)
        augmented_flat_positions = augmented_flat_positions.flatten(0, 1)
        graph_snapshots = self.summary_net(augmented_flat_positions, index=batch_idx)
        return graph_snapshots
