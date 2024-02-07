import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation


class GraphSummary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.summary_net = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            ),
            nn=nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        )

    def forward(self, x):
        """
        x shape: B x N x D
        """
        batch_idx = torch.arange(x.size(0))[:, None].to(x.device).expand(*x.shape[:-1]).flatten()
        graph_snapshots = self.summary_net(x.flatten(0, -2), index=batch_idx)
        return graph_snapshots
