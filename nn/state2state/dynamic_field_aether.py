import numpy as np
import torch
import torch.nn as nn

from nn.state2state.graph_pool import GraphSummary
from nn.state2state.film import FilmedNetwork
from nn.state2state.aether import AetherLocalizer
from nn.state2state.locs.locs import Globalizer, GNN


class LatentFieldNetwork(nn.Module):
    def __init__(self, num_dims, hidden_size, class_embedding_dim):
        super().__init__()

        self.num_dims = num_dims

        self.summary_dim = hidden_size

        self.summary_net = GraphSummary(2 * num_dims, self.summary_dim)

        self.wrapper = FilmedNetwork(
            2 * num_dims + class_embedding_dim, self.summary_dim, hidden_size, num_dims
        )

        self.class_embedding = nn.Embedding(3, class_embedding_dim)

    @staticmethod
    def charge_to_index(charges):
        return (charges + 1).long()

    def forward(self, x, class_idx, num_nodes):
        positions, velocities = torch.split(x, [self.num_dims, self.num_dims], dim=-1)
        emb_class = self.class_embedding(self.charge_to_index(class_idx)).squeeze(1)

        graph_summary = self.summary_net(x.reshape(-1, num_nodes, x.shape[-1]))
        batch = (
            torch.arange(0, graph_summary.shape[0])
            .repeat_interleave(num_nodes)
            .to(graph_summary.device)
        )

        graph_summary = graph_summary[batch]

        field_inputs = torch.cat([positions, velocities, emb_class], dim=-1)
        predicted_field = self.wrapper(field_inputs, graph_summary)

        return predicted_field


class DynamicFieldAether(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims, device="cuda"):
        super().__init__()
        self.gnn = GNN(
            input_size,
            hidden_size,
            dropout_prob,
            num_dims,
            additional_features=num_dims,
        )
        self.localizer = AetherLocalizer(num_dims)
        self.globalizer = Globalizer(num_dims)
        self.num_dims = num_dims

        class_embedding_dim = 16
        field_hidden_size = 32
        self.field_net = LatentFieldNetwork(
            num_dims, field_hidden_size, class_embedding_dim
        )

        self.to(device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        return str(params)

    def forward(
        self, h, x, edges, vel, edge_attr_orig, charges, num_nodes
    ):
        """inputs shape: [batch_size, num_objects, input_size]"""

        inputs = torch.cat([x, vel], dim=-1)

        # Predict field
        predicted_field = self.field_net(inputs, charges, num_nodes)
        extended_inputs = torch.cat([inputs, predicted_field], dim=-1)
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(extended_inputs, edges)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)

        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)
        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = x + pred
        return outputs
