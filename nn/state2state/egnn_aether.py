import logging

import numpy as np
import torch
import torch.nn as nn

from nn.state2state.aether import FieldNetwork
from nn.state2state.gcl import E_GCL_vel_field


class EGNN_vel_Aether(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        num_dims=3,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL_vel_field(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    num_dims=num_dims,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )

        class_embedding_dim = 16
        field_hidden_size = 32
        self.field_net = FieldNetwork(num_dims, field_hidden_size, class_embedding_dim)

        self.to(self.device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        logging.info("Network Size {}".format(params))
        return str(params)

    def forward(self, h, x, edges, vel, edge_attr, charges):
        inputs = torch.cat([x, vel], dim=-1)
        # Predict field
        predicted_field = self.field_net(inputs, charges)

        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h, edges, x, vel, edge_attr=edge_attr, predicted_field=predicted_field
            )
        return x
