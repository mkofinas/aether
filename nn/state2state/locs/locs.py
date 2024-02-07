import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from nn.utils.geometry import (
    cart_to_n_spherical,
    velocity_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotate,
)
from nn.utils.local_to_global import Globalizer

"""
Transformation from global to local coordinates and vice versa
"""


class Localizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        self.num_relative_features = 3 * self.num_dims + self.num_orientations

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def sender_receiver_features(self, x):
        x_j = x[self.send_edges]
        x_i = x[self.recv_edges]
        return x_j, x_i

    def canonicalize_inputs(self, inputs):
        vel = inputs[..., self.num_dims : 2 * self.num_dims]
        R = velocity_to_rotation_matrix(vel)
        Rinv = R.transpose(-1, -2)

        canon_vel = rotate(vel, Rinv)
        canon_inputs = torch.cat([torch.zeros_like(canon_vel), canon_vel], dim=-1)

        return canon_inputs, R

    def create_edge_attr(self, x):
        x_j, x_i = self.sender_receiver_features(x)

        # We approximate orientations via the velocity vector
        R = velocity_to_rotation_matrix(x_i[..., self.num_dims : 2 * self.num_dims])
        R_inv = R.transpose(-1, -2)

        # Positions
        relative_positions = x_j[..., : self.num_dims] - x_i[..., : self.num_dims]
        rotated_relative_positions = rotate(relative_positions, R_inv)

        # Orientations
        send_R = velocity_to_rotation_matrix(
            x_j[..., self.num_dims : 2 * self.num_dims]
        )
        rotated_orientations = R_inv @ send_R
        rotated_euler = rotation_matrix_to_euler(rotated_orientations, self.num_dims)

        # Rotated relative positions in spherical coordinates
        node_distance = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)
        spherical_relative_positions = torch.cat(
            cart_to_n_spherical(rotated_relative_positions, symmetric_theta=True)[1:],
            -1,
        )

        # Velocities
        rotated_velocities = rotate(x_j[..., self.num_dims : 2 * self.num_dims], R_inv)

        edge_attr = torch.cat(
            [
                rotated_relative_positions,
                rotated_euler,
                node_distance,
                spherical_relative_positions,
                rotated_velocities,
            ],
            -1,
        )
        return edge_attr

    def forward(self, x, edges):
        self.set_edge_index(*edges)
        rel_feat, R = self.canonicalize_inputs(x)
        edge_attr = self.create_edge_attr(x)

        edge_attr = torch.cat([edge_attr, rel_feat[self.recv_edges]], -1)
        return rel_feat, R, edge_attr


"""
LoCS Graph Neural Network
"""


class LoCS(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims, device="cuda"):
        super().__init__()
        self.gnn = GNN(
            input_size,
            hidden_size,
            dropout_prob,
            num_dims,
            additional_features=0,
            out_size=0,
        )
        self.localizer = Localizer(num_dims)
        self.globalizer = Globalizer(num_dims)
        self.num_dims = num_dims
        self.to(device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        return str(params)

    def forward(self, h, x, edges, vel, edge_attr_orig):
        """inputs shape: [batch_size, num_objects, input_size]"""
        inputs = torch.cat([x, vel], dim=-1)
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs, edges)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)

        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)
        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = x + pred
        return outputs


class GNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout_prob,
        num_dims,
        additional_features=0,
        out_size=0,
    ):
        super().__init__()
        out_size = input_size // 2 if out_size == 0 else out_size

        num_orientations = num_dims * (num_dims - 1) // 2
        self.num_relative_features = input_size + num_dims + num_orientations

        initial_edge_features = 2

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

        self.layer_1 = GNNLayer(
            input_size + additional_features,
            hidden_size,
            only_edge_attr=True,
            num_edge_features=self.num_relative_features
            + input_size
            + initial_edge_features
            + 2 * additional_features,
        )
        self.layer_2 = GNNLayer(hidden_size, hidden_size)
        self.layer_3 = GNNLayer(hidden_size, hidden_size)
        self.layer_4 = GNNLayer(hidden_size, hidden_size)

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size * num_objects, input_size]
        """
        x, edge_attr = self.layer_1(x, edge_attr, edges)
        x, edge_attr = self.layer_2(x, edge_attr, edges)
        x, edge_attr = self.layer_3(x, edge_attr, edges)
        x, edge_attr = self.layer_4(x, edge_attr, edges)

        # Output MLP
        pred = self.out_mlp(x)
        return pred


class GNNLayer(nn.Module):
    def __init__(
        self, input_size, hidden_size, only_edge_attr=False, num_edge_features=0
    ):
        super().__init__()

        # Neural Network Layers
        self.only_edge_attr = only_edge_attr
        num_edge_features = num_edge_features if only_edge_attr else 3 * hidden_size
        self.message_fn = nn.Sequential(
            # nn.LayerNorm(num_edge_features),
            nn.Linear(num_edge_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

        self.res = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )

        self.update_fn = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size, num_objects, input_size]
        """
        send_edges, recv_edges = edges
        if not self.only_edge_attr:
            edge_attr = torch.cat([x[send_edges], x[recv_edges], edge_attr], dim=-1)

        edge_attr = self.message_fn(edge_attr)
        message_aggr = scatter(
            edge_attr, recv_edges.to(x.device), dim=0, reduce="mean"
        ).contiguous()

        x = self.res(x) + message_aggr
        x = x + self.update_fn(x)

        return x, edge_attr
