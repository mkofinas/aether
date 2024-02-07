import numpy as np
import torch
import torch.nn as nn

from nn.state2state.locs.locs import Localizer, Globalizer, GNN
from nn.utils.geometry import (
    velocity_to_rotation_matrix,
    rotate,
    rotation_matrix_to_euler,
    cart_to_n_spherical,
)


class AetherLocalizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, velocities, and forces
        self.num_relative_features = 4 * self.num_dims + self.num_orientations

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def sender_receiver_features(self, x):
        x_j = x[self.send_edges]
        x_i = x[self.recv_edges]
        return x_j, x_i

    def canonicalize_inputs(self, inputs):
        vel = inputs[..., self.num_dims : 2 * self.num_dims]
        forces = inputs[..., 2 * self.num_dims : 3 * self.num_dims]
        R = velocity_to_rotation_matrix(vel)
        Rinv = R.transpose(-1, -2)

        canon_vel = rotate(vel, Rinv)
        canon_forces = rotate(forces, Rinv)
        canon_inputs = torch.cat(
            [
                torch.zeros_like(canon_vel),
                canon_vel,
                canon_forces,
            ],
            dim=-1,
        )

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
        rotated_forces = rotate(x_j[..., 2 * self.num_dims : 3 * self.num_dims], R_inv)

        edge_attr = torch.cat(
            [
                rotated_relative_positions,
                rotated_euler,
                node_distance,
                spherical_relative_positions,
                rotated_velocities,
                rotated_forces,
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
Neural Field
"""


class FieldNetwork(nn.Module):
    def __init__(self, num_dims, hidden_size, class_embedding_dim):
        super().__init__()

        self.num_dims = num_dims
        self.net = nn.Sequential(
            nn.Linear(2 * num_dims + class_embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_dims),
        )

        self.class_embedding = nn.Embedding(3, class_embedding_dim)

    @staticmethod
    def charge_to_index(charges):
        return (charges + 1).long()

    def forward(self, x, class_idx):
        positions, velocities = torch.split(x, [self.num_dims, self.num_dims], dim=-1)

        emb_class = self.class_embedding(self.charge_to_index(class_idx)).squeeze(1)
        field_inputs = torch.cat([positions, velocities, emb_class], dim=-1)

        predicted_field = self.net(field_inputs)
        return predicted_field


"""
Aether
"""


class Aether(nn.Module):
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
        self.field_net = FieldNetwork(num_dims, field_hidden_size, class_embedding_dim)

        self.to(device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        return str(params)

    def forward(self, h, x, edges, vel, edge_attr_orig, charges):
        """inputs shape: [batch_size, num_objects, input_size]"""
        inputs = torch.cat([x, vel], dim=-1)
        # Predict field
        predicted_field = self.field_net(inputs, charges)
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


class ParallelAether(nn.Module):
    """Ablation study"""
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims, device="cuda"):
        super().__init__()
        self.gnn = GNN(
            input_size,
            hidden_size,
            dropout_prob,
            num_dims,
            additional_features=num_dims,
        )
        self.localizer = Localizer(num_dims)
        self.globalizer = Globalizer(num_dims)
        self.num_dims = num_dims

        class_embedding_dim = 16
        field_hidden_size = 32
        self.field_net = FieldNetwork(num_dims, field_hidden_size, class_embedding_dim)

        self.to(device)
        self.params = self.__str__()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Network Size", params)
        return str(params)

    def forward(self, h, x, edges, vel, edge_attr_orig, charges):
        """inputs shape: [batch_size, num_objects, input_size]"""

        inputs = torch.cat([x, vel], dim=-1)
        # Predict field
        predicted_field = self.field_net(inputs, charges)
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs, edges)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)

        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)
        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = x + pred + predicted_field
        return outputs
