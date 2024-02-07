import torch
import torch.nn as nn

from dnri.utils.canonicalization import (
    canonicalize_inputs, create_edge_attr_pos_vel, create_3d_edge_attr_pos_vel
)


class OracleLocalizer(nn.Module):
    # Boolean Tuple Keys: use_3d
    _global_to_local_fn = {
        0: create_edge_attr_pos_vel,
        1: create_3d_edge_attr_pos_vel,
    }

    # Tuple Keys: use_3d & position_representation (cartesian vs polar)
    _edge_pos_idx_fn = {
        (0, 'cart'): [0, 1, 2],
        (0, 'polar'): [2, 3, 4],
        (1, 'cart'): [0, 1, 2, 3, 4, 5],
        (1, 'polar'): [3, 4, 5, 6, 7, 8],
    }

    def __init__(self, num_vars, num_sources, use_3d=False, pos_representation='polar'):
        super().__init__()
        self.num_vars = num_vars
        self.num_sources = num_sources
        self.use_3d = use_3d

        self.send_edges, self.recv_edges = torch.where(
            ~torch.eye(self.num_vars, dtype=bool))

        extra_edges = torch.cartesian_prod(
            torch.arange(self.num_vars, self.num_vars + self.num_sources),
            torch.arange(self.num_vars)).T
        self._send_edges = torch.cat([self.send_edges, extra_edges[0]])
        self._recv_edges = torch.cat([self.recv_edges, extra_edges[1]])

        if pos_representation not in ('cart', 'polar'):
            raise ValueError
        self.edge_pos_idx = self._edge_pos_idx_fn[
            (self.use_3d, pos_representation)]
        self.num_dims = 3 if self.use_3d else 2
        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        self.num_relative_features = 3 * self.num_dims + self.num_orientations
        self.num_pos_features = self.num_dims + self.num_orientations

        self._global_to_local = self._global_to_local_fn[int(self.use_3d)]

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def forward(self, x, extended_inputs):
        # NOTE: Features should be "properly" normalized
        rel_feat, Rinv = canonicalize_inputs(x, self.use_3d)

        edge_attr = self._global_to_local(extended_inputs, self._send_edges, self._recv_edges)
        edge_pos = edge_attr[..., self.edge_pos_idx]
        edge_attr = torch.cat([edge_attr, rel_feat[:, self._recv_edges]], -1)

        field_edge_attr = edge_attr[:, self.recv_edges.shape[0]:]
        field_edge_pos = edge_pos[:, self.recv_edges.shape[0]:]
        edge_attr = edge_attr[:, :self.recv_edges.shape[0]]
        edge_pos = edge_pos[:, :self.recv_edges.shape[0]]

        return rel_feat, Rinv, edge_attr, edge_pos, field_edge_attr, field_edge_pos
