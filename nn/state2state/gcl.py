import torch
import torch.nn as nn

from nn.state2state.egnn.gcl import E_GCL


class E_GCL_vel_field(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        num_dims=3,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
    ):
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_att_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
        )
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf + num_dims, hidden_nf), act_fn, nn.Linear(hidden_nf, 1)
        )

    def forward(
        self,
        h,
        edge_index,
        coord,
        vel,
        edge_attr=None,
        node_attr=None,
        predicted_field=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(
            h[row],
            h[col],
            radial,
            torch.cat([edge_attr, predicted_field[row], predicted_field[col]], dim=-1),
        )
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        coord += self.coord_mlp_vel(torch.cat([h, predicted_field], dim=-1)) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr
