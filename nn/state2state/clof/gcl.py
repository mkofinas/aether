import torch
import torch.nn as nn

from nn.state2state.gnn.gcl import unsorted_segment_mean
from nn.state2state.egnn.gcl import E_GCL


class Clof_GCL(E_GCL):
    """
    Basic message passing module of ClofNet.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh, out_basis_dim=3)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.layer_norm = nn.LayerNorm(hidden_nf)

    def coord2localframe(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_cross = torch.cross(coord[row], coord[col])
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
            cross_norm = (
                torch.sqrt(torch.sum((coord_cross)**2, 1).unsqueeze(1))) + 1
            coord_cross = coord_cross / cross_norm

        coord_vertical = torch.cross(coord_diff, coord_cross)

        return radial, coord_diff, coord_cross, coord_vertical

    def coord_model(self, coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat):
        row, col = edge_index
        coff = self.coord_mlp(edge_feat)
        trans = coord_diff * coff[:, :1] + coord_cross * coff[:, 1:2] + coord_vertical * coff[:, 2:3]
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        residue = h
        # h = self.layer_norm(h)
        radial, coord_diff, coord_cross, coord_vertical = self.coord2localframe(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, coord_cross, coord_vertical, edge_feat)

        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        h = residue + h
        h = self.layer_norm(h)
        return h, coord, edge_attr

