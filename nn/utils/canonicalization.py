import torch

from nn.utils.geometry import (
    cart_to_n_spherical,
    rotation_matrix_to_euler,
    wrap_angles,
    angle_diff,
    rotation_matrix,
)


def canonicalize_inputs(inputs, use_3d=False):
    if use_3d:
        if inputs.size(-1) != 6:
            raise NotImplementedError
        vel = inputs[..., 3:]

        _, theta, phi = cart_to_n_spherical(vel)
        Rinv = rotation_matrix(3, theta, phi)
        r = Rinv.transpose(-1, -2)

        rot_vel = (r @ vel.unsqueeze(-1)).squeeze(-1)
        canon_inputs = torch.cat([torch.zeros_like(inputs[..., :3]), rot_vel], dim=-1)
    else:
        vel = inputs[..., 2:4]
        angle = torch.atan2(vel[..., [1]], vel[..., [0]])
        Rinv = rotation_matrix(2, angle)
        canon_inputs = torch.zeros_like(inputs)
        canon_inputs[..., 2] = torch.norm(vel, dim=-1)
    return canon_inputs, Rinv


def canonicalize_augmented_inputs(inputs, use_3d=False):
    if use_3d:
        vel = inputs[..., 3:6]
        forces = inputs[..., 6:9]

        _, theta, phi = cart_to_n_spherical(vel)
        Rinv = rotation_matrix(3, theta, phi)
        r = Rinv.transpose(-1, -2)

        rot_vel = (r @ vel.unsqueeze(-1)).squeeze(-1)
        rot_forces = (r @ forces.unsqueeze(-1)).squeeze(-1)
        canon_inputs = torch.cat([torch.zeros_like(inputs[..., :3]), rot_vel,
                                  rot_forces], dim=-1)
    else:
        vel = inputs[..., 2:4]
        forces = inputs[..., 4:6]

        angle = torch.atan2(vel[..., [1]], vel[..., [0]])
        Rinv = rotation_matrix(2, angle)
        R = Rinv.transpose(-1, -2)
        canon_inputs = torch.zeros_like(inputs)
        canon_inputs[..., 2] = torch.norm(vel, dim=-1)
        canon_inputs[..., 4:6] = (R @ forces.unsqueeze(-1)).squeeze(-1)
    return canon_inputs, Rinv


def sender_receiver_features(x, send_edges, recv_edges, batched=False):
    """
    batched: used in dynamicvars settings
    """
    if batched:
        x_j = x[send_edges]
        x_i = x[recv_edges]
    else:
        if send_edges.ndim == 2:
            batch_range = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
            x_j = x[batch_range, send_edges]
            x_i = x[batch_range, recv_edges]
        else:
            x_j = x[:, send_edges]
            x_i = x[:, recv_edges]

    return x_j, x_i


def create_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    x_j, x_i = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    # recv_yaw is the yaw angle, approximated via the velocity vector
    recv_yaw = torch.atan2(x_i[..., [3]], x_i[..., [2]])
    r = rotation_matrix(2, recv_yaw).transpose(-1, -2)

    # delta_yaw is the signed difference in yaws
    delta_yaw = angle_diff(x_i[..., 2:], x_j[..., 2:]).unsqueeze(-1)
    rotated_relative_positions = (r @ (x_j[..., :2] - x_i[..., :2]).unsqueeze(-1)).squeeze(-1)
    node_distance = torch.norm(x_j[..., :2] - x_i[..., :2], dim=-1, keepdim=True)
    # delta_theta is the rotated azimuth. Subtracting the receiving yaw angle
    # is equal to a rotation
    delta_theta = (
        torch.atan2(x_j[..., 1] - x_i[..., 1], x_j[..., 0] - x_i[..., 0]).unsqueeze(-1)
        - recv_yaw
    )
    delta_theta = wrap_angles(delta_theta, normalize=True)

    rotated_velocities = (r @ x_j[..., 2:].unsqueeze(-1)).squeeze(-1)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         delta_yaw,
         node_distance,
         delta_theta,
         rotated_velocities,
        ], -1)
    return edge_attr


def create_augmented_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    x_j, x_i = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    recv_yaw = torch.atan2(x_i[..., [3]], x_i[..., [2]])
    r = rotation_matrix(2, recv_yaw).transpose(-1, -2)

    # delta_yaw is the signed difference in yaws
    delta_yaw = angle_diff(x_i[..., 2:4], x_j[..., 2:4]).unsqueeze(-1)
    rotated_relative_positions = (r @ (x_j[..., :2] - x_i[..., :2]).unsqueeze(-1)).squeeze(-1)
    node_distance = torch.norm(x_j[..., :2] - x_i[..., :2], dim=-1, keepdim=True)
    # delta_theta is the rotated azimuth
    delta_theta = (
        torch.atan2(x_j[..., 1] - x_i[..., 1], x_j[..., 0] - x_i[..., 0]).unsqueeze(-1)
        - recv_yaw
    )
    delta_theta = wrap_angles(delta_theta, normalize=True)
    rotated_velocities = (r @ x_j[..., 2:4].unsqueeze(-1)).squeeze(-1)
    rotated_forces = (r @ x_j[..., 4:6].unsqueeze(-1)).squeeze(-1)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         delta_yaw,
         node_distance,
         delta_theta,
         rotated_velocities,
         rotated_forces,
        ], -1)
    return edge_attr


def create_augmented_3d_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    send_embed, recv_embed = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    _, send_yaw, send_pitch = cart_to_n_spherical(send_embed[..., 3:6])
    _, recv_yaw, recv_pitch = cart_to_n_spherical(recv_embed[..., 3:6])
    r = rotation_matrix(3, recv_yaw, recv_pitch).transpose(-1, -2)

    node_distance, _, _ = cart_to_n_spherical(send_embed[..., :3] - recv_embed[..., :3])

    send_r = rotation_matrix(3, send_yaw, send_pitch).transpose(-1, -2)
    rotated_euler = rotation_matrix_to_euler(r @ send_r, num_dims=3, normalize=False)

    rotated_relative_positions = (r @ (send_embed[..., :3] - recv_embed[..., :3]).unsqueeze(-1)).squeeze(-1)
    rotated_velocities = (r @ send_embed[..., 3:6].unsqueeze(-1)).squeeze(-1)
    rotated_forces = (r @ send_embed[..., 6:9].unsqueeze(-1)).squeeze(-1)
    # Theta: azimuth, phi: elevation
    _, delta_theta, delta_phi = cart_to_n_spherical(rotated_relative_positions)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         rotated_euler,
         node_distance,
         delta_theta,
         delta_phi,
         rotated_velocities,
         rotated_forces,
        ], -1)
    return edge_attr


def create_3d_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    send_embed, recv_embed = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    _, send_yaw, send_pitch = cart_to_n_spherical(send_embed[..., 3:])
    _, recv_yaw, recv_pitch = cart_to_n_spherical(recv_embed[..., 3:])
    r = rotation_matrix(3, recv_yaw, recv_pitch).transpose(-1, -2)

    node_distance, _, _ = cart_to_n_spherical(send_embed[..., :3] - recv_embed[..., :3])

    send_r = rotation_matrix(3, send_yaw, send_pitch).transpose(-1, -2)
    rotated_euler = rotation_matrix_to_euler(r @ send_r, num_dims=3, normalize=False)

    rotated_relative_positions = (r @ (send_embed[..., :3] - recv_embed[..., :3]).unsqueeze(-1)).squeeze(-1)
    rotated_velocities = (r @ send_embed[..., 3:].unsqueeze(-1)).squeeze(-1)
    # Theta: azimuth, phi: elevation
    _, delta_theta, delta_phi = cart_to_n_spherical(rotated_relative_positions)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         rotated_euler,
         node_distance,
         delta_theta,
         delta_phi,
         rotated_velocities,
        ], -1)
    return edge_attr
