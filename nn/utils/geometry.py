"""Geometric primitives, rotations, cartesian to spherical"""

import numpy as np
import torch


def rotation_matrix(ndim, theta, phi=None, psi=None, /):
    """
    theta, phi, psi: yaw, pitch, roll

    NOTE: We assume that each angle is has the shape [dims] x 1
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if ndim == 2:
        R = torch.stack(
            [
                torch.cat([cos_theta, -sin_theta], -1),
                torch.cat([sin_theta, cos_theta], -1),
            ],
            -2,
        )
        return R
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    R = torch.stack(
        [
            torch.cat([cos_phi * cos_theta, -sin_theta, sin_phi * cos_theta], -1),
            torch.cat([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta], -1),
            torch.cat([-sin_phi, torch.zeros_like(cos_theta), cos_phi], -1),
        ],
        -2,
    )
    return R


def cart_to_n_spherical(x, symmetric_theta=False):
    """Transform Cartesian to n-Spherical Coordinates

    NOTE: Not tested thoroughly for n > 3

    Math convention, theta: azimuth angle, angle in x-y plane

    x: torch.Tensor, [dims] x D
    return rho, theta, phi
    """
    ndim = x.size(-1)

    rho = torch.norm(x, p=2, dim=-1, keepdim=True)

    theta = torch.atan2(x[..., [1]], x[..., [0]])
    if not symmetric_theta:
        theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    if ndim == 2:
        return rho, theta

    cum_sqr = (
        rho
        if ndim == 3
        else torch.sqrt(torch.cumsum(torch.flip(x**2, [-1]), dim=-1))[..., 2:]
    )
    EPS = 1e-7
    phi = torch.acos(torch.clamp(x[..., 2:] / (cum_sqr + EPS), min=-1.0, max=1.0))

    return rho, theta, phi


def velocity_to_rotation_matrix(vel):
    num_dims = vel.size(-1)
    orientations = cart_to_n_spherical(vel)[1:]
    R = rotation_matrix(num_dims, *orientations)
    return R


def rotation_matrix_to_euler(R, num_dims, normalize=True):
    """Convert rotation matrix to euler angles

    In 3 dimensions, we follow the ZYX convention
    Functionally identical to matrix_to_euler_angles(R, 'ZYX') from PyTorch3D

    ```py
    from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles
    euler = matrix_to_euler_angles(R, 'ZYX')
    ```
    """
    if num_dims == 2:
        euler = torch.atan2(R[..., 1, [0]], R[..., 0, [0]])
    else:
        euler = torch.stack(
            [
                torch.atan2(R[..., 1, 0], R[..., 0, 0]),
                torch.asin(-R[..., 2, 0]),
                torch.atan2(R[..., 2, 1], R[..., 2, 2]),
            ],
            -1,
        )

    if normalize:
        euler = euler / np.pi
    return euler


def rotate(x, R):
    return torch.einsum("...ij,...j->...i", R, x)


def wrap_angles(theta, normalize=False):
    theta = theta + (theta <= -np.pi).type_as(theta) * (2 * np.pi)
    theta = theta - (theta > np.pi).type_as(theta) * (2 * np.pi)
    if normalize:
        theta = theta / np.pi
    return theta


def angle_diff(v1, v2):
    # x1 = v1[..., 0]
    # y1 = v1[..., 1]
    # x2 = v2[..., 0]
    # y2 = v2[..., 1]
    # return torch.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    delta_angle = (torch.atan2(v2[..., 1], v2[..., 0])
                   - torch.atan2(v1[..., 1], v1[..., 0]))
    delta_angle[delta_angle >= np.pi] -= 2 * np.pi
    delta_angle[delta_angle < -np.pi] += 2 * np.pi
    delta_angle = delta_angle / np.pi
    return delta_angle
