import os

import numpy as np
import torch

from experiments.utils.normalization.abstract_normalization import AbstractNormalization


class MinMaxNormalization(AbstractNormalization):
    def __init__(self, data_path, params):
        self.ndim = params.get('ndim', 2)

        train_data = torch.load(os.path.join(data_path, 'train_feats'))

        self.loc_max = train_data[:, :, :, :self.ndim].max()
        self.loc_min = train_data[:, :, :, :self.ndim].min()
        self.position_range = self.loc_max - self.loc_min
        self.vel_max = train_data[:, :, :, self.ndim:].max()
        self.vel_min = train_data[:, :, :, self.ndim:].min()
        self.velocity_range = self.vel_max - self.vel_min

    def normalize(self, x):
        norm_pos = (x[:, :, :, :self.ndim] - self.loc_min) * 2.0 / self.position_range - 1.0
        norm_vel = (x[:, :, :, self.ndim:] - self.vel_min) * 2.0 / self.velocity_range - 1.0
        return torch.cat([norm_pos, norm_vel], dim=-1)

    def normalize_positions(self, x):
            return (x - self.loc_min) * 2.0 / self.position_range - 1.0

    def normalize_velocities(self, x):
            return (x - self.vel_min) * 2.0 / self.velocity_range - 1.0

    @staticmethod
    def cat(x, y):
        if isinstance(x, np.ndarray):
            return np.concatenate([x, y], axis=-1)
        elif isinstance(x, torch.Tensor):
            return torch.cat([x, y], dim=-1)
        else:
            raise NotImplementedError

    def unnormalize(self, x):
        unnorm_pos = (x[:, :, :, :self.ndim] + 1.0) * self.position_range / 2.0 + self.loc_min
        unnorm_vel = (x[:, :, :, self.ndim:] + 1.0) * self.velocity_range / 2.0 + self.vel_min
        return self.cat(unnorm_pos, unnorm_vel)

    def unnormalize_positions(self, x):
        return (x + 1.0) * self.position_range / 2.0 + self.loc_min

    def unnormalize_velocities(self, x):
        return (x + 1.0) * self.velocity_range / 2.0 + self.vel_min

    def torch_unnormalize(self, x):
        unnorm_pos = (x[:, :, :, :self.ndim] + 1.0) * self.position_range / 2.0 + self.loc_min
        unnorm_vel = (x[:, :, :, self.ndim:] + 1.0) * self.velocity_range / 2.0 + self.vel_min
        return torch.cat([unnorm_pos, unnorm_vel], dim=-1)
