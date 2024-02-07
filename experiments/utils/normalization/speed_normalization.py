import os

import numpy as np
import torch

from experiments.utils.normalization.abstract_normalization import AbstractNormalization


class SpeedNormalization(AbstractNormalization):
    def __init__(self, data_path, params):
        self.ndim = params.get('ndim', 2)

        train_data = torch.load(os.path.join(data_path, 'train_feats'))
        self.speed_max = np.linalg.norm(train_data[..., self.ndim:], axis=-1).max()

    def normalize(self, x):
        return x / self.speed_max

    def unnormalize(self, x):
        return x * self.speed_max
