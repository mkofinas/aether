import os

import torch

from experiments.utils.normalization.abstract_normalization import AbstractNormalization


class SameNormNormalization(AbstractNormalization):
    def __init__(self, data_path, params):
        train_data = torch.load(os.path.join(data_path, 'train_feats'))

        self.feat_max = train_data.max()
        self.feat_min = train_data.min()
        self.feat_range = self.feat_max - self.feat_min

    def normalize(self, x):
        return (x - self.feat_min) * 2.0 / self.feat_range - 1.0

    def unnormalize(self, x):
        return (x + 1.0) * self.feat_range / 2.0 + self.feat_min
