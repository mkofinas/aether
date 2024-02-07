import os

import numpy as np
import torch
from torch.utils.data import Dataset


class StaticFieldData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
            charge_path = os.path.join(data_path, 'train_charges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'valid_feats')
            edge_path = os.path.join(data_path, 'valid_edges')
            charge_path = os.path.join(data_path, 'valid_charges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')
            charge_path = os.path.join(data_path, 'test_charges')
        else:
            raise NotImplementedError

        self.feats = torch.load(path)
        self.ndim = params.get('ndim', 2)
        self.edges = torch.load(edge_path)
        self.charges = torch.load(charge_path)
        self.static_field = torch.load(os.path.join(data_path, 'static_field'))
        self.static_charges = torch.load(os.path.join(data_path, 'static_charges'))
        self.same_norm = params['same_data_norm']
        self.symmetric_norm = params['symmetric_data_norm']
        self.no_norm = params['no_data_norm']
        self.vel_norm_norm = params['vel_norm_norm']
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min) * 2 / (self.feat_max - self.feat_min) - 1
        elif self.vel_norm_norm:
            self.vel_norm_max = np.linalg.norm(train_data[..., self.ndim:], axis=-1).max()
            self.feats[..., :self.ndim] = self.feats[..., :self.ndim] / self.vel_norm_max
            self.feats[..., self.ndim:] = self.feats[..., self.ndim:] / self.vel_norm_max
        else:
            if self.symmetric_norm:
                self.loc_max = train_data[:, :, :, :self.ndim].abs().max()
                self.loc_min = -self.loc_max
                self.vel_max = train_data[:, :, :, self.ndim:].abs().max()
                self.vel_min = -self.vel_max
            else:
                self.loc_max = train_data[:, :, :, :self.ndim].max()
                self.loc_min = train_data[:, :, :, :self.ndim].min()
                self.vel_max = train_data[:, :, :, self.ndim:].max()
                self.vel_min = train_data[:, :, :, self.ndim:].min()
            self.feats[:, :, :, :self.ndim] = (self.feats[:, :, :, :self.ndim] - self.loc_min) * 2 / (self.loc_max - self.loc_min) - 1
            self.feats[:, :, :, self.ndim:] = (self.feats[:, :, :, self.ndim:] - self.vel_min) * 2 / (self.vel_max - self.vel_min) - 1

    def unnormalize(self, data):
        if self.no_norm:
            return data.numpy()
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        elif self.vel_norm_norm:
            result1 = data[..., :self.ndim] * self.vel_norm_max
            result2 = data[..., self.ndim:] * self.vel_norm_max
            return np.concatenate([result1, result2], axis=-1)
        else:
            result1 = (data[:, :, :, :self.ndim] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            result2 = (data[:, :, :, self.ndim:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
            return np.concatenate([result1, result2], axis=-1)

    def torch_unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        elif self.vel_norm_norm:
            result1 = data[..., :self.ndim] * self.vel_norm_max
            result2 = data[..., self.ndim:] * self.vel_norm_max
            return torch.cat([result1, result2], axis=-1)
        else:
            result1 = (data[:, :, :, :self.ndim] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            result2 = (data[:, :, :, self.ndim:] + 1) * (self.vel_max - self.vel_min) / 2. + self.vel_min
            return torch.cat([result1, result2], axis=-1)

    def __getitem__(self, idx):
        return {
            'inputs': self.feats[idx],
            'edges': self.edges[idx],
            'charges': self.charges[idx],
        }

    def __len__(self):
        return len(self.feats)
