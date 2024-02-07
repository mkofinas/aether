import os
from functools import partial

import torch
from torch.utils.data import Dataset

from experiments.utils.normalization.normalization_factory import NormalizationFactory


class GravityDynamicFieldData(Dataset):
    def __init__(self, data_path, mode, params):
        if mode not in ('train', 'val', 'test'):
            raise ValueError('mode must be one of (train, val, test)')
        self.mode = mode if mode != 'val' else 'valid'
        self.ndim = params.get('ndim', 2)
        self.data_path = data_path

        self.feats = torch.load(os.path.join(data_path, f'{self.mode}_feats'))
        self.field_feats = torch.load(os.path.join(data_path, f'{self.mode}_field'))

        self.normalization = params['normalization']
        self.normalizer = NormalizationFactory.create(
            self.normalization, data_path, params)
        self._normalize_data()

        self.field = GravitationalField(
            self.field_feats, normalizer=self.normalizer,
            delta_t=params['delta_t'], ndim=self.ndim)

    def _normalize_data(self):
        self.feats = self.normalizer.normalize(self.feats)

    def unnormalize(self, x):
        return self.normalizer.unnormalize(x)

    def torch_unnormalize(self, x):
        return self.normalizer.torch_unnormalize(x)

    def __getitem__(self, idx):
        return {
            'inputs': self.feats[idx],
            'field': partial(self.field.__call__, idx),
            'oracle': self.field.norm_positions[idx],
        }

    def __len__(self):
        return len(self.feats)


class GravitationalField(object):
    def __init__(self, positions, interaction_strength=1.0,
                 normalizer=None, delta_t=1e-3, ndim=2):
        """
        positions: 1 x N x 2 or B x N x 2
        """
        self.ndim = ndim
        self.positions = positions[:, :, :ndim].contiguous().cuda()
        self.interaction_strength = interaction_strength
        self._normalizer = normalizer
        self.norm_positions = self._normalize(self.positions.cpu())

        # TODO: Un-hardcode
        self.masses = 1e1 * torch.ones_like(self.positions[..., [0]])

        self._dt = delta_t
        self.sampling = 1e-2
        self._max_force = 0.1 / self._dt
        self._eps = 1e-6

    def _normalize(self, data):
        return self._normalizer.normalize_velocities(data)

    def _unnormalize(self, data):
        return self._normalizer.unnormalize_positions(data)

    def _clip_field(self, field):
        force_norm = torch.norm(field, dim=-1)
        maxed_out_mask = force_norm > self._max_force
        field[maxed_out_mask] = (
            self._max_force * field[maxed_out_mask]
            / force_norm[maxed_out_mask].unsqueeze(-1))
        # Keep it to handle infinities
        field[field > self._max_force] = self._max_force
        field[field < -self._max_force] = -self._max_force
        if torch.any(torch.isinf(field)):
            print('infinities')
        return field

    @torch.no_grad()
    def __call__(self, indices, test_positions, normalize_force=False):
        if isinstance(indices, int):
            indices = [indices]
        inputs = self._unnormalize(test_positions)
        electric_field = self.get_field(indices, inputs)
        electric_field = self._clip_field(electric_field)
        electric_field = self.sampling * electric_field
        outputs = (self._normalize(electric_field) if normalize_force
                   else electric_field)
        return outputs

    def get_field(self, indices, test_positions):
        """
        indices: int or list of ints
        test_positions: B x M x 2
        """
        if isinstance(indices, int):
            indices = [indices]

        # Test particles are assumed to carry unit mass, shape: M x 1
        test_masses = torch.ones_like(test_positions[..., [0]])
        test_masses = (test_masses[None, ...] if test_positions.ndim == 2
                       else test_masses)
        edges = test_masses @ self.masses[indices].transpose(-2, -1)

        l2_norm_cubed = torch.cdist(test_positions, self.positions[indices]) ** 3.0
        force_mag = self.interaction_strength * edges / (l2_norm_cubed + self._eps)

        directions = (
            test_positions[None, :, None, :] - self.positions[indices, None, :, :]
            if test_positions.ndim == 2
            else test_positions[:, :, None, :] - self.positions[indices, None, :, :]
        )
        # directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        forces = directions * force_mag[..., None]
        electric_field = forces.sum(-2)
        return electric_field

    @staticmethod
    def _make_grid(box_size=5.0, grid_size=21, ndim=2):
        # Create meshgrid of "test particles, positively charged"
        linspaces = [torch.linspace(-box_size, box_size, grid_size)
                     for _ in range(ndim)]
        test_positions = torch.reshape(torch.stack(torch.meshgrid(*linspaces)),
                                       (ndim, -1)).flip(0).T
        return test_positions

    @torch.no_grad()
    def grid_field(self, indices, box_size=5.0, grid_size=21, ndim=2,
                   normalize=False):
        if isinstance(indices, int):
            indices = [indices]
        test_positions = self._make_grid(box_size, grid_size, ndim).cuda()
        field = self.get_field(indices, test_positions)
        field = self._clip_field(field)
        field = self.sampling * field
        if normalize:
            field = self._normalize(field)
        return field
