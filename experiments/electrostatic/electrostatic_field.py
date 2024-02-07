import torch


class ElectrostaticField(object):
    def __init__(self, positions, charges, interaction_strength=1.0,
                 dataset=None, delta_t=1e-1, ndim=2, device='cuda'):
        """
        positions: 1 x N x 2
        charges: N x 1
        """
        self.ndim = ndim
        self.device = device
        self.positions = positions[:, :, :ndim].squeeze().contiguous().to(device)
        self.charges = charges.to(device)
        self.interaction_strength = interaction_strength
        self._dataset = dataset

        self._dt = delta_t
        self._max_force = 0.1 / self._dt

    def _normalize(self, data):
        if self._dataset.no_norm:
            return data
        elif self._dataset.same_norm:
            feat_max = self._dataset.feat_max
            feat_min = self._dataset.feat_min
            return (data - feat_min) * 2 / (feat_max - feat_min) - 1
        elif self._dataset.vel_norm_norm:
            vel_norm_max = self._dataset.vel_norm_max
            return data / vel_norm_max
        else:
            vel_max = self._dataset.vel_max
            vel_min = self._dataset.vel_min
            return (data - vel_min) * 2 / (vel_max - vel_min) - 1

    def _unnormalize(self, data):
        if self._dataset.no_norm:
            return data
        elif self._dataset.same_norm:
            feat_max = self._dataset.feat_max
            feat_min = self._dataset.feat_min
            return (data + 1) * (feat_max - feat_min) / 2.0 + feat_min
        elif self._dataset.vel_norm_norm:
            vel_norm_max = self._dataset.vel_norm_max
            return data * vel_norm_max
        else:
            loc_max = self._dataset.loc_max
            loc_min = self._dataset.loc_min
            return (data + 1) * (loc_max - loc_min) / 2.0 + loc_min

    def _clip_field(self, field):
        force_norm = torch.norm(field, dim=1)
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
    def __call__(self, test_positions, normalize_force=True):
        inputs = self._unnormalize(test_positions)
        electric_field = self.get_field(inputs)
        electric_field = self._clip_field(electric_field)
        outputs = (self._normalize(electric_field) if normalize_force
                   else electric_field)
        return outputs

    def get_field(self, test_positions):
        """
        test_positions: M x 2
        """
        if test_positions.shape[0] == 2:
            print('Transpose it! Deprecated')
            test_positions = test_positions.T.contiguous()

        # Test particles are assumed to carry positive charges, shape: M x 1
        test_charges = torch.ones_like(test_positions[:, [0]])

        l2_norm_cubed = torch.cdist(test_positions, self.positions) ** 3.0
        edges = test_charges @ self.charges.T
        eps = 1e-6
        force_mag = self.interaction_strength * edges / (l2_norm_cubed + eps)

        directions = test_positions[:, None, :] - self.positions[None, :, :]
        # directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        forces = directions * force_mag[..., None]
        electric_field = forces.sum(1)
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
    def grid_field(self, box_size=5.0, grid_size=21, ndim=2, normalize=False):
        test_positions = self._make_grid(box_size, grid_size, ndim).to(self.device)
        field = self.get_field(test_positions)
        field = self._clip_field(field)
        if normalize:
            field = self._normalize(field)
        return field
