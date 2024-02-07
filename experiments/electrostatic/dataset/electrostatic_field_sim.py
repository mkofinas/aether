import numpy as np
from scipy.spatial.distance import cdist


class ElectrostaticFieldSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0., dim=2, static_balls=0,
                 static_charge_strength=1.0):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self.dim = dim

        self.static_balls = static_balls

        self._charge_types = np.array([-1., 0., 1.])
        self._static_charge_strength = static_charge_strength
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        self._particle_seed = 0
        self.reset_particle_rng()
        self._field_seed = 1
        self.reset_field_rng()

        self.sampler = self.sample_location_inside_box

    def reset_particle_rng(self):
        self.particle_rng = np.random.default_rng(self._particle_seed)

    def reset_field_rng(self):
        self.field_rng = np.random.default_rng(self._field_seed)

    @staticmethod
    def _l2(A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:] - B[j,:]||^2
        """
        dist = cdist(A, B, 'sqeuclidean')
        return dist

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            K = 0.5 * (vel ** 2).sum()
            non_diag_entries = np.where(~np.eye(loc.shape[1], dtype=bool))
            U = 0.5 * self.interaction_strength * edges / cdist(loc, loc)
            U = U[non_diag_entries].sum()
            return U + K, U, K

    def sample_location_inside_box(self):
        '''Select points inside or on a 2D rectangle/3D cube'''
        return self.field_rng.uniform(-self.box_size, self.box_size, (self.static_balls, self.dim))

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[0.5, 0.0, 0.5], field_charge_prob=None):
        n = self.n_balls
        total_balls = self.n_balls + self.static_balls

        # T_save is number of (saved) measurements/observations
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        counter = 0  # count number of measurements

        # create matrix of 1s with 0s on diag
        diag_mask = np.ones((total_balls, total_balls), dtype=bool)
        np.fill_diagonal(diag_mask, 0)

        # Sample charges and get edges
        if self.static_balls > 0:
            field_charge_prob = (charge_prob if field_charge_prob is None
                                 else field_charge_prob)
            charges = np.concatenate([
                self.particle_rng.choice(
                    self._charge_types, size=(self.n_balls, 1),
                    p=charge_prob),
                self.field_rng.choice(
                    self._charge_types, size=(self.static_balls, 1),
                    p=field_charge_prob) * self._static_charge_strength
            ])
        else:
            charges = self.particle_rng.choice(
                self._charge_types, size=(self.n_balls, 1), p=charge_prob)
        edges = charges @ charges.T

        # Initialize location and velocity
        loc = np.zeros((T_save, total_balls, self.dim))
        vel = np.zeros((T_save, total_balls, self.dim))

        loc_next = np.concatenate(
            [self.particle_rng.normal(size=(n, self.dim)) * self.loc_std,
             self.sampler()], 0)
        vel_next = np.concatenate(
            [self.particle_rng.normal(size=(n, self.dim)),
             np.zeros((self.static_balls, self.dim))], 0)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=1, keepdims=True))
        vel_next = vel_next * self.vel_norm / v_norm
        vel_next[n:, :] = 0.0
        loc[0, :, :], vel[0, :, :] = loc_next, vel_next
        loc[:, n:, :] = loc[[0], n:, :]

        # count number of times forces were capped
        count_maxedout = 0

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(self._l2(loc_next, loc_next), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(total_balls, total_balls, 1)
                 * (loc_next[:, None, :] - loc_next[None, :, :]))
            F = F.sum(axis=1)

            # cap maximum force strength
            force_norm = np.linalg.norm(F, axis=-1, keepdims=True)
            maxed_out_mask = (force_norm > self._max_F).squeeze()
            F[maxed_out_mask] = (
                self._max_F * F[maxed_out_mask]
                / force_norm[maxed_out_mask]
            )
            count_maxedout += np.sum(maxed_out_mask)

            vel_next[:n, :] += self._delta_T * F[:n, :]
            # run leapfrog
            for i in range(1, T):
                loc_next[:n, :] += self._delta_T * vel_next[:n, :]

                if i % sample_freq == 0:
                    loc[counter, :n, :] = loc_next[:n, :]
                    vel[counter, :n, :] = vel_next[:n, :]
                    counter += 1

                l2_dist_power3 = np.power(self._l2(loc_next, loc_next), 3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                F = (forces_size.reshape(total_balls, total_balls, 1)
                     * (loc_next[:, None, :] - loc_next[None, :, :]))
                F = F.sum(axis=1)

                # cap maximum force strength
                force_norm = np.linalg.norm(F, axis=-1, keepdims=True)
                maxed_out_mask = (force_norm > self._max_F).squeeze()
                F[maxed_out_mask] = (
                    self._max_F * F[maxed_out_mask]
                    / force_norm[maxed_out_mask]
                )

                count_maxedout += np.sum(maxed_out_mask)

                vel_next[:n, :] += self._delta_T * F[:n, :]
            # Add noise to observations
            loc[:, :n] += self.particle_rng.normal(size=(T_save, n, self.dim)) * self.noise_var
            vel[:, :n] += self.particle_rng.normal(size=(T_save, n, self.dim)) * self.noise_var
            print(count_maxedout)

            return loc, vel, edges, charges
