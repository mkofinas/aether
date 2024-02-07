import numpy as np


class GravitationalFieldSim(object):
    def __init__(self, n_balls=100, box_size=1.0, loc_std=1, vel_norm=0.5,
                 interaction_strength=1, noise_var=0, dt=0.001, softening=0.1,
                 dim=3, static_balls=0, static_mass=1.0, **kwargs):
        self.n_balls = n_balls
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        # self.interaction_strength = 6.6743 * 1e-11

        self.noise_var = noise_var
        self.dt = dt
        self.softening = softening
        self.position_variance = 1.0

        self.dim = dim
        self.static_balls = static_balls
        self.static_mass = static_mass

        self._field_seed = 1
        self.reset_field_rng()
        self.box_size = box_size

    def reset_field_rng(self):
        self.field_rng = np.random.default_rng(self._field_seed)

    def sample_location_inside_box(self):
        '''Select points inside or on a 2D rectangle/3D cube'''
        return self.field_rng.uniform(-self.box_size, self.box_size, (self.static_balls, self.dim))

    def compute_acceleration(self, pos, mass, G, softening):
        # matrix that stores all pairwise particle separations: r_j - r_i
        diff = pos[None, :, :] - pos[:, None, :]

        # matrix that stores 1/r^3 for all particle pairwise particle separations
        inv_r3 = (diff ** 2).sum(-1) + softening ** 2
        inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

        a = np.einsum('ijd,je->id', G * (diff * inv_r3[:, :, None]), mass)
        return a

    def _energy(self, pos, vel, mass, G):
        # Kinetic Energy:
        KE = 0.5 * np.sum(np.sum(mass * vel**2))

        # Potential Energy:

        # positions r = [x,y,z] for all particles
        x = pos[:, [0]]
        y = pos[:, [1]]
        if self.dim == 3:
            z = pos[:, [2]]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y
        if self.dim == 3:
            dz = z.T - z

        # matrix that stores 1/r for all particle pairwise particle separations
        if self.dim == 2:
            inv_r = np.sqrt(dx**2 + dy**2)
        else:
            inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
        inv_r[inv_r > 0] = 1.0/inv_r[inv_r > 0]

        # sum over upper triangle, to count each interaction only once
        PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r, 1)))

        return KE, PE, KE+PE

    def sample_trajectory(self, T=10000, sample_freq=10):
        assert (T % sample_freq == 0)

        T_save = int(T/sample_freq)

        N = self.n_balls
        total_balls = self.n_balls + self.static_balls

        pos_save = np.zeros((T_save, total_balls, self.dim))
        vel_save = np.zeros((T_save, total_balls, self.dim))
        force_save = np.zeros((T_save, total_balls, self.dim))

        # Specific sim parameters
        mass = np.concatenate([np.ones((N, 1)),
                               self.static_mass*np.ones((self.static_balls, 1))], 0)
        t = 0
        pos = self.position_variance * np.random.randn(total_balls, self.dim)  # randomly selected positions and velocities
        vel = np.concatenate([np.random.randn(N, self.dim),
                              np.zeros((self.static_balls, self.dim))], 0)

        # Convert to Center-of-Mass frame
        vel -= np.mean(mass * vel, 0) / np.mean(mass)

        # calculate initial gravitational accelerations
        acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

        for i in range(T):
            if i % sample_freq == 0:
                if i == 0:
                    pos_save[0, :, :] = pos
                    vel_save[0, :, :] = 0.0
                    force_save[0, :, :] = 0.0
                else:
                    pos_save[int(i/sample_freq)] = pos
                    vel_save[int(i/sample_freq)] = vel
                    force_save[int(i/sample_freq)] = acc*mass

            # (1/2) kick
            vel[:N] += acc[:N] * self.dt/2.0

            # drift
            pos[:N] += vel[:N] * self.dt

            # update accelerations
            acc = self.compute_acceleration(pos, mass, self.interaction_strength, self.softening)

            # (1/2) kick
            vel[:N] += acc[:N] * self.dt/2.0

            # update time
            t += self.dt

        # Add noise to observations
        pos_save[:, :N] += np.random.randn(T_save, N, self.dim) * self.noise_var
        vel_save[:, :N] += np.random.randn(T_save, N, self.dim) * self.noise_var
        force_save[:, :N] += np.random.randn(T_save, N, self.dim) * self.noise_var
        return pos_save, vel_save, force_save, mass
