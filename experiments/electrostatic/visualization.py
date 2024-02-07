import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

COLORS = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
COLORS = COLORS + 20 * ['black']

PRED_COLORS = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']
MARKERS = ['o', 'p', '^', 's', 'X']
MARKERS = MARKERS + 20 * ['o']


def plot_trajectories(
    axes, positions, num_objects, num_timesteps, y_pred=None,
    noise_std=0.0, observed_objects=None
):
    x = positions[:num_timesteps, :num_objects]
    y = positions[num_timesteps:, :num_objects]
    ndim = x.shape[-1]

    # Add noise to trajectories
    if noise_std > 0.0:
        assert observed_objects is not None
        vel = np.diff(positions, axis=0)[-y.shape[0]:, :observed_objects]
        vel = vel + noise_std * np.random.randn(*vel.shape)
        y[:, :observed_objects] = x[[-1], :observed_objects] + np.cumsum(vel, axis=0)
    plot_trajectories_and_predictions(axes, x, y, y_pred=y_pred, ndim=ndim)


def plot_trajectories_and_predictions(ax, x, y, y_pred=None, ndim=2, line_only=False):
    num_obj = x.shape[1]
    num_gt_steps = x.shape[0]
    num_pred_steps = y.shape[0]
    full_traj = np.concatenate([x, y], axis=0)
    gt_positions = full_traj[..., :ndim]
    num_steps = num_gt_steps + num_pred_steps
    full_y_pred = np.concatenate([x[[-1]], y_pred], axis=0) if y_pred is not None else None
    pred_positions = full_y_pred[..., :ndim] if full_y_pred is not None else None

    marker_range = np.linspace(1.5, 5.0, num_steps)
    alpha_range = np.linspace(0.1, 1.0, num_steps) ** 2

    for i in range(num_obj):
        for t in range(1, num_steps):
            ax.plot(*(gt_positions[[t-1, t], i, :ndim]).T, '-', color=COLORS[i],
                    alpha=alpha_range[t])

        if line_only:
            ax.plot(*(gt_positions[num_steps-1, i, :ndim]), MARKERS[i], color=COLORS[i],
                    alpha=1.0, markersize=marker_range[-1],
                    markeredgecolor='black')
        else:
            for t in range(1, num_steps):
                ax.plot(*(gt_positions[t, i, :ndim]), MARKERS[i], color=COLORS[i],
                        alpha=alpha_range[t] if t != (num_gt_steps-1) else 1.0,
                        markersize=marker_range[t],
                        markeredgecolor='black' if t == (num_gt_steps-1) else None)

        if y_pred is not None:
            for t in range(1, num_pred_steps+1):
                ax.plot(*(pred_positions[[t-1, t], i, :ndim]).T,
                        '-', color=PRED_COLORS[i], alpha=0.5)
            if line_only:
                ax.plot(*(pred_positions[t, i, :ndim]), MARKERS[i],
                        color=PRED_COLORS[i], alpha=0.5,
                        markersize=marker_range[-1])
            else:
                for t in range(1, num_pred_steps+1):
                    ax.plot(*(pred_positions[t, i, :ndim]), MARKERS[i],
                            color=PRED_COLORS[i], alpha=0.5,
                            markersize=marker_range[t+num_gt_steps-1])


def l2(A, B):
    """
    Input: A is a Nxd matrix
        B is a Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm
        between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:] - B[j,:]||^2

    Can be used with numpy arrays and Pytorch tensors

    For numpy arrays, see also scipy.spatial.distance.cdist

    ```python
    dist = cdist(A, B, 'sqeuclidean')
    ```
    """
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A @ B.T
    return dist


def get_field(positions, charges, test_positions, interaction_strength=1.0):
    """
    positions: N x D
    charges: N x 1
    test_positions: M x D

    Charges can also represent masses
    """
    # Test particles are assumed to carry positive charges
    test_charges = np.ones_like(test_positions[:, [0]])

    l2_norm_cubed = l2(test_positions, positions) ** 1.5

    edges = test_charges @ charges.T
    forces_size = interaction_strength * edges / l2_norm_cubed
    np.fill_diagonal(forces_size, 0)

    directions = test_positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    forces = directions * forces_size[..., np.newaxis]
    field = forces.sum(1)
    return field


def clip_field(field, max_force=10.0):
    field_norm = np.linalg.norm(field, axis=-1, keepdims=True)
    maxed_out_mask = field_norm.squeeze() > max_force
    field[maxed_out_mask] = (
        max_force * field[maxed_out_mask]
        / field_norm[maxed_out_mask])
    # Keep it to handle infinities
    field[field > max_force] = max_force
    field[field < -max_force] = -max_force
    if np.any(np.isinf(field)):
        print('infinities')
    return field


def plot_sampled_field(ax, positions, charges, box_size, ndim=2,
                       interaction_strength=1.0, num_samples=1000):
    test_positions = np.random.uniform(-box_size, box_size, (num_samples, ndim))
    field = get_field(
        positions[-1], charges, test_positions, interaction_strength=interaction_strength)

    if ndim == 2:
        ax.quiver(test_positions[:, 0], test_positions[:, 1], field[..., 0],
                  field[..., 1], color='cornflowerblue')
    else:
        ax.quiver(test_positions[:, 0], test_positions[:, 1], test_positions[:, 2],
                  field[..., 0], field[..., 1], field[..., 2], color='cornflowerblue')


def grid_field(positions, charges, linspaces, ndim, interaction_strength=1.0):
    space_grid = np.stack(np.meshgrid(*linspaces, indexing='ij'), axis=-1)
    test_positions = np.reshape(space_grid, (-1, ndim))
    field = get_field(positions, charges, test_positions,
                      interaction_strength=interaction_strength)
    return field, space_grid


def plot_field(
    ax, positions, charges, box_size, grid_size=101, ndim=2, interaction_strength=1.0,
    clip=0.0, add_noise=0.0, subsample=1,
):
    # Create meshgrid of "test particles, positively charged"
    if isinstance(box_size, (int, float)):
        linspaces = [np.linspace(-box_size, box_size, grid_size)
                     for _ in range(ndim)]
    else:
        linspaces = [np.linspace(box_size[i, 0], box_size[i, 1], grid_size)
                     for i in range(ndim)]
    linspaces = [linspaces[i][::subsample] for i in range(ndim)]

    field, space_grid = grid_field(
        positions, charges, linspaces, ndim, interaction_strength=interaction_strength)
    field = field.reshape(*[ls.shape[0] for ls in linspaces], ndim)
    field = clip_field(field, max_force=clip) if clip > 0.0 else field
    # Add noise for visualizations
    if add_noise > 0.0:
        field += add_noise * np.random.randn(*field.shape)

    field_norm = np.linalg.norm(field, axis=-1)
    field_norm = field_norm / np.max(field_norm)

    if ndim == 2:
        stream = ax.streamplot(
            space_grid[..., 0].T, space_grid[..., 1].T, field[..., 0].T,
            field[..., 1].T, color=field_norm.T, cmap='plasma', density=1.5)
        stream.lines.set_alpha(0.3)
        stream.arrows.set_alpha(0.3)  # Not working
        ax.quiver(
            space_grid[..., 0], space_grid[..., 1], field[..., 0], field[..., 1], field_norm, units='dots', cmap='plasma')
    else:
        q = ax.quiver(
            space_grid[..., 0], space_grid[..., 1], space_grid[..., 2],
            field[..., 0], field[..., 1], field[..., 2], length=0.2, normalize=True,
            cmap='plasma')
        q.set_array(field_norm.flatten())


def extend_range(min_range, max_range, multiplier=0.1):
    full_range = max_range - min_range
    min_range -= multiplier * full_range
    max_range += multiplier * full_range
    return min_range, max_range


def reset_ticks(ax, ndim):
    ax.set_xticks([])
    ax.set_yticks([])
    if ndim == 3:
        ax.set_zticks([])


def set_ax_limits(ax, min_range, max_range, ndim):
    ax.set_xlim(min_range[0], max_range[0])
    ax.set_ylim(min_range[1], max_range[1])
    if ndim == 3:
        ax.set_zlim(min_range[2], max_range[2])


def plot_range(ax, positions, multiplier=0.1):
    ndim = positions.shape[-1]
    max_range = positions.reshape(-1, ndim).max(0)
    min_range = positions.reshape(-1, ndim).min(0)
    min_range, max_range = extend_range(min_range, max_range, multiplier=multiplier)
    set_ax_limits(ax, min_range, max_range, ndim)
    reset_ticks(ax, ndim)
    return min_range, max_range


def plot_charges(axes, positions, charges, num_objects):
    def pick_markersize(x):
        return max(min(abs(x), 16), 8)

    def pick_marker(x):
        return charge_symbols[int(np.sign(x))]

    def pick_text(x):
        return charge_text[int(np.sign(x))]

    def pick_color(x):
        return charged_colors[int(np.sign(x))]

    charged_colors = {1: 'b', 0: 'g', -1: 'r'}
    charge_text = {1: '+', 0: '0', -1: '-'}
    charge_symbols = {1: 'o', 0: 'o', -1: 'o'}

    for i in range(num_objects, positions.shape[-2]):
        axes.plot(
            positions[-1, i, 0], positions[-1, i, 1],
            marker=pick_marker(charges[i, 0]),
            markersize=pick_markersize(charges[i, 0]),
            color=pick_color(charges[i, 0]))
        axes.annotate(
            pick_text(charges[i, 0]), xy=(positions[-1, i, 0], positions[-1, i, 1]),
            xytext=(positions[-1, i, 0], positions[-1, i, 1]), color='lightgray',
            ha="center", va="center")


def plot_video_trajectories(
    loc, num_objects, num_timesteps, field_charges, y_pred=None, ndim=2,
    grid_size=101, interaction_strength=1.0, clip=0.0
):
    plotter = VideoPlotter(
        loc[:num_timesteps, :num_objects],
        loc[num_timesteps:, :num_objects],
        y_pred=y_pred,
        field_positions=loc[:, num_objects:],
        field_charges=field_charges,
        grid_size=grid_size,
        interaction_strength=interaction_strength,
        clip=clip,
        ndim=ndim,
    )
    plotter.new_fig()
    anim = plotter.plot_video()
    return anim, plotter


class VideoPlotter(object):
    colors = ['firebrick', 'forestgreen', 'dodgerblue', 'mediumvioletred', 'darkturquoise']
    pred_colors = ['lightsalmon', 'lightgreen', 'lightskyblue', 'palevioletred', 'lightskyblue']
    markers = ['o', 'p', '^', 's', 'X']

    """Docstring for VideoPlotter. """
    def __init__(self, x, y, y_pred=None, ndim=2, grid_size=101, field_positions=None,
                 field_charges=None, interaction_strength=1.0, clip=10.0):
        self.x = x
        self.y = y
        self.y_pred = y_pred
        self.ndim = ndim
        self.grid_size = grid_size

        self.full_traj = np.concatenate([x, y], axis=0)

        self.num_obj = self.x.shape[1]
        self.num_gt_steps = self.x.shape[0]
        self.num_pred_steps = self.y.shape[0]
        self.num_steps = self.num_gt_steps + self.num_pred_steps

        self.max_range = self.full_traj[:, :, :ndim].reshape(-1, self.ndim).max(0)
        self.min_range = self.full_traj[:, :, :ndim].reshape(-1, self.ndim).min(0)
        full_range = self.max_range - self.min_range
        self._multiplier = 0.0
        self.max_range += full_range * self._multiplier
        self.min_range -= full_range * self._multiplier

        self.field_positions = field_positions
        self.field_charges = field_charges
        self.plot_box_size = np.stack([self.min_range, self.max_range], axis=1)
        self.interaction_strength = interaction_strength
        self.clip = clip

    def new_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d') if self.ndim == 3 else self.fig.add_subplot(111)
        if self.ndim == 2:
            self.ax.set_aspect('equal')

    def plot_video(self):
        def update(frame):
            marker_range = 2.5 + 5.5 * np.arange(0, frame+1) / (frame+1)
            alpha_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2
            pred_range = (0.1 + 0.9 * np.arange(0, frame+1) / (frame+1)) ** 2 / 2
            self.ax.clear()
            for obj in range(self.num_obj):
                for t in range(1, frame+1):
                    self.ax.plot(
                        *(self.full_traj[[t-1, t], obj, :self.ndim].T), '-',
                        color=self.colors[obj], alpha=alpha_range[t])
                    if t == frame:
                        self.ax.plot(
                            *(self.full_traj[t, obj, :self.ndim]), self.markers[obj],
                            color=self.colors[obj], alpha=1.0,
                            markersize=marker_range[t],
                            markeredgecolor='black')
                    else:
                        self.ax.plot(
                            *(self.full_traj[t, obj, :self.ndim]), self.markers[obj],
                            color=self.colors[obj], markersize=marker_range[t],
                            alpha=alpha_range[t])
            if self.y_pred is not None and frame >= self.num_gt_steps:
                for obj in range(self.num_obj):
                    for t in range(self.num_gt_steps, frame+1):
                        tmp_fr = t - self.num_gt_steps
                        if t == frame:
                            self.ax.plot(
                                *(self.y_pred[tmp_fr, obj, :self.ndim]),
                                self.markers[obj], color=self.pred_colors[obj], alpha=1.0,
                                markersize=marker_range[t],
                                markeredgecolor='black')
                        else:
                            self.ax.plot(
                                *(self.y_pred[tmp_fr, obj, :self.ndim]),
                                self.markers[obj], color=self.pred_colors[obj],
                                markersize=marker_range[t], alpha=pred_range[t])

                        if t > self.num_gt_steps:
                            self.ax.plot(
                                *(self.y_pred[[tmp_fr-1, tmp_fr], obj, :self.ndim].T),
                                '-', color=self.pred_colors[obj], alpha=alpha_range[t])
                        else:
                            first_pred = np.concatenate([self.x[[t-1]], self.y_pred[[tmp_fr]]], 1)
                            self.ax.plot(
                                *(first_pred[:, obj, :self.ndim].T), '-',
                                color=self.pred_colors[obj], alpha=alpha_range[t])
            self.axes_lim(frame)

            self.ax.scatter(*self.field_positions[frame, 0])

            plot_field(
                self.ax, self.field_positions[frame], self.field_charges,
                self.plot_box_size, grid_size=self.grid_size, ndim=self.ndim,
                interaction_strength=self.interaction_strength,
                clip=self.clip,
            )

        ani = animation.FuncAnimation(self.fig, update, interval=100,
                                      frames=self.num_steps, blit=False)
        return ani

    def axes_lim(self, frame):
        set_ax_limits(self.ax, self.min_range, self.max_range, self.ndim)
        reset_ticks(self.ax, self.ndim)
