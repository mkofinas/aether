import os

import numpy as np
import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.electrostatic.visualization import (
    plot_trajectories, plot_video_trajectories, plot_range, plot_field
)


if __name__ == '__main__':
    num_balls = 5
    box_size = 3.0
    ndim = 3
    interaction_strength = 1.0
    input_steps = 44
    grid_size = 16

    data_dir = 'dataset/data/gravitational_field_3d'

    model_name = 'aether'
    result_dir = 'results/gravitational_field_3d/nn.seq2seq.dynamic_field_aether.DynamicFieldAether/seed_1'

    test_feats = torch.load(os.path.join(data_dir, 'test_feats'))
    test_field = torch.load(os.path.join(data_dir, 'test_field'))
    # NOTE: Hardcoded source masses to 10.0
    test_masses = 10.0 * torch.ones_like(test_field[..., 0])
    test_preds = np.load(os.path.join(result_dir, 'all_outputs_44_5.npy'))
    test_preds = torch.from_numpy(test_preds).squeeze(1)

    for example_idx in [0, 1, 2, 3]:
        loc = test_feats[example_idx, ..., :ndim].numpy()
        vel = test_feats[example_idx, ..., ndim:].numpy()
        field_positions = test_field[example_idx].numpy()
        masses = test_masses[example_idx][..., None].numpy()

        loc = np.concatenate(
            [
                loc,
                np.broadcast_to(
                    field_positions[..., :ndim],
                    (loc.shape[0], field_positions.shape[0], ndim)
                )
            ],
            -2
        )

        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d') if ndim == 3 else fig.add_subplot(111)
        if ndim == 2:
            axes.set_aspect('equal')
        min_range, max_range = plot_range(axes, loc)

        plot_trajectories(
            axes, loc, num_balls, input_steps, y_pred=test_preds[example_idx, ..., :ndim]
        )
        plot_field(
            axes, field_positions[..., :ndim], masses,
            np.stack([min_range, max_range], axis=1), ndim=ndim, grid_size=grid_size,
            interaction_strength=interaction_strength, clip=10.0
        )
        # plt.show()
        fig.savefig(
            os.path.join(result_dir, f'pred_trajectories_gravitational_{ndim}d_{model_name}_test_{example_idx}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

        ani, plotter = plot_video_trajectories(
            loc, num_balls, input_steps, masses,
            y_pred=test_preds[example_idx, ..., :ndim], ndim=ndim,
            grid_size=grid_size, interaction_strength=interaction_strength, clip=10.0
        )
        path = os.path.join(result_dir, f'pred_trajectory_{model_name}_{example_idx}.mp4')
        ani.save(path, dpi=300, codec='mpeg4', bitrate=8000)
        plt.close(plotter.fig)
