import os

import numpy as np
import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


from experiments.electrostatic.visualization import (
    plot_trajectories,
    plot_field,
    plot_range,
    plot_video_trajectories,
)


if __name__ == '__main__':
    num_balls = 5
    box_size = 5.0
    ndim = 2
    interaction_strength = 1.0
    input_steps = 29

    model_name = 'aether'
    data_dir = 'dataset/data/electrostatic_field'
    result_dir = 'results/nn.seq2seq.aether.Aether/seed_1'

    test_feats = torch.load(os.path.join(data_dir, 'test_feats'))
    static_field = torch.load(os.path.join(data_dir, 'static_field')).numpy()
    test_charges = torch.load(os.path.join(data_dir, 'test_charges'))
    static_charges = torch.load(os.path.join(data_dir, 'static_charges')).numpy()
    test_preds = np.load(os.path.join(result_dir, 'all_outputs_29_20.npy'))
    test_preds = torch.from_numpy(test_preds).squeeze(1)

    for example_idx in [0]:
        loc = test_feats[example_idx, ..., :ndim].numpy()
        vel = test_feats[example_idx, ..., ndim:].numpy()
        charges = test_charges[example_idx][..., None].numpy()

        loc = np.concatenate(
            [
                loc,
                np.broadcast_to(
                    static_field[..., :ndim],
                    (loc.shape[0], static_field.shape[1], ndim)
                )
            ],
            -2
        )

        fig = plt.figure()
        axes = plt.gca()
        if ndim == 2:
            axes.set_aspect('equal')
        min_range, max_range = plot_range(axes, loc)

        plot_trajectories(
            axes, loc, num_balls, input_steps, y_pred=test_preds[example_idx, ..., :ndim]
        )
        plot_field(
            axes, static_field[0, :, :ndim], static_charges,
            np.stack([min_range, max_range], axis=1), ndim=ndim, grid_size=101,
            interaction_strength=interaction_strength, clip=10.0
        )
        # plt.show()
        fig.savefig(
            os.path.join(result_dir, f'electrostatic_{model_name}_{example_idx}.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)

        ani, plotter = plot_video_trajectories(
            loc, num_balls, input_steps, static_charges,
            y_pred=test_preds[example_idx, ..., :ndim],
            grid_size=101, interaction_strength=interaction_strength, clip=10.0
        )
        path = os.path.join(result_dir, f'pred_trajectory_{example_idx}.mp4')
        ani.save(path, dpi=300, codec='mpeg4', bitrate=8000)
        plt.close(plotter.fig)
