import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments.gravitational.dataset.gravitational_field_sim import GravitationalFieldSim
from experiments.electrostatic.visualization import plot_trajectories, plot_field, plot_range, plot_sampled_field, VideoPlotter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', action='store_true',
                        help='If true, plot saved data')
    parser.add_argument('--dynamic', action='store_true',
                        help='If true, use different fields fror each simulation')
    parser.add_argument('--video', action='store_true',
                        help='If true, visualize simulation as video')
    parser.add_argument('--name', type=str,
                        help='Name of saved data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Name of saved data')
    parser.add_argument('--idx', type=int, default=0,
                        help='Index of saved data to plot')
    parser.add_argument('--num-balls', type=int, default=5,
                        help='Number of balls in the simulation.')
    parser.add_argument('--static-balls', type=int, default=20,
                        help='Number of static balls in the simulation.')
    parser.add_argument('--num-timesteps', type=int, default=5000,
                        help='Length of trajectory.')
    parser.add_argument('--sample-freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--box-size', type=float, default=5.0,
                        help='Size of a surrounding box. If 0, then no box.')
    parser.add_argument('--dt', type=float, default=1e-3,
                        help='Time step of the simulation.')
    parser.add_argument('--ndim', type=int, default=2, choices=[2, 3],
                        help='Spatial simulation dimension (2 or 3).')
    parser.add_argument('--strength', type=float,
                        default=1.0, help='Strength of field particle charges')
    args = parser.parse_args()
    print(args)

    interaction_strength = 1.0
    ndim = args.ndim
    grid_size = 8

    if args.saved:
        data_dir = args.name
        data = torch.load(os.path.join(data_dir, f'{args.split}_feats'))

        example_idx = args.idx
        loc = data[example_idx, :, :, :ndim].numpy()
        vel = data[example_idx, :, :, ndim:].numpy()
        if args.dynamic:
            data_field = torch.load(os.path.join(data_dir, f'{args.split}_field'))
            expanded_field_pos = np.repeat(data_field[[example_idx], :, :ndim].numpy(), loc.shape[0], axis=0)
            expanded_field_vel = np.repeat(data_field[[example_idx], :, ndim:].numpy(), loc.shape[0], axis=0)
        else:
            static_field = torch.load(os.path.join(data_dir, 'static_field'))
            expanded_field_pos = np.repeat(static_field.numpy()[..., :ndim], loc.shape[0], axis=0)
            expanded_field_vel = np.repeat(static_field.numpy()[..., ndim:], loc.shape[0], axis=0)
        loc = np.concatenate([loc, expanded_field_pos], -2)
        vel = np.concatenate([vel, expanded_field_vel], -2)
    else:
        sim = GravitationalFieldSim(
            n_balls=args.num_balls, static_balls=args.static_balls,
            static_mass=args.strength, dim=args.ndim, dt=args.dt)

        loc, vel, forces, masses = sim.sample_trajectory(
            T=args.num_timesteps, sample_freq=args.sample_freq)

    field_charges = np.full_like(loc[0, args.num_balls:, 0][..., None], -args.strength)
    if args.video:
        plotter = VideoPlotter(
            loc[:39, :args.num_balls, :ndim],
            loc[39:, :args.num_balls, :ndim],
            field_positions=loc[:, args.num_balls:, :],
            field_charges=field_charges,
            ndim=args.ndim,
            grid_size=grid_size,
            interaction_strength=interaction_strength
        )
        plotter.new_fig()
        ani = plotter.plot_video()

        path = os.path.join('.', f'gravitational_field_{args.split}_{args.idx}.mp4')
        ani.save(path, codec='mpeg4')
        plt.close(plotter.fig)
    else:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d') if ndim == 3 else fig.add_subplot(111)
        if ndim == 2:
            axes.set_aspect('equal')
        min_range, max_range = plot_range(axes, loc)
        plot_trajectories(axes, loc, args.num_balls+args.static_balls, num_timesteps=50)
        plot_range(axes, loc)

        strm = plot_field(
            axes, loc[0, args.num_balls:], field_charges,
            np.stack([min_range, max_range], axis=1), ndim=args.ndim,
            grid_size=grid_size, interaction_strength=interaction_strength, clip=10.0)
        plt.show()
        # fig.savefig(f'gravitational_field_{args.split}_{args.idx}.png', dpi=300, bbox_inches='tight')
