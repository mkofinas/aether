import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments.electrostatic.dataset.electrostatic_field_sim import ElectrostaticFieldSim
from experiments.electrostatic.visualization import plot_trajectories, plot_range, plot_charges, plot_field, VideoPlotter


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
    parser.add_argument('--charge-prob', type=float, default=[0.5, 0.0, 0.5],
                        nargs=3, help='Ball charge probabilities')
    parser.add_argument('--field-charge-prob', type=float, default=[0.5, 0.0, 0.5],
                        nargs=3, help='Field Ball charge probabilities')
    parser.add_argument('--num-timesteps', type=int, default=5000,
                        help='Length of trajectory.')
    parser.add_argument('--sample-freq', type=int, default=100,
                        help='How often to sample the trajectory.')
    parser.add_argument('--box-size', type=float, default=5.0,
                        help='Size of a surrounding box. If 0, then no box.')
    parser.add_argument('--ndim', type=int, default=2, choices=[2, 3],
                        help='Spatial simulation dimension (2 or 3).')
    parser.add_argument('--strength', type=float,
                        default=1.0, help='Strength of field particle charges')
    args = parser.parse_args()
    print(args)

    interaction_strength = 1.0
    ndim = args.ndim
    grid_size = 256

    if args.saved:
        data_dir = args.name
        data = torch.load(os.path.join(data_dir, f'{args.split}_feats'))
        charges = torch.load(os.path.join(data_dir, f'{args.split}_charges')).unsqueeze(-1)

        example_idx = args.idx
        loc = data[example_idx, :, :, :ndim].numpy()
        vel = data[example_idx, :, :, ndim:].numpy()
        if args.dynamic:
            data_field = torch.load(os.path.join(data_dir, f'{args.split}_field'))
            expanded_field_pos = data_field[example_idx, :, :, :ndim].numpy()
            expanded_field_vel = data_field[example_idx, :, :, ndim:].numpy()
        else:
            static_field = torch.load(os.path.join(data_dir, 'static_field'))
            expanded_field_pos = np.repeat(static_field.numpy()[..., :ndim], loc.shape[0], axis=0)
            expanded_field_vel = np.repeat(static_field.numpy()[..., ndim:], loc.shape[0], axis=0)
        loc = np.concatenate([loc, expanded_field_pos], 1)
        vel = np.concatenate([vel, expanded_field_vel], 1)
        charges = charges[example_idx].numpy()
    else:
        sim = ElectrostaticFieldSim(
            n_balls=args.num_balls, box_size=args.box_size, static_balls=args.static_balls,
            static_charge_strength=args.strength)

        sim.reset_field_rng()
        loc, vel, edges, charges = sim.sample_trajectory(
            T=args.num_timesteps, sample_freq=args.sample_freq,
            charge_prob=args.charge_prob, field_charge_prob=args.field_charge_prob)

    if args.video:
        plotter = VideoPlotter(
            loc[:2, :args.num_balls],
            loc[2:, :args.num_balls],
            field_positions=loc[:, args.num_balls:],
            field_charges=charges[args.num_balls:],
            grid_size=grid_size,
            interaction_strength=interaction_strength
        )
        plotter.new_fig()
        ani = plotter.plot_video()

        path = os.path.join('.', f'electrostatic_field_{args.split}_{args.idx}.mp4')
        ani.save(path, codec='mpeg4')
        plt.close(plotter.fig)
    else:
        fig = plt.figure()
        axes = plt.gca()
        if ndim == 2:
            axes.set_aspect('equal')
        min_range, max_range = plot_range(axes, loc)
        plot_trajectories(axes, loc, args.num_balls+args.static_balls, num_timesteps=25)
        plot_charges(axes, loc, charges, args.num_balls)
        strm = plot_field(
            axes, loc[0, args.num_balls:], charges[args.num_balls:],
            np.stack([min_range, max_range], axis=1), ndim=args.ndim,
            grid_size=grid_size, interaction_strength=interaction_strength, clip=10.0)

        plt.show()
        # fig.savefig(f'electrostatic_field_{args.split}_{args.idx}.png', dpi=300, bbox_inches='tight')
