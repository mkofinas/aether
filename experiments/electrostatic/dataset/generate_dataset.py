import os
import math
import time
import numpy as np
import argparse
import subprocess
import pickle
from itertools import cycle

from experiments.electrostatic.dataset.electrostatic_field_sim import ElectrostaticFieldSim
from experiments.gravitational.dataset.gravitational_field_sim import GravitationalFieldSim


def generate_electrostatic_dataset(num_sims, length, sample_freq, static_field=False,
                                   seed_iter=cycle([1])):
    ds = {
        "points": list(),
        "vel": list(),
        "edges": list(),
        "charges": list(),
        "E": list(),
        "U": list(),
        "K": list(),
        "delta_T": sim._delta_T,
        "sample_freq": sample_freq,
    }
    if num_sims == 0:
        return ds

    for i in range(num_sims):
        t = time.time()
        if not static_field:
            sim._field_seed = next(seed_iter)
        if static_field or args.field_change_frequency:
            sim.reset_field_rng()
        loc, vel, edges, charges = sim.sample_trajectory(T=length, sample_freq=sample_freq)

        energies = np.array(
            [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
             range(loc.shape[0])])

        ds["E"].append(energies[..., 0])
        ds["U"].append(energies[..., 1])
        ds["K"].append(energies[..., 2])

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        ds["points"].append(loc)
        ds["vel"].append(vel)
        ds["edges"].append(edges)
        ds["charges"].append(charges)

    for key in ["points", "vel", "edges", "E", "U", "K"]:
        ds[key] = np.stack(ds[key])
    for key in ["E", "U", "K"]:
        ds[key] = np.mean(ds[key], axis=0)
    ds["charges"] = np.stack(ds["charges"])

    return ds


def generate_gravitational_dataset(num_sims, length, sample_freq, static_field=False,
                                   seed_iter=cycle([1])):
    ds = {
        "points": list(),
        "vel": list(),
        "delta_T": sim.dt,
        "sample_freq": sample_freq,
    }
    if num_sims == 0:
        return ds

    for i in range(num_sims):
        t = time.time()
        if not static_field:
            sim._field_seed = next(seed_iter)
        if static_field or args.field_change_frequency:
            sim.reset_field_rng()
        loc, vel, _, _ = sim.sample_trajectory(T=length, sample_freq=sample_freq)

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        ds["points"].append(loc)
        ds["vel"].append(vel)

    for key in ["points", "vel"]:
        ds[key] = np.stack(ds[key])

    return ds


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='electrostatic_field',
                    help='What simulation to generate.')
parser.add_argument('--static_field', action='store_true',
                    help='If true, use the same field across all simulations')
parser.add_argument('--strength', type=float, default=1.0,
                    help='Strength of field particle charges/masses')
parser.add_argument('--name', type=str, default='new',
                    help='Add string to suffix of filename.')
parser.add_argument('--data_dir', type=str, default='.',
                    help='Directory to store the dataset')
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--field-change-frequency', type=int, default=10000,
                    help='Change field every so often.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--static-balls', type=int, default=10,
                    help='Number of static balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--dim', type=int, default=3,
                    help='Spatial simulation dimension (2 or 3).')
parser.add_argument('--boxsize', type=float, default=5.0,
                    help='Size of a surrounding box. If 0, then no box.')

args = parser.parse_args()
args_dict = vars(args)
git_commit = subprocess.check_output(["git", "describe", "--always"]).strip()

if args.simulation == 'electrostatic_field':
    sim = ElectrostaticFieldSim(noise_var=0.0, n_balls=args.n_balls,
                                static_balls=args.static_balls, box_size=args.boxsize,
                                dim=args.dim, static_charge_strength=args.strength)
    suffix = '_electrostatic_field_' + str(args.dim) + 'D_'
    generate_dataset = generate_electrostatic_dataset
elif args.simulation == 'gravitational_field':
    sim = GravitationalFieldSim(noise_var=0.0, n_balls=args.n_balls,
                                static_balls=args.static_balls, dim=args.dim,
                                static_mass=args.strength)
    suffix = '_gravitational_field_' + str(args.dim) + 'D_'
    generate_dataset = generate_gravitational_dataset
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls)
suffix += '_' + str(args.name)
np.random.seed(args.seed)

print(suffix)


train_seeds = math.ceil(args.num_train / args.field_change_frequency)
valid_seeds = math.ceil(args.num_valid / args.field_change_frequency)
test_seeds = math.ceil(args.num_test / args.field_change_frequency)
total_seeds = train_seeds + valid_seeds + test_seeds

seed_range = np.concatenate([
    np.repeat(np.arange(train_seeds),
              args.field_change_frequency)[:args.num_train],
    np.repeat(train_seeds + np.arange(valid_seeds),
              args.field_change_frequency)[:args.num_valid],
    np.repeat(train_seeds + valid_seeds + np.arange(test_seeds),
              args.field_change_frequency)[:args.num_test],
])
seed_range = sim._field_seed + seed_range
seed_range = iter(seed_range)

# Generate training and test dataset.
ds = dict()
print("Generating {} training simulations".format(args.num_train))

ds["train"] = generate_dataset(
    args.num_train, args.length, args.sample_freq,
    static_field=args.static_field, seed_iter=seed_range)
ds["train"]["git_commit"] = str(git_commit)
ds["train"]["args"] = args_dict

print("Generating {} validation simulations".format(args.num_valid))

ds["valid"] = generate_dataset(
    args.num_valid, args.length, args.sample_freq,
    static_field=args.static_field, seed_iter=seed_range)
ds["valid"]["git_commit"] = str(git_commit)
ds["valid"]["args"] = args_dict

print("Generating {} test simulations".format(args.num_test))
ds["test"] = generate_dataset(
    args.num_test, args.length_test, args.sample_freq,
    static_field=args.static_field, seed_iter=seed_range)
ds["test"]["git_commit"] = str(git_commit)
ds["test"]["args"] = args_dict

# Save dataset to file.
for ds_type in ["train",  "valid", "test"]:
    os.makedirs(args.data_dir, exist_ok=True)
    filename = os.path.join(args.data_dir, "ds_" + ds_type + suffix + ".pkl")
    with open(filename, "wb") as file:
        pickle.dump(ds[ds_type], file)
