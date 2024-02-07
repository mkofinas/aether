import os
import re
import inspect

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.colors as mplcolors

from experiments.utils.flags import build_flags
import nn.utils.abstract_model_builder as model_builder
from experiments.electrostatic.static_electrostatic_field_data import StaticFieldData
import experiments.electrostatic.train as train
from experiments.utils import train_utils
from experiments.electrostatic.evaluate import eval_forward_prediction_unnormalized
from experiments.utils.seed import set_seed
from experiments.electrostatic.electrostatic_field import ElectrostaticField


def pcolor_quiver_plot(ax, x_linspace, field, field_norm, sample_step):
    x, y = np.meshgrid(x_linspace, x_linspace)
    _ = ax.pcolormesh(
        x[::sample_step, ::sample_step], y[::sample_step, ::sample_step],
        field_norm[::sample_step, ::sample_step],
        norm=mplcolors.LogNorm(vmin=field_norm.min(), vmax=field_norm.max()),
        cmap='plasma', shading='auto')
    unit_field = field / field_norm[..., None]
    # Use 4x the sample step for the quiver plot
    quiver_step = 4 * sample_step
    # Plot a quiver plot of the field
    quiv = ax.quiver(
        x_linspace[::quiver_step], x_linspace[::quiver_step],
        unit_field[::quiver_step, ::quiver_step, 0],
        unit_field[::quiver_step, ::quiver_step, 1],
        angles='xy', scale_units='xy',
    )
    return quiv


def setup_plot(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def save_samples(model, dataset, num_samples, params, end_idx=-1):
    gpu = params.get('gpu', False)
    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    burn_in_steps = params['test_burn_in_steps']
    forward_pred_steps = params['test_pred_steps']
    all_inputs = []
    all_outputs = []
    for batch_idx, batch in enumerate(data_loader):
        inputs = batch['inputs']
        charges = batch.get('charges', None)
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if charges is not None:
                    charges = charges.cuda(non_blocking=True)
            if 'charges' in inspect.getargspec(model.calculate_loss).args:
                model_preds = model.predict_future(
                    model_inputs, forward_pred_steps, charges=charges).cpu()
            else:
                model_preds = model.predict_future(
                    model_inputs, forward_pred_steps).cpu()
        unnormalized_preds = dataset.unnormalize(model_preds)
        unnormalized_gt = dataset.unnormalize(inputs)
        all_inputs.append(unnormalized_gt)
        all_outputs.append(unnormalized_preds)

        if 0 < end_idx < batch_idx:
            break

    inputs_file_name = os.path.join(
        params['working_dir'], f'all_inputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(inputs_file_name, all_inputs)

    outputs_file_name = os.path.join(
        params['working_dir'], f'all_outputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(outputs_file_name, all_outputs)

    test_mse, _, _ = eval_forward_prediction_unnormalized(
        model, dataset, burn_in_steps, forward_pred_steps, params,
        return_total_errors=True, num_dims=params['ndim'])
    error_file_name = os.path.join(params['working_dir'], f'all_errors_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(error_file_name, test_mse.numpy())


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--same_data_norm', action='store_true')
    parser.add_argument('--symmetric_data_norm', action='store_true')
    parser.add_argument('--vel_norm_norm', action='store_true')
    parser.add_argument('--no_data_norm', action='store_true')
    parser.add_argument('--error_out_name', default='{:s}prediction_errors_{:d}_{:d}_step.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--rff_std', type=float, default=1.0)
    parser.add_argument('--test_burn_in_steps', type=int, default=29)
    parser.add_argument('--test_pred_steps', type=int, default=20)
    parser.add_argument('--version_checkpoints', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--delta_t', type=float, default=1e-1)
    parser.add_argument('--error_suffix')
    parser.add_argument('--val_teacher_forcing_steps', type=int, default=-1)
    parser.add_argument('--subject_ind', type=int, default=-1)
    parser.add_argument('--plot_samples', action='store_true')
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')

    args = parser.parse_args()
    params = vars(args)

    set_seed(args.seed)

    params['num_vars'] = 5
    params['input_size'] = 4
    params['input_time_steps'] = 49
    params['use_3d'] = False
    params['ndim'] = 3 if params['use_3d'] else 2
    params['nll_loss_type'] = 'gaussian'
    train_data = StaticFieldData(args.data_path, 'train', params)
    val_data = StaticFieldData(args.data_path, 'val', params)

    electrostatic_field = ElectrostaticField(
        torch.load(os.path.join(args.data_path, 'static_field')),
        torch.load(os.path.join(args.data_path, 'static_charges')),
        dataset=train_data, delta_t=1e-3, device='cpu'
    )
    params['field'] = electrostatic_field

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer,
                        val_writer, input_steps=args.test_burn_in_steps,
                        pred_steps=args.test_pred_steps,
                        version_checkpoints=args.version_checkpoints)
    elif args.mode == 'eval':
        test_data = StaticFieldData(args.data_path, 'test', params)
        forward_pred = args.test_pred_steps

        test_mse, test_pos_mse, test_vel_mse = eval_forward_prediction_unnormalized(
            model, test_data, args.test_burn_in_steps, forward_pred, params,
            num_dims=params['ndim'])
        error_file_name = args.error_out_name.format(
            "norm_" if args.report_error_norm else "", args.test_burn_in_steps, args.test_pred_steps)
        print(error_file_name)
        path = os.path.join(args.working_dir, error_file_name)
        np.save(path, test_mse.cpu().numpy())
        pos_error_file_name = 'pos_' + error_file_name
        pos_path = os.path.join(args.working_dir, pos_error_file_name)
        np.save(pos_path, test_pos_mse.cpu().numpy())
        vel_error_file_name = 'vel_' + error_file_name
        vel_path = os.path.join(args.working_dir, vel_error_file_name)
        np.save(vel_path, test_vel_mse.cpu().numpy())
        test_mse_1 = test_mse[0].item()
        test_mse_10 = test_mse[9].item()
        test_mse_final = test_mse[-1].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_mse_1)
        print("\t10 STEP: ", test_mse_10)
        print(f"\t{len(test_mse)} STEP: ", test_mse_final)

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print("\t10 STEP: ", test_pos_mse[9].item())
        print(f"\t{len(test_mse)} STEP: ", test_pos_mse[-1].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print("\t10 STEP: ", test_vel_mse[9].item())
        print(f"\t{len(test_mse)} STEP: ", test_vel_mse[-1].item())
    elif args.mode == 'save_pred':
        test_data = StaticFieldData(args.data_path, 'test', params)
        save_samples(model, test_data, args.test_burn_in_steps, params,
                     end_idx=args.end_idx)
    elif args.mode == 'visualize_field':
        checkpoints = [
            f for f in os.listdir(params['working_dir'])
            if re.match('^checkpoint_[0-9]+$', f)
        ]
        print(checkpoints)
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
        box_size = 5.0
        grid_size = 1001
        sample_step = 5
        end_idx = 10

        grid_points = np.linspace(-box_size, box_size, grid_size)
        grid = electrostatic_field._normalize(electrostatic_field._make_grid(box_size=box_size, grid_size=grid_size))

        gt_field = electrostatic_field.grid_field(box_size=box_size, grid_size=grid_size).reshape(grid_size, grid_size, 2).cpu().numpy()
        gt_field_mag = np.linalg.norm(gt_field, axis=-1)
        field_positions = electrostatic_field.positions.cpu().numpy()

        for idx, chkpt in enumerate(checkpoints):
            epoch = chkpt.split('_')[-1]
            model.load(os.path.join(params['working_dir'], chkpt))
            model.eval()
            model.cpu()

            print(chkpt)
            predicted_field, _ = model.predict_field(grid)

            predicted_field = predicted_field.cpu().detach().numpy().reshape(grid_size, grid_size, 2)
            predicted_field_mag = np.linalg.norm(predicted_field, axis=-1)

            fig, ax = plt.subplots(1, 2)
            im0 = pcolor_quiver_plot(ax[0], grid_points, predicted_field,
                                     predicted_field_mag, sample_step)
            ax[0].set_title(f'Predicted Field, Epoch {epoch}')
            setup_plot(ax[0])
            ax[0].axis([-box_size, box_size, -box_size, box_size])
            im1 = pcolor_quiver_plot(ax[1], grid_points, gt_field, gt_field_mag,
                                     sample_step)
            ax[1].set_title('Groundtruth Field')
            setup_plot(ax[1])
            ax[1].axis([-box_size, box_size, -box_size, box_size])
            fig.savefig(os.path.join(args.working_dir, f'{chkpt}.png'), dpi=300, bbox_inches='tight')
            print(os.path.join(args.working_dir, f'{chkpt}.png'))
            plt.close(fig)

            if idx == end_idx - 1:
                break
