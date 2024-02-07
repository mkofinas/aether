import os
import inspect

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from experiments.utils.flags import build_flags
import nn.utils.abstract_model_builder as model_builder
from experiments.gravitational.dynamic_gravitational_field_data import GravityDynamicFieldData as DynamicFieldData
import experiments.electrostatic.train as train
from experiments.utils import train_utils
from experiments.electrostatic.evaluate import eval_forward_prediction_unnormalized
from experiments.gravitational.evaluate import infer_fields
from experiments.utils.seed import set_seed
from experiments.utils.collate import collate_field


def quiver(ax, positions, field, field_norm, ndim):
    if ndim == 2:
        ax.quiver(
            positions[:, 0], positions[:, 1], field[:, 0], field[:, 1],
            field_norm, units='dots', cmap='plasma'
        )
    else:
        q = ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            field[:, 0], field[:, 1], field[:, 2], length=0.2, normalize=True,
            cmap='plasma'
        )
        q.set_array(field_norm.flatten())


def save_samples(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_field)
    model.eval()
    burn_in_steps = params['test_burn_in_steps']
    forward_pred_steps = params['test_pred_steps']
    all_inputs = []
    all_outputs = []
    samples_so_far = 0
    for batch in data_loader:
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
        unnormalized_preds = dataset.unnormalize(model_preds).numpy()
        unnormalized_gt = dataset.unnormalize(inputs).numpy()
        all_inputs.append(unnormalized_gt)
        all_outputs.append(unnormalized_preds)
        samples_so_far += batch['inputs'].shape[0]
        if samples_so_far >= num_samples:
            break

    inputs_file_name = os.path.join(
        params['working_dir'], f'all_inputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(inputs_file_name, all_inputs)

    outputs_file_name = os.path.join(
        params['working_dir'], f'all_outputs_{burn_in_steps}_{forward_pred_steps}.npy')
    np.save(outputs_file_name, all_outputs)


def setup_plot(ax, ndim):
    ax.set_xticks([])
    ax.set_yticks([])
    if ndim == 2:
        ax.set_aspect('equal')
    else:
        ax.set_zticks([])


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--normalization', default='speed_norm',
                        choices=['speed_norm', 'no_norm', 'same_norm',
                                 'min_max_norm'])
    parser.add_argument('--error_out_name', default='{:s}prediction_errors_{:d}_{:d}_step.npy')
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--test_burn_in_steps', type=int, default=44)
    parser.add_argument('--test_pred_steps', type=int, default=5)
    parser.add_argument('--graph_hidden', type=int, default=32)
    parser.add_argument('--mlp_hidden', type=int, default=256)
    parser.add_argument('--delta_t', type=float, default=1e-3)
    parser.add_argument('--error_suffix')
    parser.add_argument('--val_teacher_forcing_steps', type=int, default=-1)
    parser.add_argument('--plot_samples', action='store_true')
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--use_3d', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')
    parser.add_argument('--eval_split', default='val',
                        choices=['train', 'test', 'val'])

    args = parser.parse_args()
    params = vars(args)

    set_seed(args.seed)

    params['num_vars'] = 5
    params['input_time_steps'] = 44
    params['ndim'] = 3 if params['use_3d'] else 2
    params['input_size'] = 2 * params['ndim']
    params['nll_loss_type'] = 'gaussian'
    train_data = DynamicFieldData(args.data_path, 'train', params)
    val_data = DynamicFieldData(args.data_path, 'val', params)

    params['field'] = train_data.field

    model = model_builder.build_model(params)
    if args.mode == 'train':
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer,
                        val_writer, input_steps=args.test_burn_in_steps,
                        pred_steps=args.test_pred_steps, dynamic=True)
    elif args.mode == 'eval':
        test_data = DynamicFieldData(args.data_path, 'test', params)
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
        test_mse_final = test_mse[-1].item()
        print("FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_mse_1)
        print(f"\t{len(test_mse)} STEP: ", test_mse_final)

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print(f"\t{len(test_mse)} STEP: ", test_pos_mse[-1].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print(f"\t{len(test_mse)} STEP: ", test_vel_mse[-1].item())

    elif args.mode == 'save_pred':
        test_data = DynamicFieldData(args.data_path, 'test', params)
        save_samples(model, test_data, 10, params)
    elif args.mode == 'visualize_field':
        if args.eval_split == 'train':
            eval_data = train_data
        elif args.eval_split == 'val':
            eval_data = val_data
        else:
            eval_data = DynamicFieldData(args.data_path, 'test', params)
        params['grid_size'] = 8
        params['box_size'] = 5.0
        fields = infer_fields(model, eval_data, params, num_dims=params['ndim'])

        test_positions = fields['positions']
        gt_field = fields['gt_field']
        pred_field = fields['predicted_field']
        gt_field_mag = gt_field.norm(dim=-1)
        gt_field_unit_mag = gt_field / gt_field_mag.unsqueeze(-1)
        pred_field_mag = pred_field.norm(dim=-1)
        pred_field_unit_mag = pred_field / pred_field_mag.unsqueeze(-1)

        batch_indices = [0, 100, 200, 300, 400] if args.eval_split == 'train' else [0]
        fig, ax = plt.subplots(
            len(batch_indices), 2, squeeze=False,
            subplot_kw={'projection': '3d'} if params['use_3d'] else None
        )

        for i, batch_idx in enumerate(batch_indices):
            quiver(
                ax[i, 0],
                test_positions[..., :params['ndim']],
                pred_field[batch_idx, ..., :params['ndim']],
                pred_field_unit_mag[batch_idx],
                params['ndim'],
            )
            setup_plot(ax[i, 0], params['ndim'])
            ax[i, 0].set_title('Predicted Field')
            quiver(
                ax[i, 1],
                test_positions[..., :params['ndim']],
                gt_field[batch_idx, ..., :params['ndim']],
                gt_field_unit_mag[batch_idx],
                params['ndim'],
            )
            setup_plot(ax[i, 1], params['ndim'])
            ax[i, 1].set_title('Groundtruth Field')

        fig.savefig(
            os.path.join(args.working_dir, f"gravitational_field_{params['ndim']}d_{args.eval_split}_{batch_idx}.png"),
            dpi=300,
            bbox_inches='tight'
        )
        # plt.show()
        plt.close()
