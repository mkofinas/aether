import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from PIL import Image

from experiments.utils.flags import build_flags
import nn.utils.abstract_model_builder as model_builder
from experiments.ind.single_ind_data import SingleIndData, ind_collate_fn
import experiments.ind.train_dynamicvars as train
from experiments.utils import train_utils
from experiments.ind.evaluate import eval_forward_prediction_dynamicvars_unnormalized
from experiments.utils.seed import set_seed


if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='val_prediction_errors.npy')
    parser.add_argument('--train_data_len', type=int, default=-1)
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--expand_train', action='store_true')
    parser.add_argument('--final_test', action='store_true')
    parser.add_argument('--vel_norm_norm', action='store_true')
    parser.add_argument('--report_error_norm', action='store_true')
    parser.add_argument('--test_short_sequences', action='store_true')
    parser.add_argument('--present_gnn', action='store_true')
    parser.add_argument('--pos_representation', choices=['cart', 'polar'], default='polar')
    parser.add_argument('--strict', type=int, default=-1)

    args = parser.parse_args()
    params = vars(args)

    set_seed(args.seed)

    params['input_size'] = 4
    params['nll_loss_type'] = 'gaussian'
    params['dynamic_vars'] = True
    params['collate_fn'] = ind_collate_fn

    model = model_builder.build_model(params)
    if args.mode == 'train':
        train_data = SingleIndData(args.data_path, 'train', params)
        val_data = SingleIndData(args.data_path, 'valid', params)
        with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            train.train(model, train_data, val_data, params, train_writer, val_writer)
    elif args.mode == 'eval':
        if args.final_test:
            test_data = SingleIndData(args.data_path, 'test', params)
            test_mse, test_pos_mse, test_vel_mse, counts = eval_forward_prediction_dynamicvars_unnormalized(model, test_data, params)
        else:
            val_data = SingleIndData(args.data_path, 'valid', params)
            test_mse, test_pos_mse, test_vel_mse, counts = eval_forward_prediction_dynamicvars_unnormalized(model, val_data, params)

        file_prefix = "norm_" if args.report_error_norm else ""
        error_file_name = file_prefix + args.error_out_name

        if not os.path.exists(args.working_dir):
            os.makedirs(args.working_dir, exist_ok=True)
        path = os.path.join(args.working_dir, error_file_name)
        np.save(path, test_mse.cpu().numpy())
        path = os.path.join(args.working_dir, 'counts_' + error_file_name)
        np.save(path, counts.cpu().numpy())

        mid_pred_step = int(len(test_mse) / 2)
        test_mse_1 = test_mse[0].item()
        test_mse_mid = test_mse[mid_pred_step].item()
        test_mse_final = test_mse[-1].item()
        if args.final_test:
            print("TEST FORWARD PRED RESULTS:")
        else:
            print("VAL FORWARD PRED RESULTS:")
        print("\t1 STEP:  ", test_mse_1, counts[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_mse_mid, counts[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_mse_final, counts[-1].item())

        pos_path = os.path.join(args.working_dir, 'pos_' + error_file_name)
        np.save(pos_path, test_pos_mse.cpu().numpy())
        vel_path = os.path.join(args.working_dir, 'vel_' + error_file_name)
        np.save(vel_path, test_vel_mse.cpu().numpy())

        print("POSITION FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_pos_mse[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_pos_mse[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_pos_mse[-1].item())

        print("VELOCITY FORWARD PRED RESULTS:")
        print("\t1 STEP: ", test_vel_mse[0].item())
        print(f"\t{mid_pred_step+1} STEP: ", test_vel_mse[mid_pred_step].item())
        print(f"\t{len(test_mse)} STEP: ", test_vel_mse[-1].item())
    elif args.mode == 'visualize_field':
        print('Visualizing field...')

        val_data = SingleIndData(args.data_path, 'valid', params)

        # NOTE: Hardcoded for now
        scale = 0.00814636091724502
        multiplier = 12
        speed_norm = torch.load(os.path.join(val_data._data_path, 'train_speed_norm_stats'))
        map_img = val_data.load_road_image()
        image_extent = np.array(map_img.size) * scale * multiplier
        image_extent = image_extent.astype(np.float32)
        model.cpu()
        field_min = np.array([0.0, -image_extent[1]]) / speed_norm
        field_max = np.array([image_extent[0], 0.0]) / speed_norm

        grid_size = 201
        angle_grid_size = 36
        predicted_field, test_positions = model.field_at_se2_range(
            field_min, field_max, grid_size=grid_size,
            angle_grid_size=angle_grid_size)
        predicted_field = predicted_field.detach().cpu()

        force_field_color = predicted_field[..., :2].norm(dim=-1)
        test_positions[..., :2] *= speed_norm
        center_point = test_positions[..., :2].mean([0, 1])[0]

        force_norm = mpl.colors.Normalize(vmin=0.0, vmax=force_field_color.max())
        force_cmap = mpl.cm.plasma
        force_colormap = mpl.cm.ScalarMappable(
            norm=force_norm, cmap=force_cmap)

        def update_force_field(angle_idx):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(map_img, alpha=0.4, extent=(0.0, image_extent[0], -image_extent[1], 0.0))
            ax.quiver(
                test_positions[:, :, angle_idx, 0],
                test_positions[:, :, angle_idx, 1],
                predicted_field[:, :, angle_idx, 0],
                predicted_field[:, :, angle_idx, 1],
                force_field_color[:, :, angle_idx],
                alpha=0.6, cmap=force_cmap)

            # Visualize angle
            angle_vec = np.array([
                np.cos(angle_idx*2.0*np.pi/angle_grid_size),
                np.sin(angle_idx*2.0*np.pi/angle_grid_size),
            ])
            ax.quiver(*center_point, *angle_vec)

        fig, ax = plt.subplots()
        angle_idx = 0
        ani = animation.FuncAnimation(fig, update_force_field, interval=75,
                                      frames=angle_grid_size, repeat=False)
        ani.save(os.path.join(args.working_dir, 'single_ind_force_field.mp4'),
                 dpi=300, codec='mpeg4', bitrate=8000)

        for i in range(2):
            for j in range(2):
                fig, ax = plt.subplots()
                angle_idx = i * (angle_grid_size // 2) + j * (angle_grid_size // 4)
                ax.set_xticks([])
                ax.set_yticks([])

                ax.imshow(map_img, alpha=0.4, extent=(0.0, image_extent[0], -image_extent[1], 0.0))
                ax.quiver(
                    test_positions[:, :, angle_idx, 0],
                    test_positions[:, :, angle_idx, 1],
                    predicted_field[:, :, angle_idx, 0],
                    predicted_field[:, :, angle_idx, 1],
                    force_field_color[:, :, angle_idx],
                    alpha=0.6, cmap=force_cmap)

                # Visualize angle
                angle_vec = np.array([
                    np.cos(angle_idx*2.0*np.pi/angle_grid_size),
                    np.sin(angle_idx*2.0*np.pi/angle_grid_size),
                ])
                ax.quiver(*center_point, *angle_vec)
                fig.savefig(
                    os.path.join(args.working_dir, f'single_ind_force_field_{angle_idx*360.0/angle_grid_size:.1f}.png'),
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)
