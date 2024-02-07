import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import Affine2D

from experiments.utils.flags import build_flags
import nn.utils.abstract_model_builder as model_builder
from experiments.ind.single_ind_data import SingleIndData, ind_collate_fn
import experiments.ind.visualize as visualize
from experiments.utils.seed import set_seed


def plot_strict_ind(ax, X, pred=False, invert_yaxis=True, extent=None):
    pred_traj = X['pred'].clone()
    full_traj = X['in'].clone().squeeze()
    if invert_yaxis:
        pred_traj[..., 1] = -pred_traj[..., 1]
        full_traj[..., 1] = -full_traj[..., 1]
    burn_mask = X['burn_in_masks'].squeeze().bool()
    full_mask = X['mask'].squeeze().bool()
    object_sizes = X['object_sizes']
    num_steps = full_traj.shape[0]
    max_inp_step = burn_mask.nonzero()[:, 0].max().item()
    num_pred_steps = num_steps - max_inp_step - 1

    object_classes = X['object_classes']
    colors = ['#abd9e9', '#2c7bb6', '#d7191c']
    color_palette = [colors[oc] for oc in object_classes]

    full_traj[~(full_mask.bool())] = np.nan

    num_objects = full_traj.shape[1]
    pred_traj = torch.cat([full_traj[:max_inp_step+1], pred_traj[-num_pred_steps:]], 0)

    marker_range = np.linspace(1, 2, num_steps)
    alpha_range = np.linspace(0.1, 1.0, num_steps)
    alpha_range[:max_inp_step+1] = 0.2

    for i in range(num_objects):
        if full_mask[:max_inp_step+1, i].int().sum(0) < 2 or full_mask[max_inp_step+1:, i].int().sum(0) == 0:
            continue

        if not pred:
            ax.plot(full_traj[:max_inp_step+1, i, 0], full_traj[:max_inp_step+1, i, 1], '-',
                    color=color_palette[i], alpha=alpha_range[0])
            for t in range(max_inp_step+1, num_steps):
                if not full_mask[[t-1, t], i].all():
                    continue
                ax.plot(full_traj[[t-1, t], i, 0], full_traj[[t-1, t], i, 1], '-',
                        color=color_palette[i], alpha=alpha_range[t])

            # Use markers only for present onwards
            for t in range(max_inp_step, num_steps):
                if not full_mask[t, i]:
                    continue
                ax.plot(full_traj[t, i, 0], full_traj[t, i, 1], 'o',
                        color=color_palette[i] if t != max_inp_step else 'k',
                        alpha=alpha_range[t] if t != max_inp_step else 1.0,
                        markersize=marker_range[t])

                vel_norm = torch.norm(full_traj[t, i, 2:4])
                if vel_norm < 0.5:
                    continue
                vel_angle = torch.atan2(full_traj[t, i, 3], full_traj[t, i, 2])
                ax.add_patch(
                    plt.Rectangle(
                        (full_traj[t, i, 0] - object_sizes[i, 1] / 2,
                         full_traj[t, i, 1] - object_sizes[i, 0] / 2),
                        width=object_sizes[i, 1], height=object_sizes[i, 0],
                        alpha=alpha_range[t], facecolor='none', edgecolor=color_palette[i],
                        transform=(
                            Affine2D().rotate_deg_around(
                                *(full_traj[t, i, :2]), np.degrees(vel_angle))
                            + ax.transData
                        )
                    )
                )
        else:
            color_ = color_palette[i]
            alpha_ = alpha_range[0]
            ax.plot(pred_traj[:max_inp_step+1, i, 0], pred_traj[:max_inp_step+1, i, 1], '-',
                    color=color_palette[i], alpha=alpha_)
            for t in range(max_inp_step+1, num_steps):
                if not full_mask[[t-1, t], i].all():
                    continue
                if ((pred_traj[[t-1, t], i, 0] > extent[0]).any()
                    or (pred_traj[[t-1, t], i, 0] < 0.0).any()
                    or (pred_traj[[t-1, t], i, 1] < -extent[1]).any()
                    or (pred_traj[[t-1, t], i, 1] > 0.0).any()):

                    continue
                alpha_ = alpha_range[t]
                ax.plot(pred_traj[[t-1, t], i, 0], pred_traj[[t-1, t], i, 1], '-',
                        color=color_, alpha=alpha_)

            for t in range(max_inp_step, num_steps):
                if not full_mask[t, i]:
                    continue

                if ((pred_traj[t, i, 0] > extent[0]).any()
                    or (pred_traj[t, i, 0] < 0.0).any()
                    or (pred_traj[t, i, 1] < -extent[1]).any()
                    or (pred_traj[t, i, 1] > 0.0).any()):

                    continue
                alpha_ = alpha_range[t]
                alpha_ = alpha_ if t != max_inp_step else 1.0
                ax.plot(pred_traj[t, i, 0], pred_traj[t, i, 1], 'o',
                        color=color_ if t != max_inp_step else 'k',
                        alpha=alpha_, markersize=marker_range[t])

                vel_norm = torch.norm(pred_traj[t, i, 2:4])
                if vel_norm < 0.5:
                    continue
                vel_angle = torch.atan2(pred_traj[t, i, 3], pred_traj[t, i, 2])
                ax.add_patch(
                    plt.Rectangle(
                        (pred_traj[t, i, 0] - object_sizes[i, 1] / 2,
                         pred_traj[t, i, 1] - object_sizes[i, 0] / 2),
                        width=object_sizes[i, 1], height=object_sizes[i, 0],
                        alpha=alpha_, facecolor='none', edgecolor=color_,
                        transform=(
                            Affine2D().rotate_deg_around(
                                *(pred_traj[t, i, :2]), np.degrees(vel_angle))
                            + ax.transData
                        )
                    )
                )


def colornorm(c):
    r = c[1:3]
    g = c[3:5]
    b = c[5:]
    norm = (int(r, 16)/255.0, int(g, 16)/255.0, int(b, 16)/255.0)
    return np.linalg.norm(norm)


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
    parser.add_argument('--eval_sequentially', action='store_true')
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

    model_names = [
        'Aether',
    ]
    model_params = [
        {
            'model_type': 'nn.dynamicvars.aether_dynamicvars.AetherDynamicVars',
            'working_dir': 'experiments/ind/results/single_ind_processed/nn.dynamicvars.aether_dynamicvars.AetherDynamicVars/seed_1/',
            'vel_norm_norm': True,
            'data_path': 'experiments/ind/dataset/data/single_ind/',
        },
    ]

    vis_indices = list(range(40))

    different_model_preds = []
    model_list = []
    test_data = []
    for i in range(len(model_params)):
        params.update(model_params[i])
        test_data.append(SingleIndData(args.data_path, 'test', params))
        print(params)
        model = model_builder.build_model(params)

        test_subset = torch.utils.data.Subset(test_data[-1], vis_indices)
        vis_preds, _ = visualize.vis_forward_prediction_dynamicvars_unnormalized(model, test_subset, params)
        different_model_preds.append(vis_preds)
        model_list.append(model)

    c_norms = {k: colornorm(c) for k, c in mcolors.CSS4_COLORS.items()}
    c_mask = np.array(list(c_norms.values())) <= 1.2

    max_no = max([different_model_preds[0][i]['in'].shape[2]
                  for i in range(len(different_model_preds[0]))])

    num_rows = len(vis_indices)
    num_cols = len(model_params) + 1

    scale = 0.00814636091724502
    multiplier = 12
    figs = []
    for r in range(num_rows):
        figs.append(plt.figure())
        ax = figs[-1].add_subplot(1, num_cols, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        bg_image = test_data[0].load_road_image()
        # NOTE: Hardcoded conversion
        image_extent = np.array(bg_image.size) * scale * multiplier
        ax.imshow(bg_image, alpha=0.5,
                  extent=(0.0, image_extent[0], -image_extent[1], 0.0))

        plot_strict_ind(ax, different_model_preds[0][r],
                        invert_yaxis=False, extent=image_extent)
        ax.invert_yaxis()

        for c in range(1, num_cols):
            ax = figs[-1].add_subplot(1, num_cols, c+1, sharex=ax, sharey=ax)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.imshow(bg_image, alpha=0.5,
                      extent=(0.0, image_extent[0], -image_extent[1], 0.0))
            plot_strict_ind(ax, different_model_preds[c-1][r],
                            pred=True, invert_yaxis=False,
                            extent=image_extent)

    for idx, fig in enumerate(figs):
        file_name = f'experiments/ind/results/single_ind_processed/ind_qualitative_results_{idx}.png'
        fig.tight_layout()
        fig.savefig(file_name, bbox_inches='tight', dpi=300)
