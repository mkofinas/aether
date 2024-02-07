import inspect

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.utils.collate import collate_field


def eval_forward_prediction_unnormalized(
    model, dataset, burn_in_steps, forward_pred_steps, params,
    return_total_errors=False, num_dims=2
):
    dataset.return_edges = False
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu,
                             collate_fn=collate_field)
    model.eval()
    total_se = 0
    total_pos_se = 0
    total_vel_se = 0
    batch_count = 0
    all_errors = []
    all_pos_errors = []
    all_vel_errors = []
    report_error_norm = params.get('report_error_norm', False)
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        charges = batch.get('charges', None)
        field = batch.get('field', None)

        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if charges is not None:
                    charges = charges.cuda(non_blocking=True)
            # NOTE: Hacky solution
            optional_kwargs = {}
            if 'charges' in inspect.getargspec(model.calculate_loss).args:
                optional_kwargs.update({'charges': charges})
            if 'field' in inspect.getargspec(model.calculate_loss).args:
                optional_kwargs.update({'field': field})
            model_preds = model.predict_future(
                model_inputs, forward_pred_steps, **optional_kwargs).cpu()
            # NOTE: End Hacky solution
            if isinstance(dataset, torch.utils.data.Subset):
                unnorm_model_preds = dataset.dataset.torch_unnormalize(model_preds)
                unnorm_gt_predictions = dataset.dataset.torch_unnormalize(gt_predictions)
            else:
                unnorm_model_preds = dataset.torch_unnormalize(model_preds)
                unnorm_gt_predictions = dataset.torch_unnormalize(gt_predictions)
            batch_count += 1

            if report_error_norm:
                pos_errors = torch.norm(unnorm_model_preds[..., :num_dims] - unnorm_gt_predictions[..., :num_dims], dim=-1).mean(-1)
                vel_errors = torch.norm(unnorm_model_preds[..., num_dims:] - unnorm_gt_predictions[..., num_dims:], dim=-1).mean(-1)
            else:
                pos_errors = F.mse_loss(unnorm_model_preds[..., :num_dims], unnorm_gt_predictions[..., :num_dims], reduction='none').view(unnorm_model_preds.size(0), unnorm_model_preds.size(1), -1).mean(dim=-1)
                vel_errors = F.mse_loss(unnorm_model_preds[..., num_dims:], unnorm_gt_predictions[..., num_dims:], reduction='none').view(unnorm_model_preds.size(0), unnorm_model_preds.size(1), -1).mean(dim=-1)
            if return_total_errors:
                all_errors.append(F.mse_loss(unnorm_model_preds, unnorm_gt_predictions, reduction='none').view(unnorm_model_preds.size(0), unnorm_model_preds.size(1), -1).mean(dim=-1))
                all_pos_errors.append(pos_errors)
                all_vel_errors.append(vel_errors)
            else:
                total_se += F.mse_loss(unnorm_model_preds, unnorm_gt_predictions, reduction='none').view(unnorm_model_preds.size(0), unnorm_model_preds.size(1), -1).mean(dim=-1).sum(dim=0)
                total_pos_se += pos_errors.sum(dim=0)
                total_vel_se += vel_errors.sum(dim=0)
    if return_total_errors:
        return torch.cat(all_errors, dim=0), torch.cat(all_pos_errors, dim=0), torch.cat(all_vel_errors, dim=0)
    else:
        return total_se / len(dataset), total_pos_se / len(dataset), total_vel_se / len(dataset)
