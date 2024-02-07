import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def eval_forward_prediction_dynamicvars_unnormalized(model, dataset, params):
    gpu = params.get('gpu', False)
    collate_fn = params.get('collate_fn', None)
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=gpu, collate_fn=collate_fn)
    model.eval()
    final_errors = torch.zeros(0)
    final_pos_errors = torch.zeros(0)
    final_vel_errors = torch.zeros(0)
    final_counts = torch.zeros(0)
    bad_count = 0
    report_error_norm = params.get('report_error_norm', False)

    for batch_ind, batch in enumerate(data_loader):
        print("DATA POINT ",batch_ind)
        inputs = batch['inputs']
        gt_preds = inputs[0, 1:]
        masks = batch['masks']
        node_inds = batch.get('node_inds', None)
        graph_info = batch.get('graph_info', None)
        burn_in_masks = batch['burn_in_masks']
        strict = params.get('strict', -1)
        if strict > 0:
            burn_in_masks[:, strict:] = 0.0
        pred_masks = (masks.float() - burn_in_masks)[0, 1:]

        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                burn_in_masks = burn_in_masks.cuda(non_blocking=True)
            model_preds = model.predict_future(inputs, masks, node_inds, graph_info, burn_in_masks)[0].cpu()

            unnorm_model_preds = dataset.unnormalize_data(model_preds)
            unnorm_gt_predictions = dataset.unnormalize_data(gt_preds)

            max_len = pred_masks.sum(dim=0).max().int().item()
            if max_len > len(final_errors):
                final_errors = torch.cat([final_errors, torch.zeros(max_len - len(final_errors))])
                final_pos_errors = torch.cat([final_pos_errors, torch.zeros(max_len - len(final_pos_errors))])
                final_vel_errors = torch.cat([final_vel_errors, torch.zeros(max_len - len(final_vel_errors))])
                final_counts = torch.cat([final_counts, torch.zeros(max_len - len(final_counts))])

            for var in range(masks.size(-1)):
                var_gt = unnorm_gt_predictions[:, var]
                var_preds = unnorm_model_preds[:, var]
                var_pred_masks = pred_masks[:, var]
                if not burn_in_masks[0, :, var].bool().any():
                    # print('Missing input')
                    continue
                if hasattr(dataset, 'non_linear_masks'):
                    non_linear_mask = dataset.non_linear_masks[batch_ind][var]
                    if not non_linear_mask:
                        continue
                if var_preds.size(-1) >= 5:
                    var_losses = F.mse_loss(var_preds[..., :4], var_gt[..., :4], reduction='none').mean(dim=-1)*var_pred_masks
                else:
                    var_losses = F.mse_loss(var_preds, var_gt, reduction='none').mean(dim=-1)*var_pred_masks
                if report_error_norm:
                    var_pos_losses = torch.norm(var_preds[..., :2] - var_gt[..., :2], dim=-1)*var_pred_masks
                    var_vel_losses = torch.norm(var_preds[..., 2:4] - var_gt[..., 2:4], dim=-1)*var_pred_masks
                else:
                    var_pos_losses = F.mse_loss(var_preds[..., :2], var_gt[..., :2], reduction='none').mean(dim=-1)*var_pred_masks
                    var_vel_losses = F.mse_loss(var_preds[..., 2:4], var_gt[..., 2:4], reduction='none').mean(dim=-1)*var_pred_masks
                tmp_inds = torch.nonzero(var_pred_masks)
                if len(tmp_inds) == 0:
                    continue
                for i in range(len(tmp_inds)-1):
                    if tmp_inds[i+1] - tmp_inds[i] != 1:
                        bad_count += 1
                        break
                num_entries = var_pred_masks.sum().int().item()
                final_errors[:num_entries] += var_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
                final_pos_errors[:num_entries] += var_pos_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
                final_vel_errors[:num_entries] += var_vel_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
                final_counts[:num_entries] += var_pred_masks[tmp_inds[0]:tmp_inds[0]+num_entries]
    print("FINAL BAD COUNT: ",bad_count)
    return final_errors/final_counts, final_pos_errors/final_counts, final_vel_errors/final_counts, final_counts
