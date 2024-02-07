from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def vis_forward_prediction_dynamicvars_unnormalized(model, dataset, params):
    gpu = params.get('gpu', False)
    collate_fn = params.get('collate_fn', None)
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=gpu, collate_fn=collate_fn)
    model.eval()
    final_errors = torch.zeros(0)
    final_counts = torch.zeros(0)
    bad_count = 0
    vis_preds = []
    batch_errors = {}
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
                gt_preds = gt_preds.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                burn_in_masks = burn_in_masks.cuda(non_blocking=True)
            model_preds = model.predict_future(inputs, masks, node_inds, graph_info, burn_in_masks)[0]

            if isinstance(dataset, torch.utils.data.Subset):
                unnorm_model_preds = dataset.dataset.unnormalize_data(model_preds)
                unnorm_gt_predictions = dataset.dataset.unnormalize_data(gt_preds)
                unnorm_inputs = dataset.dataset.unnormalize_data(inputs.cpu())
            else:
                unnorm_model_preds = dataset.unnormalize_data(model_preds)
                unnorm_gt_predictions = dataset.unnormalize_data(gt_preds)
                unnorm_inputs = dataset.unnormalize_data(inputs.cpu())

            max_len = pred_masks.sum(dim=0).max().int().item()
            if max_len > len(final_errors):
                final_errors = torch.cat([final_errors, torch.zeros(max_len - len(final_errors))])
                final_counts = torch.cat([final_counts, torch.zeros(max_len - len(final_counts))])

            vis_preds.append(
                {
                    'mask': masks.cpu(),
                    'in': unnorm_inputs.cpu(),
                    'gt': unnorm_gt_predictions.cpu(),
                    'pred': unnorm_model_preds.cpu(),
                    'burn_in_masks': burn_in_masks.cpu(),
                    'object_sizes': dataset.dataset.object_sizes[batch_ind],
                    'object_classes': dataset.dataset.object_classes[batch_ind]
                }
            )
            unnorm_model_preds = unnorm_model_preds.cpu()
            unnorm_gt_predictions = unnorm_gt_predictions.cpu()
            for var in range(masks.size(-1)):
                var_gt = unnorm_gt_predictions[:, var]
                var_preds = unnorm_model_preds[:, var]
                # var_gt = gt_preds[:, var]
                # var_preds = model_preds[:, var]
                var_pred_masks = pred_masks[:, var]
                var_losses = F.mse_loss(var_preds, var_gt, reduction='none').mean(dim=-1)*var_pred_masks
                tmp_inds = torch.nonzero(var_pred_masks)
                if len(tmp_inds) == 0:
                    continue
                for i in range(len(tmp_inds)-1):
                    if tmp_inds[i+1] - tmp_inds[i] != 1:
                        bad_count += 1
                        break
                num_entries = var_pred_masks.sum().int().item()
                final_errors[:num_entries] += var_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
                final_counts[:num_entries] += var_pred_masks[tmp_inds[0]:tmp_inds[0]+num_entries]
                batch_errors[batch_ind] = var_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries].mean()
    # NOTE: batch_errors is incomplete
    return vis_preds, batch_errors
