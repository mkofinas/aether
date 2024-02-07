import torch
from torch.utils.data import DataLoader

from experiments.utils.collate import collate_field


def infer_fields(model, dataset, params, num_dims=2, num_batches=-1):
    gpu = params.get('gpu', False)
    batch_size = params.get('val_batch_size', 256)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu,
                             collate_fn=collate_field)
    model.eval()

    grid_size = params['grid_size']
    box_size = params['box_size']
    burn_in_steps = params['test_burn_in_steps']
    print('burn_in_steps', burn_in_steps)
    test_positions = model.create_grid_points(
        box_size=box_size, grid_size=grid_size, normalize=True)
    out_fields = {
        'predicted_field': [],
        'gt_field': [],
        'positions': test_positions
    }

    for batch_idx, batch in enumerate(data_loader):
        inputs = batch['inputs']
        field = batch.get('field', None)
        oracle = batch.get('oracle', None)

        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if oracle is not None:
                    oracle = oracle.cuda(non_blocking=True)

            pred_field = model.predict_field_at_grid(
                model_inputs, box_size=box_size, grid_size=grid_size,
                oracle=oracle
            ).cpu()
            gt_field = field(
                test_positions.unsqueeze(0).repeat(
                    model_inputs.size(0), 1, 1).to(model_inputs.device)
            ).cpu()
            out_fields['predicted_field'].append(pred_field)
            out_fields['gt_field'].append(gt_field)

        if 0 < num_batches < batch_idx:
            break

    out_fields['predicted_field'] = torch.cat(out_fields['predicted_field'], dim=0)
    out_fields['gt_field'] = torch.cat(out_fields['gt_field'], dim=0)
    return out_fields
