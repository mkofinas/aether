import inspect
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.electrostatic.evaluate import eval_forward_prediction_unnormalized
from experiments.utils import train_utils
from experiments.utils.seed import set_seed
from experiments.utils.collate import collate_field


def train(model, train_data, val_data, params, train_writer, val_writer,
          input_steps=None, pred_steps=None, version_checkpoints=0,
          dynamic=False):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    val_batch_size = params.get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params.get('accumulate_steps')
    training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    normalize_nll = params.get('normalize_nll', False)
    tune_on_mse = params.get('tune_on_mse', False)
    tune_on_nll = params.get('tune_on_nll', False)
    verbose = params.get('verbose', False)
    val_teacher_forcing = params.get('val_teacher_forcing', False)
    continue_training = params.get('continue_training', False)
    train_data_loader = DataLoader(train_data, batch_size=batch_size,
                                   shuffle=True, drop_last=True,
                                   collate_fn=collate_field)
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size,
                                 collate_fn=collate_field)
    lr = params['lr']
    wd = params.get('wd', 0.)
    mom = params.get('mom', 0.)

    model_params = [param for param in model.parameters() if param.requires_grad]
    if params.get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if continue_training:
        print("RESUMING TRAINING")
        model.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_loss = train_params['best_val_loss']
        best_val_epoch = train_params['best_val_epoch']
        print("STARTING EPOCH: ",start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_loss = float('inf')
        best_val_result = 10000000

        # Save a checkpoint before we start training
        if version_checkpoints > 0:
            version_chkpt_dir = os.path.join(working_dir, 'checkpoint_0')
            model.save(version_chkpt_dir)

    training_scheduler = train_utils.build_scheduler(opt, params)
    end = start = 0
    set_seed(1)

    for epoch in range(start_epoch, num_epochs+1):
        print("EPOCH", epoch, (end-start))
        model.train()
        model.train_percent = epoch / num_epochs
        start = time.time()
        for batch_ind, batch in enumerate(train_data_loader):
            inputs = batch['inputs']
            charges = batch.get('charges', None)
            field = batch.get('field', None)
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                if charges is not None:
                    charges = charges.cuda(non_blocking=True)
            # NOTE: Using burn-in ONLY for gravity
            if dynamic:
                inputs = inputs[:, :input_steps]
            # NOTE: Hacky solution
            optional_kwargs = {}
            if 'charges' in inspect.getargspec(model.calculate_loss).args:
                optional_kwargs.update({'charges': charges})
            if 'field' in inspect.getargspec(model.calculate_loss).args:
                optional_kwargs.update({'field': field})
            loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(
                inputs, is_train=True, return_logits=True, **optional_kwargs)
            # NOTE: END Hacky solution
            loss.backward()
            if verbose:
                print("\tBATCH %d OF %d: %f, %f, %f" % (batch_ind+1, len(train_data_loader), loss.item(), loss_nll.mean().item(), loss_kl.mean().item()))
            if accumulate_steps == -1 or (batch_ind+1) % accumulate_steps == 0:
                if verbose and accumulate_steps > 0:
                    print("\tUPDATING WEIGHTS")
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                elif clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                opt.step()
                opt.zero_grad()
                if accumulate_steps > 0 and accumulate_steps > len(train_data_loader) - batch_ind - 1:
                    break

        if training_scheduler is not None:
            training_scheduler.step()

        if train_writer is not None:
            train_writer.add_scalar('loss', loss.item(), global_step=epoch)
            if normalize_nll:
                train_writer.add_scalar('NLL', loss_nll.mean().item(), global_step=epoch)
            else:
                train_writer.add_scalar('NLL', loss_nll.mean().item()/(inputs.size(1)*inputs.size(2)), global_step=epoch)

            train_writer.add_scalar("KL Divergence", loss_kl.mean().item(), global_step=epoch)
        model.eval()
        opt.zero_grad()

        total_nll = 0
        total_kl = 0
        if verbose:
            print("COMPUTING VAL LOSSES")
        with torch.no_grad():
            for batch_ind, batch in enumerate(val_data_loader):
                inputs = batch['inputs']
                charges = batch.get('charges', None)
                field = batch.get('field', None)
                if gpu:
                    inputs = inputs.cuda(non_blocking=True)
                    if charges is not None:
                        charges = charges.cuda(non_blocking=True)

                # NOTE: Using burn-in ONLY for gravity
                if dynamic:
                    inputs = inputs[:, :input_steps]
                # NOTE: Hacky solution
                optional_kwargs = {}
                if 'charges' in inspect.getargspec(model.calculate_loss).args:
                    optional_kwargs.update({'charges': charges})
                if 'field' in inspect.getargspec(model.calculate_loss).args:
                    optional_kwargs.update({'field': field})
                loss, loss_nll, loss_kl, logits, _ = model.calculate_loss(
                    inputs, is_train=False, teacher_forcing=val_teacher_forcing,
                    return_logits=True, **optional_kwargs)
                # NOTE: END Hacky solution
                total_kl += loss_kl.sum().item()
                total_nll += loss_nll.sum().item()

                if verbose:
                    print("\tVAL BATCH %d of %d: %f, %f"%(batch_ind+1, len(val_data_loader), loss_nll.mean(), loss_kl.mean()))

        total_kl /= len(val_data)
        total_nll /= len(val_data)
        total_loss = model.kl_coef*total_kl + total_nll

        with torch.no_grad():
            val_mse, _, _ = eval_forward_prediction_unnormalized(
                model, val_data, input_steps, pred_steps, params,
                return_total_errors=False, num_dims=3 if params['use_3d'] else 2)
        mean_mse = val_mse.mean()

        if val_writer is not None:
            val_writer.add_scalar('loss', total_loss, global_step=epoch)
            val_writer.add_scalar("NLL", total_nll, global_step=epoch)
            val_writer.add_scalar("KL Divergence", total_kl, global_step=epoch)
            val_writer.add_scalar("MSE", mean_mse, global_step=epoch)
        if tune_on_mse:
            tuning_loss = mean_mse
        elif tune_on_nll:
            tuning_loss = total_nll
        else:
            tuning_loss = total_loss
        if tuning_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = tuning_loss
            best_val_loss = total_nll
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        model.save(checkpoint_dir)
        if version_checkpoints > 0 and epoch % version_checkpoints == 0:
            version_chkpt_dir = os.path.join(working_dir, f'checkpoint_{epoch}')
            model.save(version_chkpt_dir)
        torch.save(
            {
                'epoch': epoch+1,
                'optimizer': opt.state_dict(),
                'best_val_result': best_val_result,
                'best_val_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
            },
            training_path
        )
        print("EPOCH %d EVAL: " % epoch)
        print("\tCURRENT VAL MSE : %f" % tuning_loss)
        print("\tCURRENT VAL LOSS: %f" % total_nll)
        print("\tBEST VAL MSE:     %f" % best_val_result)
        print("\tBEST VAL LOSS:    %f" % best_val_loss)
        print("\tBEST VAL EPOCH:   %d" % best_val_epoch)
        end = time.time()
