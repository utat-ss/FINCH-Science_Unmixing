import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import copy

from src.dl.metric import get_n_params, get_flops, get_r2, get_mae
from src.dl.plotting import plot_preds, plot_avg_losses, plot_metrics, plot_train_losses, pareto_plot

def train_model(
    model:(nn.Module), 
    loss_fn:(nn.Module), 
    optimizer:(optim.Optimizer), 
    lr_scheduler:(optim.lr_scheduler.LRScheduler), 
    configured_data:(list[iter]), 
    n_epoch:(int), 
    n_tb_epoch:(int), 
    device:(torch.device), 
    dtype:(torch.dtype), 
    save_dir:(str),
    model_name:(str),
    save_artifacts:(bool)=True,
    fine_tuning:(bool)=False
):
    """
    This is the function which trains the critic model on ksi_train (synthesized from psi_1), then validates and tests on ksi_val and ksi_test (parts of psi_2).
    """
    # Create the arrays to log stuff
    model_losses_step = np.zeros(shape=(n_epoch*n_tb_epoch, 1)) # (step, [train_loss])
    model_losses_average = np.zeros(shape=(n_epoch, 2)) # (epoch, [avg_train_loss, val_loss])
    model_metrics_epoch = np.zeros(shape=(n_epoch, 9)) # (epoch, [val_loss, val_r2_total, val_r2_gv, val_r2_npv, val_r2_soil, val_mae_total, val_mae_gv, val_mae_npv, val_mae_soil])
    model_metrics_total = np.zeros(shape=(1, 11)) # [param, flops, test_loss, test_r2_total, test_r2_gv, test_r2_npv, test_r2_soil, test_mae_total, test_mae_gv, test_mae_npv, test_mae_soil]

    # Throw the critic to device
    model.to(device)

    # Unpack the configured data list, get iterators and unnorming function
    iter_train, iter_val, iter_test = configured_data

    # Initial logs and losses
    model_metrics_total[0,0] = get_n_params(model)
    model_metrics_total[0,1] = get_flops(model, i_shape=(4, 80))
    best_val_loss = float('inf')

    if fine_tuning:
        # If fine tuning is enabled, we first forward prop 5 epochs worth of batches to calibrate batchnorm stats, in case it has been used
        model.train()
        for _ in range(n_tb_epoch * 5):
            batch = next(iter_train)
            spectrum = batch['spectrum'].to(device=device, dtype=dtype)
            _ = model(spectrum)
        print(f"Initial fine tuning complete, with {n_tb_epoch*5} batches")

    # Log the param amount of the critic
    counter=0
    for epoch in range(1, n_epoch + 1):

        ### TRAINING STEP ###
        # Set the critic in training mode
        model.train()

        if fine_tuning:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

        for _ in range(n_tb_epoch):
            # Get the batch and unpack it
            batch = next(iter_train)
            spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)
            # Zero the grads
            optimizer.zero_grad()
            # Get the predictions
            pred_abundances = model(spectrum)
            # Calculate the loss, backprop it, and take optimizer step
            train_loss = loss_fn(pred_abundances, abundances)
            train_loss.backward()
            optimizer.step()

            # Log train step results
            model_losses_step[counter] = np.array([train_loss.item()])
            counter += 1
            # Step Scheduler, every step
            lr_scheduler.step()
            print(f"Epoch {epoch}/{n_epoch}, Step {_+1}/{n_tb_epoch}, Train Loss: {train_loss.item():.6f}")

        # Log average train results
        model_losses_average[epoch-1, 0] = np.mean(model_losses_step[(epoch-1)*n_tb_epoch:epoch*n_tb_epoch, 0])

        ### VALIDATION STEP ###
        # Set critic to evaluation
        model.eval()

        with torch.no_grad():
            # Get the batch and unpack it
            batch = next(iter_val)
            spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
            spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

            pred_abundances = model(spectrum)
            val_loss = loss_fn(pred_abundances, abundances)

            #### LOG MAE AND R2 METRICS ####
            r2_array = get_r2(pred_abundances.cpu().numpy(), abundances.cpu().numpy())
            mae_array = get_mae(pred_abundances.cpu().numpy(), abundances.cpu().numpy())
            model_losses_average[epoch-1, 1] = val_loss.item()
            model_metrics_epoch[epoch-1] = np.array([val_loss.item(), r2_array[0], r2_array[1], r2_array[2], r2_array[3], mae_array[0], mae_array[1], mae_array[2], mae_array[3]])

        # Save the critic if it is the best as per loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch}/{n_epoch} complete. Train Loss: {model_losses_average[epoch-1,0]:.6f}, Val Loss: {model_losses_average[epoch-1,1]:.6f}")


    torch.save(best_model_state, rf'{save_dir}\model_weights.pth')
    print("Training complete, starting testing")
    ### TESTING STEP ###
    # Load the best model
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        # Get the batch and unpack it
        batch = next(iter_test)
        spectrum, abundances, name, orig_index = batch['spectrum'], batch['abundances'], batch['names'], batch['orig_index']
        spectrum = spectrum.to(device=device, dtype=dtype); abundances = abundances.to(device=device, dtype=dtype)

        pred_abundances = model(spectrum)
        test_loss = loss_fn(pred_abundances, abundances)

        # Log metrics from testing
        r2_array = get_r2(pred_abundances.cpu().numpy(), abundances.cpu().numpy())
        mae_array = get_mae(pred_abundances.cpu().numpy(), abundances.cpu().numpy())
        model_metrics_total[0,2:] = np.array([test_loss.item(), r2_array[0], r2_array[1], r2_array[2], r2_array[3], mae_array[0], mae_array[1], mae_array[2], mae_array[3]])
   
    if save_artifacts:
        plot_preds(
            abundances.cpu().numpy(),
            pred_abundances.cpu().numpy(), 
            save_path=rf'{save_dir}\abundance_plot.svg',
            model_name=model_name,
        )
        plot_metrics(
            list(range(1, epoch + 1)),
            model_metrics_epoch[:,:5],
            metric_names=['Validation Loss', 'Total R2', 'GV R2', 'NPV R2', 'Soil R2'],
            save_path=rf'{save_dir}\metrics.svg',
            model_name=model_name
        )
        plot_avg_losses(
            list(range(1, epoch + 1)),
            model_losses_average, 
            rf'{save_dir}\avg_losses.svg', 
            model_name=model_name
        )
        # Save the test abundance array of shape (n_test_samples, 6)
        np.savez_compressed(
            rf'{save_dir}\test_abundances',
            pred_abundances=pred_abundances.cpu().numpy(),
            true_abundances=abundances.cpu().numpy(),
            orig_index=orig_index
        )
    print("Testing complete")

    model_dict = {'losses_step': model_losses_step, 'losses_average': model_losses_average, 'metrics_epoch': model_metrics_epoch, 'metrics_total': model_metrics_total}

    return model_dict