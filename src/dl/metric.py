import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import r2_score
import numpy as np

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_flops(model:(nn.Module), i_shape:(torch.Tensor)):
    """
    Gets the flops of a model

    Args:
        model (nn.Module): The model of interest
        i_shape (torch.Tensor): A tensor with the input shape that matches with the model (B, ch, seq) or (B, seq)

    Returns:
        flops_per_sample (int): The number of flops per sample
    """
    flop_counter = FlopCounterMode(display=False)
    flop_tensor = torch.ones(size=i_shape, dtype=torch.float32, device=next(model.parameters()).device)
    with flop_counter:
        model(flop_tensor)
    flops_per_sample = flop_counter.get_total_flops()
    return flops_per_sample/i_shape[0]

def get_r2(ab_true:(np.ndarray), ab_pred:(np.ndarray)) -> np.ndarray:
    """
    Takes in the true and predicted abundances, returns a list of R^2 related stuff and linreg related stuff.
    Both ab_true and ab_pred must be detached

    Returns:
        r2_metrics (np.ndarray): In the form [total, gv, npv, soil]
    """
    individual = r2_score(ab_true, ab_pred, multioutput='raw_values')
    total = np.mean(individual)
    r2_metrics = np.array([total, individual[0], individual[1], individual[2]])
    return r2_metrics

def get_mae(ab_true:(np.ndarray), ab_pred:(np.ndarray)) -> np.ndarray:
    """
    Takes in the true and predicted abundances, returns a list of MAE related stuff.
    Both ab_true and ab_pred must be detached

    Returns:
        mae_metrics (np.ndarray): In the form [total, gv, npv, soil]
    """
    individual = np.mean(np.abs(ab_true - ab_pred), axis=0)
    total = np.mean(individual)
    mae_metrics = np.array([total, individual[0], individual[1], individual[2]])
    return mae_metrics
