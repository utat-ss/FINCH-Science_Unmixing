import torch
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

def get_r2(ab_true:(torch.Tensor), ab_pred:(torch.Tensor), mode:(str)) -> dict:
    """
    Takes in the true and predicted abundances, returns a list of R^2 related stuff and linreg related stuff.
    Both ab_true and ab_pred must be detached

    Returns:
        r2_metrics (dict): The r2 values in keys 'total', 'gv', 'npv', 'soil'
    """
    if mode not in ['val', 'test']:
        raise ValueError(f"Unknown/Unsupported r2 mode: {mode}")
    em_list = [f'general/{mode}_gv_r2', f'general/{mode}_npv_r2', f'general/{mode}_soil_r2']

    ab_true = ab_true.cpu().numpy()
    ab_pred = ab_pred.cpu().numpy()
    temp = r2_score(ab_true, ab_pred, multioutput='raw_values').tolist()

    r2_metrics = {f'general/{mode}_total_r2': np.mean(temp)}
    for i, j in zip(em_list, temp):
        r2_metrics[i] = j

    return r2_metrics
