import torch
import torch.nn as nn

from .models.mlp import *
from .models.cnn import *
from .models.fno.FNO import *

def load_frozen(config:(dict), statedict_path:(str), model_type:(str)) -> nn.Module:
    """
    Initializes model, loads lab trained weights, and freezes feature extractor layers.

    Args:
        config (dict): The config dict for the model
        statedict_path (str): The path for the statedict of the saved model
        model_type (str): The str, specifying the model to be loaded, one of "MLP", "CNN", "FNO"

    Returns
        model (nn.Module): The laoded, and frozen model
    """

    if model_type == 'MLP':
        model = MLP(**config)
    elif model_type == 'CNN':
        model = CNN(**config)
    elif model_type == 'FNO':
        model = FNO_nD(**config)    
    else:
        raise ValueError(f"Unknown/Unsupported model_type: {model_type}")
    
    # Loads the statedict
    model.load_state_dict(torch.load(statedict_path, map_location='cpu'))

    # Freezes the whole thing
    for param in model.parameters():
        param.requires_grad = False
    
    # Depending on the model, unfreezes

    # We unfreeze last 9 components, which means 4 components per 2 last layers, and also 1 output layer
    if model_type == 'MLP':
        for module in list(model.layers.children())[-9:]:
            for param in module.parameters():
                param.requires_grad = True

    # We unfreeze the final nn.Linear module in the sequential container
    elif model_type == 'CNN':
        for param in model.layers[-1].parameters():
            param.requires_grad = True

    # we unfreeze only the Q network. This is because all the Fourier blocks and P network are feature extractors
    elif model_type == 'FNO':
        for param in model.Q_Net.parameters():
            param.requires_grad = True
        for param in model.final_proj.parameters():
            param.requires_grad = True

    return model

