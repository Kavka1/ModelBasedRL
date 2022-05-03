from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def call_activation(name: str) -> nn.Module:
    if name == 'Identity':
        return nn.Identity
    elif name == 'ReLU':
        return nn.ReLU
    elif name == 'Tanh':
        return nn.Tanh
    elif name == 'Sigmoid':
        return nn.Sigmoid
    elif name == 'SoftMax':
        return nn.Softmax
    elif name == 'ELU':
        return nn.ELU
    elif name == 'LeakyReLU':
        return nn.LeakyReLU
    else:
        raise NotImplementedError(f"Invalid activation name: {name}")


def call_mlp(
    in_dim: int, 
    out_dim: int, 
    hidden_layers: List[int],
    inter_activation: str = 'ReLU',
    output_activation: str = 'Tanh'
) -> nn.Module:
    module_seq = []
    InterActivation = call_activation(inter_activation)
    OutActivation = call_activation(output_activation)
    
    last_dim = in_dim
    for hidden in hidden_layers:
        module_seq += [nn.Linear(last_dim, hidden), InterActivation()]
        last_dim = hidden
    module_seq += [nn.Linear(last_dim, out_dim), OutActivation()]

    return nn.Sequential(*module_seq)