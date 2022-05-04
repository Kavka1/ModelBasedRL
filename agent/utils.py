from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import os


def hard_update(source_net: nn.Module, target_net: nn.Module) -> None:
    target_net.load_state_dict(source_net.state_dict())


def soft_update(source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for param, param_tar in zip(source_net.parameters(), target_net.parameters()):
        param_tar.data.copy_(tau * param.data + (1 - tau) * param_tar.data)


def array2tensor(obs: np.array, a: np.array, r: np.array, done: np.array, obs_: np.array, device: torch.device) -> Tuple:
    obs = torch.from_numpy(obs).to(device).float()
    a = torch.from_numpy(a).to(device).float()
    r = torch.from_numpy(r).to(device).float().unsqueeze(dim=-1)
    done = torch.from_numpy(done).to(device).int().unsqueeze(dim=-1)
    obs_ = torch.from_numpy(obs_).to(device).float()
    return obs, a, r, done, obs_


def check_path(path: str) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)