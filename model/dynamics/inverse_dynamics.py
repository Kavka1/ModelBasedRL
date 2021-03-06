from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ModelBasedRL.model.utils import call_mlp


class DeterministicIDM(nn.Module):
    """
    Inverse Dynamics Model with deterministic action output
    """
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int]) -> None:
        super(DeterministicIDM, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers

        self.in_dim = 2 * self.o_dim
        self.out_dim = self.a_dim
        self.model = call_mlp(
            self.in_dim, 
            self.out_dim, 
            self.hidden_layers, 
            inter_activation= 'ReLU',
            output_activation= 'Tanh'
        )

    def __call__(self, obs: torch.tensor, obs_: torch.tensor) -> torch.tensor:
        x = torch.concat([obs, obs_], dim=-1)
        return self.model(x)


class DiagGaussianIDM(nn.Module):
    """
    Inverse Dynamics Model with Diagnose Gaussian action distribution output
    """
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int], logstd_min: float, logstd_max: float) -> None:
        super(DiagGaussianIDM, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers

        self.in_dim = 2 * self.o_dim
        self.out_dim = self.a_dim * 2
        self.model = call_mlp(
            self.in_dim,
            self.out_dim,
            self.hidden_layers,
            inter_activation= 'ReLU',
            output_activation= 'Identity'
        )
        self.logstd_min = nn.Parameter(torch.ones(size=(self.a_dim,)).float() * logstd_min, requires_grad=True) 
        self.logstd_max = nn.Parameter(torch.ones(size=(self.a_dim,)).float() * logstd_max, requires_grad=True)

    def __call__(self, obs: torch.tensor, obs_: torch.tensor) -> torch.distributions.Distribution:
        x = torch.concat([obs, obs_], dim=-1)
        x = self.model(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)

        mean = F.tanh(mean)
        logstd = self.logstd_max - F.softplus(self.logstd_max - logstd)
        logstd = self.logstd_min + F.softplus(logstd - self.logstd_min)

        return Normal(mean, torch.exp(logstd))