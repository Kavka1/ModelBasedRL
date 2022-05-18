from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ModelBasedRL.model.utils import call_mlp


class DeterministicFDM(nn.Module):
    """
    Forward Dynamics Model with deterministic action output
    """
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int]) -> None:
        super(DeterministicFDM, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers

        self.in_dim = self.o_dim + self.a_dim
        self.out_dim = self.o_dim
        self.model = call_mlp(
            self.in_dim, 
            self.out_dim, 
            self.hidden_layers, 
            inter_activation= 'ReLU',
            output_activation= 'Identity'
        )

    def __call__(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        x = torch.concat([obs, a], dim=-1)
        return self.model(x)


class DiagGaussianFDM(nn.Module):
    """
    Forward Dynamics Model with Diagnose Gaussian action distribution output
    """
    def __init__(self, o_dim: int, a_dim: int, hidden_layers: List[int], logstd_min: float, logstd_max: float) -> None:
        super(DiagGaussianFDM, self).__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.hidden_layers = hidden_layers

        self.in_dim = self.o_dim + self.a_dim
        self.out_dim = self.o_dim * 2
        self.model = call_mlp(
            self.in_dim,
            self.out_dim,
            self.hidden_layers,
            inter_activation= 'ReLU',
            output_activation= 'Identity'
        )
        self.logstd_min = nn.Parameter(torch.ones(size=(self.o_dim,)).float() * logstd_min, requires_grad=True) 
        self.logstd_max = nn.Parameter(torch.ones(size=(self.o_dim,)).float() * logstd_max, requires_grad=True)

    def __call__(self, obs: torch.tensor, a: torch.tensor) -> torch.distributions.Distribution:
        x = torch.concat([obs, a], dim=-1)
        x = self.model(x)
        mean, logstd = torch.chunk(x, 2, dim=-1)

        logstd = self.logstd_max - F.softplus(self.logstd_max - logstd)
        logstd = self.logstd_min + F.softplus(logstd - self.logstd_min)

        return Normal(mean, torch.exp(logstd))