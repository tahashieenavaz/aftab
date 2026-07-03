import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableGELU(nn.Module):
    def __init__(self, in_channels: int, dim: int = -1, min_lambda: float = 1e-4):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.min_lambda = min_lambda
        self.lambdas = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        shape = [1] * x.ndim
        shape[self.dim] = self.in_channels
        lam = self.lambdas.view(*shape)
        lam = torch.clamp(lam, min=self.min_lambda)
        return F.gelu(x * lam) / lam
