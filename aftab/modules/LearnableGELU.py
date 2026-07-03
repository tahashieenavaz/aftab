import torch
import math


class LearnableGELU(torch.nn.Module):
    def __init__(self, in_channels: int, dim: int = -1):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.lambdas = torch.nn.Parameter(torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.ndim
        shape[self.dim] = self.in_channels
        lam = self.lambdas.view(*shape)
        z = lam * x
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return x * cdf
