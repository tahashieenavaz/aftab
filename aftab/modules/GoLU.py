import torch
from aftab.functions import gompertz


class GoLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * gompertz(x)
