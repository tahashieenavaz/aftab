import torch


def gompertz(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.exp(-x))


class GoLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * gompertz(x)
