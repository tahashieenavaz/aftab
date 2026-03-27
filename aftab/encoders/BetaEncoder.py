import torch
from ..common import LayerNorm2d, ModuleType


class BetaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=6, stride=2, padding=2),
            LayerNorm2d(32),
            activation(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(64),
            activation(),
            torch.nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(32),
            activation(),
            torch.nn.Conv2d(32, 16, kernel_size=8, stride=1, padding=0),
            LayerNorm2d(16),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
