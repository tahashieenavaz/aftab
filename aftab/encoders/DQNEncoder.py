import torch
from ..common import LayerNorm2d


class DQNEncoder(torch.nn.Module):
    def __init__(self, *, activation=torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            LayerNorm2d(32),
            activation(),
            torch.nn.Conv2d(32, 64, 4, 2),
            LayerNorm2d(64),
            activation(),
            torch.nn.Conv2d(64, 64, 3, 1),
            LayerNorm2d(64),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
