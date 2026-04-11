import torch
from ..modules import LayerNorm2d, EfficientMultiScaleAttention
from ..constants import ModuleType


class GammaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(32),
            activation(),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(48),
            activation(),
            EfficientMultiScaleAttention(48),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            LayerNorm2d(64),
            activation(),
            EfficientMultiScaleAttention(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            LayerNorm2d(64),
            activation(),
            EfficientMultiScaleAttention(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            LayerNorm2d(64),
            activation(),
            EfficientMultiScaleAttention(64),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
