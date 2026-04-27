import torch
from ..constants import ModuleType
from .LayerNorm2d import LayerNorm2d


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: ModuleType = torch.nn.ReLU,
    ):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = activation()
        self.normalization = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x
