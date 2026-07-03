import torch
from aftab.constants import ModuleType
from .HadamaxLayerNorm2d import HadamaxLayerNorm2d
from .MixedActivation import MixedActivation


class HadamaxBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        pool_kernel: int,
        pool_stride: int,
        pool_padding: int = 0,
        chi: ModuleType = torch.nn.GELU,
        psi: ModuleType = torch.nn.GELU,
    ):
        super().__init__()

        self.convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.normalization = HadamaxLayerNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )
        self.chi = chi(out_channels) if chi is MixedActivation else chi()
        self.psi = psi(out_channels) if psi is MixedActivation else psi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(self.convolutional(x))
        a, b = torch.chunk(x, 2, dim=1)
        a, b = self.chi(a), self.psi(b)
        return self.pool(a * b)
