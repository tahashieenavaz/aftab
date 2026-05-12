import torch
from aftab.constants import ModuleType
from .HadamaxLayerNorm2d import HadamaxLayerNorm2d


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
        )
        self.normalization = HadamaxLayerNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )

        self.same_activations = chi == psi
        self.chi = chi()
        self.psi = psi() if not self.same_activations else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(self.convolutional(x))
        if self.same_activations:
            a, b = torch.chunk(self.chi(x), 2, dim=1)
        else:
            a, b = torch.chunk(x, 2, dim=1)
            a, b = self.chi(a), self.psi(b)
        return self.pool(a * b)
