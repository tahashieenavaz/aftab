import torch
from aftab.constants import ModuleType
from .LayerNorm2d import LayerNorm2d
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
        mixed: bool = False,
    ):
        super().__init__()
        self.mixed = mixed

        if self.mixed:
            self.convolutional_a = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.convolutional_b = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.convolutional = torch.nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.normalization = LayerNorm2d(in_channels)
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )
        self.chi = chi(out_channels, dim=1) if chi is MixedActivation else chi()
        self.psi = psi(out_channels, dim=1) if psi is MixedActivation else psi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)
        if self.mixed:
            adam = self.convolutional_a(x)
            eve = self.convolutional_b(x)
        else:
            adam, eve = self.convolutional(x).chunk(2, 1)
        return self.pool(self.chi(adam).mul_(self.psi(eve)))
