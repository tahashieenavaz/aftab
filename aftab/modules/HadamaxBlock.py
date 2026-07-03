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
        self.normalization = HadamaxLayerNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )
        self.chi = chi(out_channels) if chi is MixedActivation else chi()
        self.psi = psi(out_channels) if psi is MixedActivation else psi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixed:
            padding_size = self.convolutional_a.kernel_size[0] // 2
            w_b_padded = torch.nn.functional.pad(
                self.convolutional_b.weight,
                (padding_size, padding_size, padding_size, padding_size),
            )
            fused_weight = torch.cat([self.convolutional_a.weight, w_b_padded], dim=0)
            x = torch.nn.functional.conv2d(
                x,
                weight=fused_weight,
                stride=self.convolutional_a.stride,
                padding=self.convolutional_a.padding,
                bias=None,
            )
        else:
            x = self.convolutional(x)

        x = self.normalization(x)
        a, b = torch.chunk(x, 2, dim=1)
        a, b = self.chi(a), self.psi(b)
        return self.pool(a * b)
