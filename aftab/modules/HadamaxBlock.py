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
        mix: bool = False,
    ):
        super().__init__()
        self.mix = mix
        self.convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.normalization = HadamaxLayerNorm2d(out_channels)
        self.chi = chi()
        self.psi = psi()

        self.temperature = torch.nn.Parameter(torch.ones(1))
        if mix:
            self.mixer = torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=1, bias=False
            )
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(self.convolutional(x))
        adam, eve = x.chunk(2, dim=1)
        gate = self.psi(eve / self.temperature)
        gated = self.chi(adam) * gate
        if self.mix:
            mixed = self.mixer(gated)
            return self.pool(mixed)
        return self.pool(gated)
