import torch
import torch.nn.functional as F
from aftab.constants import ModuleType
from .HadamaxLayerNorm2d import HadamaxLayerNorm2d
from .LearnableGELU import LearnableGELU


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

        if self.mixed:
            self.avg_pool = torch.nn.AvgPool2d(
                kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
            )
            self.pool_mix = torch.nn.Parameter(torch.tensor(0.0))
            self.cross_talk = torch.nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.chi = chi(out_channels, dim=1) if chi is LearnableGELU else chi()
        self.psi = psi(out_channels, dim=1) if psi is LearnableGELU else psi()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(self.convolutional(x))
        if self.mixed:
            B, C, H, W = x.shape
            x = x.view(B, 2, C // 2, H, W).transpose(1, 2).contiguous().view(B, C, H, W)
        adam, eve = x.chunk(2, dim=1)
        if self.mixed:
            adam = adam + self.cross_talk * eve
        gated = self.chi(adam).mul_(self.psi(eve))
        if self.mixed:
            mix_weight = torch.sigmoid(self.pool_mix)
            return mix_weight * self.pool(gated) + (1.0 - mix_weight) * self.avg_pool(
                gated
            )
        else:
            return self.pool(gated)
