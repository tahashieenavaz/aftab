import torch
from aftab.modules import HadamaxBlock, LearnableGELU
from aftab.constants import ModuleType


class HadamaxGammaEncoderV1Mixed(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = LearnableGELU, in_channels: int = 4):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxBlock(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
                mixed=True,
            ),
            HadamaxBlock(
                32,
                48,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
                mixed=True,
            ),
            HadamaxBlock(
                48,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=3,
                pool_stride=1,
                pool_padding=0,
                chi=activation,
                psi=activation,
                mixed=True,
            ),
            HadamaxBlock(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
                mixed=True,
            ),
            HadamaxBlock(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=3,
                pool_stride=1,
                pool_padding=0,
                chi=activation,
                psi=activation,
                mixed=True,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
