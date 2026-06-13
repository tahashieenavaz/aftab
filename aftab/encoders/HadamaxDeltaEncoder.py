import torch
from aftab.modules import HadamaxBlock
from aftab.constants import ModuleType


class HadamaxDeltaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.GELU, in_channels: int = 4):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxBlock(
                in_channels,
                24,
                kernel_size=9,
                stride=1,
                padding=4,
                pool_kernel=4,
                pool_stride=4,
                chi=activation,
                psi=activation,
            ),
            HadamaxBlock(
                24,
                48,
                kernel_size=5,
                stride=1,
                padding=2,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
            ),
            HadamaxBlock(
                48,
                96,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=3,
                pool_stride=1,
                pool_padding=1,
                chi=activation,
                psi=activation,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
