import torch
from aftab.modules import HadamaxBlock
from aftab.constants import ModuleType


class HadamaxZetaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.GELU, in_channels: int = 4):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxBlock(
                in_channels,
                48,
                kernel_size=4,
                stride=1,
                padding=2,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
            ),
            HadamaxBlock(
                48,
                48,
                kernel_size=4,
                stride=1,
                padding=2,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
            ),
            HadamaxBlock(
                48,
                48,
                kernel_size=4,
                stride=1,
                padding=2,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
