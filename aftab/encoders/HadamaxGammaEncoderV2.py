import torch
from ..modules import HadamaxBlock


class HadamaxGammaEncoderV2(torch.nn.Module):
    def __init__(self, *, activation=torch.nn.GELU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxBlock(
                4,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=2,
                pool_stride=2,
                chi=activation,
                psi=activation,
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
            ),
            HadamaxBlock(
                48,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_kernel=3,
                pool_stride=1,
                pool_padding=1,
                chi=activation,
                psi=activation,
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
            ),
            HadamaxBlock(
                64,
                64,
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

    def forward(self, x):
        return self.stream(x)
