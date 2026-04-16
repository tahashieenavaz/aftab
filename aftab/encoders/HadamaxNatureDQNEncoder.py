import torch
from ..modules import HadamaxBlock


class HadamaxNatureDQNEncoder(torch.nn.Module):
    def __init__(self, *, activation=torch.nn.GELU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxBlock(
                4,
                32,
                kernel_size=8,
                stride=1,
                padding=4,
                pool_kernel=4,
                pool_stride=4,
                chi=activation,
                psi=activation,
            ),
            HadamaxBlock(
                32,
                64,
                kernel_size=4,
                stride=1,
                padding=2,
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
