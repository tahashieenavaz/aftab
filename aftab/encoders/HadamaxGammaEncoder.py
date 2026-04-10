import torch
from .HadamaxLayer import HadamaxLayer


class HadamaxGammaEncoder(torch.nn.Module):
    def __init__(self, activation=torch.nn.GELU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxLayer(
                4,
                32,
                kernel_size=3,
                padding=1,
                pool_k=2,
                pool_s=2,
                activation=activation,
            ),
            HadamaxLayer(
                32,
                48,
                kernel_size=3,
                padding=1,
                pool_k=2,
                pool_s=2,
                activation=activation,
            ),
            HadamaxLayer(
                48,
                64,
                kernel_size=3,
                padding=1,
                pool_k=3,
                pool_s=1,
                pool_p=0,
                activation=activation,
            ),
            HadamaxLayer(
                64,
                64,
                kernel_size=3,
                padding=1,
                pool_k=2,
                pool_s=2,
                activation=activation,
            ),
            HadamaxLayer(
                64,
                64,
                kernel_size=3,
                padding=1,
                pool_k=3,
                pool_s=1,
                pool_p=0,
                activation=activation,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
