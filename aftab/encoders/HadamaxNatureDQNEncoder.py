import torch
from .HadamaxLayer import HadamaxLayer


class HadamaxNatureDQNEncoder(torch.nn.Module):
    def __init__(self, activation=torch.nn.GELU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            HadamaxLayer(
                4,
                32,
                kernel_size=8,
                padding=4,
                stride=1,
                pool_k=4,
                pool_s=4,
                activation=activation,
            ),
            HadamaxLayer(
                32,
                64,
                kernel_size=4,
                stride=1,
                padding=2,
                pool_k=2,
                pool_s=2,
                activation=activation,
            ),
            HadamaxLayer(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                pool_k=3,
                pool_s=1,
                pool_p=1,
                activation=activation,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
