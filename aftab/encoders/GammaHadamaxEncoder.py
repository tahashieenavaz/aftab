import torch
from ..common import LayerNorm2d, ModuleType


class HadamaxLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        pool_k,
        pool_s,
        pool_p=0,
        activation=torch.nn.GELU,
    ):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.conv_b = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.norm_a = LayerNorm2d(out_channels)
        self.norm_b = LayerNorm2d(out_channels)

        self.activation = activation()

        self.pool_k = pool_k
        self.pool_s = pool_s
        self.pool_p = pool_p

    def forward(self, x):
        a = self.activation(self.norm_a(self.conv_a(x)))
        b = self.activation(self.norm_b(self.conv_b(x)))
        x = a * b
        return torch.nn.functional.max_pool2d(
            x, kernel_size=self.pool_k, stride=self.pool_s, padding=self.pool_p
        )


class GammaHadamaxFeatureExtractor(torch.nn.Module):
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
                padding=0,
                pool_k=3,
                pool_s=1,
                pool_p=1,
                activation=activation,
            ),
            HadamaxLayer(
                64,
                64,
                kernel_size=3,
                padding=0,
                pool_k=2,
                pool_s=2,
                activation=activation,
            ),
            HadamaxLayer(
                64,
                64,
                kernel_size=3,
                padding=0,
                pool_k=3,
                pool_s=1,
                pool_p=1,
                activation=activation,
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
