import torch
from .ResidualGammaBlock import ResidualGammaBlock
from ..modules import LayerNorm2d
from ..constants import ModuleType


class ResidualGammaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.alef = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(32),
            activation(),
        )

        self.be = ResidualGammaBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
            first_convolutional=torch.nn.Conv2d(
                32, 48, kernel_size=3, stride=2, padding=1
            ),
            first_normalization=LayerNorm2d(48),
            first_activation=activation(),
            second_convolutional=torch.nn.Conv2d(
                48, 64, kernel_size=3, stride=1, padding=0
            ),
            second_normalization=LayerNorm2d(64),
        )
        self.be_activation = activation()

        self.pe = ResidualGammaBlock(
            in_channels=64,
            out_channels=64,
            stride=2,
            first_convolutional=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=2, padding=0
            ),
            first_normalization=LayerNorm2d(64),
            first_activation=activation(),
            second_convolutional=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=0
            ),
            second_normalization=LayerNorm2d(64),
        )
        self.pe_activation = activation()

    def forward(self, x):
        x = self.alef(x)
        x = self.be(x)
        x = self.be_activation(x)
        x = self.pe(x)
        x = self.pe_activation(x)
        return torch.flatten(x, 1)
