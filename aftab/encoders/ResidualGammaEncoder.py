import torch
from ..common import LayerNorm2d, ModuleType


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        first_convolutional: torch.nn.Module,
        first_normalization: torch.nn.Module,
        first_activation: torch.nn.Module,
        second_convolutional: torch.nn.Module,
        second_normalization: torch.nn.Module,
    ):
        super().__init__()
        self.first_convolutional = first_convolutional
        self.first_normalization = first_normalization
        self.first_activation = first_activation
        self.second_convolutional = second_convolutional
        self.second_normalization = second_normalization

        if in_channels != out_channels or stride != 1:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                LayerNorm2d(out_channels),
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.first_convolutional(x)
        out = self.first_normalization(out)
        out = self.first_activation(out)

        out = self.second_convolutional(out)
        out = self.second_normalization(out)

        if identity.shape[2:] != out.shape[2:]:
            diff_h = identity.shape[2] - out.shape[2]
            diff_w = identity.shape[3] - out.shape[3]
            top = diff_h // 2
            bottom = identity.shape[2] - (diff_h - top)
            left = diff_w // 2
            right = identity.shape[3] - (diff_w - left)
            identity = identity[:, :, top:bottom, left:right]

        out += identity
        return out


class GammaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.alef = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm2d(32),
            activation(),
        )

        self.be = ResBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
            first_convolutional=torch.nn.Conv2d(
                32, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            first_normalization=LayerNorm2d(48),
            first_activation=activation(),
            second_convolutional=torch.nn.Conv2d(
                48, 64, kernel_size=3, stride=1, padding=0, bias=False
            ),
            second_normalization=LayerNorm2d(64),
        )
        self.be_activation = activation()

        self.pe = ResBlock(
            in_channels=64,
            out_channels=64,
            stride=2,
            first_convolutional=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=2, padding=0, bias=False
            ),
            first_normalization=LayerNorm2d(64),
            first_activation=activation(),
            second_convolutional=torch.nn.Conv2d(
                64, 64, kernel_size=3, stride=1, padding=0, bias=False
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
