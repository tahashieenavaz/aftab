import torch
from typing import Type
from ..modules import LayerNorm2d


class ResidualGammaBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        first_convolutional: Type[torch.nn.Module],
        first_normalization: Type[torch.nn.Module],
        first_activation: Type[torch.nn.Module],
        second_convolutional: Type[torch.nn.Module],
        second_normalization: Type[torch.nn.Module],
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
