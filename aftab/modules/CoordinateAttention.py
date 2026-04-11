import torch
from typing import Type


class CoordinateAttention(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        reduction: int = 32,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.pool_h = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = torch.nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, in_dim // reduction)
        self.conv1 = torch.nn.Conv2d(
            in_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = torch.nn.BatchNorm2d(hidden_dim)
        self.activation = activation()
        self.conv_h = torch.nn.Conv2d(
            hidden_dim, out_dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_w = torch.nn.Conv2d(
            hidden_dim, out_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(-1, -2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.activation(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out
