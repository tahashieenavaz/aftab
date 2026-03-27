import torch
from ..common import LayerNorm2d
from typing import Type


class BlockA(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module]):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.normalization = LayerNorm2d(32)
        self.activation = activation()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class BlockB(torch.nn.Module):
    def __init__(self, activation=Type[torch.nn.Module]):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.normalization = LayerNorm2d(48)
        self.activation = activation()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class BlockC(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module]):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.normalization = LayerNorm2d(64)
        self.activation = activation()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class BlockD(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module]):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.normalization = LayerNorm2d(64)
        self.activation = activation()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class BlockE(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module]):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.normalization = LayerNorm2d(64)
        self.activation = activation()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class HadamaxGammaEncoder(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        super().__init__()

        self.a = BlockA(activation=activation)
        self.a_prime = BlockA(activation=activation)

        self.b = BlockB(activation=activation)
        self.b_prime = BlockB(activation=activation)

        self.c = BlockC(activation=activation)
        self.c_prime = BlockC(activation=activation)

        self.d = BlockD(activation=activation)
        self.d_prime = BlockD(activation=activation)

        self.e = BlockE(activation=activation)
        self.e_prime = BlockE(activation=activation)

    def forward(self, x):
        a = self.a(x)
        a_prime = self.a_prime(x)
        x = a * a_prime
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        b = self.b(x)
        b_prime = self.b_prime(x)
        x = b * b_prime
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        c = self.c(x)
        c_prime = self.c_prime(x)
        x = c * c_prime
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        d = self.d(x)
        d_prime = self.d_prime(x)
        x = d * d_prime
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        e = self.e(x)
        e_prime = self.e_prime(x)
        x = e * e_prime
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x.flatten(start_dim=1)
