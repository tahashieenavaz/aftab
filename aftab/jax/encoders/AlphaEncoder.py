import jax
import flax.linen as nn
from typing import Callable, List
from aftab.jax.modules import EncoderBlock


class AlphaEncoder(nn.Module):
    activation: Callable = nn.relu
    configuration: List[List] = [
        [4, 32, 4, 2, 1],
        [32, 64, 4, 2, 1],
        [64, 64, 3, 2, 1],
        [64, 64, 5, 0, 1],
    ]

    @nn.compact
    def __call__(self, x: jax.Array):
        for in_channels, out_channels, kernel, stride, padding in self.configuration:
            x = EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
                activation=self.activation,
            )(x)
        return x
