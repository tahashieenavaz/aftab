import jax
import flax.linen as nn
from typing import Callable, List
from aftab.jax.modules import EncoderBlock


class BaseEncoder(nn.Module):
    activation: Callable
    configuration: List[List]

    def setup(self):
        self.stream = [
            EncoderBlock(
                output_channels=output_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
            )
            for output_channels, kernel, stride, padding in self.configuration
        ]

    @nn.compact
    def __call__(self, x: jax.Array):
        batch_size = x.shape[0]
        for block in self.stream:
            x = block(x)
        return x.reshape(batch_size, -1)
