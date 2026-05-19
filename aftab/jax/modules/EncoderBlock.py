import jax
import flax.linen as nn
from typing import Callable
from .LayerNorm2d import LayerNorm2d


class EncoderBlock(nn.Module):
    in_channels: int
    out_channels: int
    kernel: int
    stride: int
    padding: int | str = "SAME"
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel, self.kernel),
            strides=(self.stride, self.stride),
            padding=self.padding,
        )(x)
        x = LayerNorm2d()(x)
        x = self.activation(x)
        return x
