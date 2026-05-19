import jax
import flax.linen as nn


class Conv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int | str = "SAME"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(self.out_channels, self.kernel_size, self.stride, self.padding)(
            x
        )
