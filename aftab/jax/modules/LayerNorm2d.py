import jax
import flax.linen as nn


class LayerNorm2d(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.LayerNorm(
            epsilon=self.epsilon,
            reduction_axes=-1,
            feature_axes=-1,
        )(x)
