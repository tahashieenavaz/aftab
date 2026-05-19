import jax
import flax.linen as nn


class Linear(nn.Module):
    input_dimension: int
    output_dimension: int

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(self.output_dimension)(x)
