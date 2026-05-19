import jax
import flax.linen as nn


class Stream(nn.Module):
    hidden_dimension: int
    output_dimension: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dimension)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dimension)(x)
        return x
