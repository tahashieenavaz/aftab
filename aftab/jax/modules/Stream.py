import jax
import flax.linen as nn
from .Linear import Linear


class Stream(nn.Module):
    input_dimension: int
    hidden_dimension: int
    output_dimension: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = Linear(self.input_dimension, self.hidden_dimension)(x)
        x = nn.relu(x)
        x = Linear(self.hidden_dimension, self.output_dimension)(x)
        return x
