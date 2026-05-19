import jax
import flax.linen as nn
from aftab.jax.modules import Conv2d


class NatureDQNEncoderJAX(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        x = Conv2d(4, 32, 8, 4)(x)
        x = nn.relu(x)
        x = Conv2d(32, 64, 4, 2)(x)
        x = nn.relu(x)
        x = Conv2d(64, 64, 3, 1)(x)
        x = nn.relu(x)
        return x
