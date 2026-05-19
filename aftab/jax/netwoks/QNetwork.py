import jax
import flax.linen as nn
from aftab.jax.modules import Stream


class QNetowrk(nn.Module):
    encoder: nn.Module
    action_dimension: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        features = self.encoder(x)
        return Stream(512, self.action_dimension)(features)
