import flax.linen as nn
from typing import Callable
from .BaseEncoder import BaseEncoder


def BetaEncoder(activation: Callable = nn.relu):
    return BaseEncoder(
        configuration=[
            [4, 32, 6, 2, 2],
            [32, 64, 3, 1, 1],
            [64, 32, 4, 2, 1],
            [32, 16, 8, 1, 0],
        ],
        activation=activation,
    )
