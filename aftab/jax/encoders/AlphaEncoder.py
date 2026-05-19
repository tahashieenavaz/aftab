from typing import Callable
from .BaseEncoder import BaseEncoder


def AlphaEncoder(activation: Callable):
    return BaseEncoder(
        configuration=[
            [4, 32, 4, 2, 1],
            [32, 64, 4, 2, 1],
            [64, 64, 3, 2, 1],
            [64, 64, 5, 0, 1],
        ],
        activation=activation,
    )
