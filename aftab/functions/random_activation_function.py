import random
import torch
from aftab.typing import ModuleType

stream_activation_pool: list[ModuleType] = [
    torch.nn.ReLU,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Mish,
]


def random_activation_function() -> ModuleType:
    return random.choice(stream_activation_pool)
