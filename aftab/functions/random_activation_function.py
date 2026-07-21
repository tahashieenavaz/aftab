import random
from aftab.constants import stream_activation_pool
from aftab.typing import ModuleType


def random_activation_function() -> ModuleType:
    return random.choice(stream_activation_pool)
