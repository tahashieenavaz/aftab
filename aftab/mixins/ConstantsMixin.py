import os
from baloot import acceleration_device


class ConstantsMixin:
    def __init__(self):
        super().__init__()
        self.device = acceleration_device()
        self.cpu_count = os.cpu_count() or 1
