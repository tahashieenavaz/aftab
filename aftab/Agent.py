import torch
from baloot import acceleration_device


class Agent:
    def __init__(self):
        self.device = acceleration_device()
        torch.set_float32_matmul_precision("high")

    def train(self):
        pass
