import torch
from baloot import acceleration_device
from typing import Type
from .maps import AftabMapEncoder


class Aftab:
    def __init__(
        self,
        encoder: str | Type[torch.nn.Module] = "gamma",
        frameskip: int = 4,
        num_minibatches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.65,
        lr: float = 0.00025,
        logging_interval: int = 10,
    ):
        self.device = acceleration_device()
        self.frameskip = frameskip
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.num_minibatches = num_minibatches
        self.logging_interval = logging_interval
        self.encoder = encoder

        if isinstance(encoder, str):
            module = AftabMapEncoder.get(encoder)
            self.encoder = module

    def train(self, environment):
        torch.set_float32_matmul_precision("high")
