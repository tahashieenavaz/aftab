import torch
import numpy
import os
import math
from baloot import acceleration_device, seed_everything
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
        num_train_environments: int = 128,
        num_test_environments: int = 8,
        steps_per_update: int = 32,
        total_frames: int = 200_000_000,
        seed: int = 42,
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
        self.num_train_environments = num_train_environments
        self.num_test_environments = num_test_environments
        self.total_environments = int(num_train_environments + num_test_environments)
        self.cpu_count = os.cpu_count()
        self.steps_per_update = steps_per_update
        self.batch_size = int(num_train_environments * steps_per_update)
        self.actual_frames = int(total_frames / frameskip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)
        self.seed = seed

        if isinstance(encoder, str):
            module = AftabMapEncoder.get(encoder)
            self.encoder = module

    def train(self, environment):
        seed_everything(self.seed)
        torch.set_float32_matmul_precision("high")
        all_train_rewards = []
        all_test_rewards = []
        all_loss = []
        episode_returns = numpy.zeros(self.total_envs, dtype=numpy.float32)
