import torch
import numpy as np
from baloot import acceleration_device
from typing import Type


class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target=0.001):
        self.top = 1.0
        self.target = target
        self.ratio = ratio

    def get(self, frames, total_frames, all_rewards, episode_returns):
        target = self.target
        top = self.top
        decay_duration = total_frames * self.ratio
        if decay_duration == 0:
            return top
        return max(target, top - (frames / decay_duration) * (top - target))


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        # this is just here to keep the consistency. it doesn't do anything in this block.
        input_dim: int = 3136,
        hidden_dim: int = 512,
        output_dim,
        activation: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.stream(features)


def mse_loss(A, B):
    _device = acceleration_device()
    A = A.to(_device)
    B = B.to(_device)
    return 0.5 * torch.nn.functional.mse_loss(A, B)
