import torch
from typing import Tuple


class BatchMixin:
    def __init__(self):
        super().__init__()

    def make_batches(self, observation_shape: Tuple, action_dimension: int):
        batch_observations = torch.empty(
            (self.steps_per_update, self.total_environments) + observation_shape,
            dtype=torch.uint8,
            device=self.device,
        )
        batch_actions = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.int64,
            device=self.device,
        )
        batch_rewards = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        batch_terminations = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        batch_q = torch.empty(
            (self.steps_per_update, self.total_environments, action_dimension),
            dtype=torch.float32,
            device=self.device,
        )
        return (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
        )
