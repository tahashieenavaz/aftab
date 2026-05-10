import torch
from typing import Optional
from pathlib import Path


def _make_sure_directory_exists(directory: str):
    directory = directory.replace(".", "/")
    Path(directory).mkdir(exist_ok=True, parents=True)
    return directory


class LinearEpsilon:
    def __init__(self, span: float = 0.1, minimum: float = 0.001, maximum: float = 1.0):
        self.maximum = maximum
        self.minimum = minimum
        self.span = span

    def get(self, frames, total_frames):
        minimum = self.minimum
        maximum = self.maximum
        decay_duration = total_frames * self.span

        if decay_duration == 0:
            return maximum

        delta = maximum - minimum
        return max(minimum, maximum - (frames / decay_duration) * delta)


class RolloutBuffer:
    def __init__(
        self,
        *,
        observation_shape: tuple,
        steps_per_update: int,
        train_environments: int,
        device: torch.device,
        bootstrapped: bool = False,
        bootstrap_heads: int = 1,
        store_old_q_values: bool = False,
    ):
        rollout_shape = (steps_per_update, train_environments)
        state_q_shape = rollout_shape
        if bootstrapped:
            state_q_shape = rollout_shape + (bootstrap_heads,)

        self.observations = torch.empty(
            rollout_shape + observation_shape,
            dtype=torch.uint8,
            device=device,
        )
        self.actions = torch.empty(
            rollout_shape,
            dtype=torch.int64,
            device=device,
        )
        self.rewards = torch.empty(
            rollout_shape,
            dtype=torch.float32,
            device=device,
        )
        self.terminations = torch.empty(
            rollout_shape,
            dtype=torch.float32,
            device=device,
        )
        self.state_q_values = torch.empty(
            state_q_shape,
            dtype=torch.float32,
            device=device,
        )
        self.old_q_values = None
        if store_old_q_values:
            old_q_shape = state_q_shape if bootstrapped else rollout_shape
            self.old_q_values = torch.empty(
                old_q_shape,
                dtype=torch.float32,
                device=device,
            )
        self.bootstrap_masks = None
        if bootstrapped:
            self.bootstrap_masks = torch.empty(
                state_q_shape,
                dtype=torch.float32,
                device=device,
            )

    def insert(
        self,
        *,
        step: int,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        termination: torch.Tensor,
        state_q_values: torch.Tensor,
        old_q_values: Optional[torch.Tensor] = None,
        bootstrap_masks: Optional[torch.Tensor] = None,
    ) -> None:
        self.observations[step] = observation
        self.actions[step] = action
        self.rewards[step] = reward
        self.terminations[step] = termination
        self.state_q_values[step] = state_q_values

        if self.old_q_values is not None:
            if old_q_values is None:
                raise ValueError("Expected `old_q_values` for this rollout buffer.")
            self.old_q_values[step] = old_q_values

        if self.bootstrap_masks is not None:
            if bootstrap_masks is None:
                raise ValueError("Expected `bootstrap_masks` for this rollout buffer.")
            self.bootstrap_masks[step] = bootstrap_masks

    def flatten(self, targets: torch.Tensor):
        flattened_observations = self.observations.flatten(0, 1)
        flattened_actions = self.actions.reshape(-1)
        flattened_old_q_values = None
        if self.old_q_values is not None:
            flattened_old_q_values = self.old_q_values.flatten(0, 1)
        flattened_targets = targets.flatten(0, 1)
        flattened_bootstrap_masks = None
        if self.bootstrap_masks is not None:
            flattened_bootstrap_masks = self.bootstrap_masks.flatten(0, 1)
        return (
            flattened_observations,
            flattened_actions,
            flattened_old_q_values,
            flattened_targets,
            flattened_bootstrap_masks,
        )
