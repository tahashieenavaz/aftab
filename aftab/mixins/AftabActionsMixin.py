import torch
import numpy
from typing import Tuple
from .AftabBaseMixin import AftabBaseMixin
from ..functions import epsilon_greedy_vectorized


class AftabActionsMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def get_action_tensors(
        self,
        q_values_train: torch.Tensor,
        q_values_test: torch.Tensor,
        epsilon_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._network.epsilon_greedy:
            actions_train = epsilon_greedy_vectorized(q_values_train, epsilon_value)
        else:
            actions_train = q_values_train.argmax(dim=-1)
        actions_test = q_values_test.argmax(dim=-1)
        return actions_train, actions_test

    @torch.no_grad()
    def get_actions(
        self,
        q_values_train: torch.Tensor,
        q_values_test: torch.Tensor,
        epsilon_value: float,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        actions_train, actions_test = self.get_action_tensors(
            q_values_train=q_values_train,
            q_values_test=q_values_test,
            epsilon_value=epsilon_value,
        )
        return actions_train.cpu().numpy(), actions_test.cpu().numpy()
