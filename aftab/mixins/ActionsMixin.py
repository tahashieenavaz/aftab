import torch
import numpy
from typing import Tuple
from ..functions import epsilon_greedy_vectorized


class ActionsMixin:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def get_actions(
        self,
        q_values_train: torch.Tensor,
        q_values_test: torch.Tensor,
        epsilon_value: float,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self._network.epsilon_greedy:
            actions_train = epsilon_greedy_vectorized(q_values_train, epsilon_value)
        else:
            actions_train = q_values_train.argmax(dim=-1)
        actions_test = q_values_test.argmax(dim=-1)
        return actions_train.cpu().numpy(), actions_test.cpu().numpy()
