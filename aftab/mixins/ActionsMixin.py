import torch
import numpy
from typing import Tuple
from ..functions import epsilon_greedy_vectorized


class ActionsMixin:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def get_actions(self, q_values, epsilon_value) -> numpy.ndarray:
        device_type = getattr(self.device, "type", "cpu")
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            if self._network.epsilon_greedy:
                actions = epsilon_greedy_vectorized(
                    q_values, self.get_epsilons(epsilon_value)
                )
            else:
                actions = q_values.argmax(dim=-1)
        return actions.cpu().numpy()

    def split_actions(
        self, actions: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        actions_train = actions[: self.num_train_environments]
        actions_test = actions[self.num_train_environments :]
        return actions_train, actions_test
