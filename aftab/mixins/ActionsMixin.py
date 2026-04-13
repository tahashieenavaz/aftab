import torch
from ..functions import epsilon_greedy_vectorized


class ActionsMixin:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def get_actions(self, q_values, epsilon_value):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            if self._network.epsilon_greedy:
                actions = epsilon_greedy_vectorized(
                    q_values, self.get_epsilons(epsilon_value)
                )
            else:
                actions = q_values.argmax(dim=-1).cpu().numpy()
        return actions

    def split_actions(self, actions):
        actions_train = actions[: self.num_train_environments]
        actions_test = actions[self.num_train_environments :]
        return actions_train, actions_test
