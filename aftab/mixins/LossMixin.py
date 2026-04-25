import torch
from ..functions import mse_loss


class LossMixin:
    def __init__(self):
        super().__init__()

    def get_loss(
        self, mini_batch_observations, mini_batch_actions, mini_batch_targets
    ) -> torch.Tensor:
        mini_batch_observations = mini_batch_observations.float()
        q_values = self._network.get_q(mini_batch_observations)
        q_taken = q_values.gather(1, mini_batch_actions.unsqueeze(1)).squeeze(1)
        return mse_loss(q_taken, mini_batch_targets)
