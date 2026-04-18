import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def get_q_values(self, float_observations, gradient: bool = False):
        if not gradient:
            with torch.no_grad():
                q_values = self._network(float_observations)
        else:
            q_values = self._network(float_observations)
        return q_values
