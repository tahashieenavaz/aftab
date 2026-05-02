import torch
from .AftabBaseMixin import AftabBaseMixin


class AftabQValueMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def __get_q_values_from_observations(self, observations: torch.Tensor):
        return self._network.get_q(observations)

    def get_q_values(
        self,
        float_train_observations: torch.Tensor = None,
        float_test_observations: torch.Tensor = None,
        float_observations: torch.Tensor = None,
        gradient: bool = False,
    ):
        with torch.set_grad_enabled(gradient):
            if float_observations is not None:
                return self.__get_q_values_from_observations(float_observations)
            train_count = float_train_observations.shape[0]
            q_values = self.__get_q_values_from_observations(
                torch.cat([float_train_observations, float_test_observations], dim=0)
            )
            return {"train": q_values[:train_count], "test": q_values[train_count:]}
