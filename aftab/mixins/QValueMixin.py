import torch


class QValueMixin:
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
            test_q_values = self.__get_q_values_from_observations(
                float_test_observations
            )
            train_q_values = self.__get_q_values_from_observations(
                float_train_observations
            )
            return {"test": test_q_values, "train": train_q_values}
