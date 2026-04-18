import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def __regression_q_values(self, float_observations):
        return self._network(float_observations)

    def __distributional_q_values(self, float_observations):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            features = self._network.phi(float_observations)
            tau, tau_hat, q_probs, _ = self._network.fraction_proposal(features)
            quantiles = self._network.quantile_value(features, tau_hat)
            return (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)

    def get_distributional_q_values(self, float_observations, gradient: bool = False):
        if not gradient:
            with torch.no_grad():
                return self.__distributional_q_values(
                    float_observations=float_observations
                )

        return self.__distributional_q_values(float_observations=float_observations)

    def get_regression_q_values(self, float_observations, gradient: bool = False):
        if not gradient:
            with torch.no_grad():
                return self.__regression_q_values(float_observations=float_observations)

        return self.__regression_q_values(float_observations=float_observations)
