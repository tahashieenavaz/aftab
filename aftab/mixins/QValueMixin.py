import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def get_q_values(self, float_observations, gradient: bool = False):
        with torch.set_grad_enabled(gradient):
            return self._network(float_observations)

    def get_q_and_quantiles(self, float_observations, gradient: bool = False):
        with (
            torch.set_grad_enabled(gradient),
            torch.autocast(device_type=self.device.type, dtype=torch.float16),
        ):
            features = self._network.phi(float_observations)
            _, tau_hat, q_probs, _ = self._network.fraction_proposal(features)
            quantiles = self._network.quantile_value(features, tau_hat)
            q_values = (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)
            return q_values, quantiles
