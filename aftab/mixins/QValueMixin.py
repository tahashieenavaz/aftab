import torch


class QValueMixin:
    def __init__(self):
        super().__init__()

    def get_q_values(self, float_observations, gradient: bool = False):
        with torch.set_grad_enabled(gradient):
            if getattr(self, "augmentation", "none") == "none":
                return self._network(float_observations)

            q_values_list = [self._network(float_observations) for _ in range(2)]
            return torch.stack(q_values_list).mean(dim=0)

    def get_q_and_quantiles(self, float_observations, gradient: bool = False):
        with (
            torch.set_grad_enabled(gradient),
            torch.autocast(device_type=self.device.type, dtype=torch.float16),
        ):
            if getattr(self, "augmentation", "none") == "none":
                features = self._network.phi(float_observations)
                _, tau_hat, q_probs, _ = self._network.fraction_proposal(features)
                quantiles = self._network.quantile_value(features, tau_hat)
                q_values = (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)
                return q_values, quantiles

            q_values_list = []
            quantiles_list = []
            for _ in range(2):
                features = self._network.phi(float_observations)
                _, tau_hat, q_probs, _ = self._network.fraction_proposal(features)
                quantiles = self._network.quantile_value(features, tau_hat)
                q_values = (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)
                q_values_list.append(q_values)
                quantiles_list.append(quantiles)

            avg_q_values = torch.stack(q_values_list).mean(dim=0)
            avg_quantiles = torch.stack(quantiles_list).mean(dim=0)
            return avg_q_values, avg_quantiles
