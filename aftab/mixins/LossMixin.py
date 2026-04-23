import torch
from ..functions import mse_loss


class LossMixin:
    def __init__(self):
        super().__init__()

    def _ensure_finite(self, tensor: torch.Tensor, name: str):
        if torch.isfinite(tensor).all():
            return
        raise FloatingPointError(f"Detected non-finite values in `{name}`.")

    def get_loss(
        self, mini_batch_observations, mini_batch_actions, mini_batch_targets
    ) -> torch.Tensor:
        q_values = self.get_q_values(
            float_observations=mini_batch_observations.float(),
            gradient=True,
        )
        q_taken = q_values.gather(1, mini_batch_actions.unsqueeze(1)).squeeze(1)
        return mse_loss(q_taken, mini_batch_targets)

    def get_distributional_loss(
        self, mini_batch_observations, mini_batch_actions, mini_batch_targets
    ):
        mini_batch_observations_float = mini_batch_observations.float()
        features = self._network.get_features(mini_batch_observations_float)
        tau, tau_hat, q_probs, entropy = self._network.fraction_proposal(
            features.detach()
        )
        self._ensure_finite(tau, "tau")
        self._ensure_finite(tau_hat, "tau_hat")
        tau_hat_detached = tau_hat.detach()
        quantiles = self._network.quantile_value(features, tau_hat_detached)
        action_idx = (
            mini_batch_actions.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.number_quantiles, -1)
        )
        current_quantiles = quantiles.gather(2, action_idx).squeeze(-1)
        self._ensure_finite(current_quantiles, "current_quantiles")
        target_expanded = mini_batch_targets.unsqueeze(1).expand(
            -1, self.number_quantiles, self.number_quantiles
        )
        current_expanded = current_quantiles.unsqueeze(2).expand(
            -1, self.number_quantiles, self.number_quantiles
        )

        u = target_expanded - current_expanded
        huber_loss = torch.nn.functional.huber_loss(
            current_expanded,
            target_expanded,
            reduction="none",
            delta=1.0,
        )
        asym_weights = torch.abs(tau_hat_detached.unsqueeze(2) - (u < 0).float())
        quantile_loss = (asym_weights * huber_loss).sum(dim=1).mean(dim=1).mean()
        with torch.no_grad():
            quantiles_tau = self._network.quantile_value(
                features.detach(), tau[:, 1:-1]
            )
            action_idx_tau = (
                mini_batch_actions.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.number_quantiles - 1, -1)
            )
            Z_tau = quantiles_tau.gather(2, action_idx_tau).squeeze(-1)

        Z_tau_hat = current_quantiles.detach()
        gradients_tau = 2 * Z_tau - Z_tau_hat[:, :-1] - Z_tau_hat[:, 1:]
        self._ensure_finite(gradients_tau, "gradients_tau")

        entropy_coefficient = getattr(self, "entropy_coefficient")
        fraction_loss = (tau[:, 1:-1] * gradients_tau.detach()).sum(
            dim=1
        ).mean() - entropy_coefficient * entropy.mean()

        return quantile_loss, fraction_loss
