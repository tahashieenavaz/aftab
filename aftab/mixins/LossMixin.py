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
        current_quantiles = self._gather_action_quantiles(
            quantiles=quantiles, actions=mini_batch_actions
        )
        self._ensure_finite(current_quantiles, "current_quantiles")

        kappa = float(getattr(self, "kappa"))
        target_quantile_values = mini_batch_targets.detach().unsqueeze(-1)
        chosen_action_quantile_values = current_quantiles.unsqueeze(-1)
        target_expanded = target_quantile_values[:, :, None, :].expand(
            -1, -1, chosen_action_quantile_values.shape[1], -1
        )
        chosen_expanded = chosen_action_quantile_values[:, None, :, :].expand(
            -1, target_quantile_values.shape[1], -1, -1
        )
        bellman_errors = target_expanded - chosen_expanded
        huber_loss = torch.nn.functional.huber_loss(
            chosen_expanded,
            target_expanded,
            reduction="none",
            delta=kappa,
        )

        replay_quantiles = tau_hat_detached[:, None, :, None].expand(
            -1, target_quantile_values.shape[1], -1, -1
        )
        asym_weights = torch.abs(
            replay_quantiles.detach() - (bellman_errors.detach() < 0).float()
        )
        quantile_huber_loss = (asym_weights * huber_loss) / kappa
        quantile_loss = quantile_huber_loss.sum(dim=2).mean(dim=1).mean()

        with torch.no_grad():
            quantiles_tau = self._network.quantile_value(
                features.detach(), tau[:, 1:-1]
            )
            Z_tau = self._gather_action_quantiles(
                quantiles=quantiles_tau, actions=mini_batch_actions
            )

        Z_tau_hat = current_quantiles.detach()
        gradients_tau = 2 * Z_tau - Z_tau_hat[:, :-1] - Z_tau_hat[:, 1:]
        self._ensure_finite(gradients_tau, "gradients_tau")

        entropy_coefficient = getattr(self, "entropy_coefficient")
        entropy_loss_scale = getattr(self, "fqf_entropy_loss_scale")
        fraction_loss = (tau[:, 1:-1] * gradients_tau.detach()).sum(dim=1).mean()
        entropy_loss = entropy_coefficient * torch.mean(-entropy_loss_scale * entropy)

        return quantile_loss, fraction_loss, entropy_loss
