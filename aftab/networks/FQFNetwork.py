import torch
from .DuellingNetwork import BaseNetwork
from ..modules import QuantileStream, FractionProposalStream


class FQFNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fraction_proposal = FractionProposalStream(
            number_quantiles=kwargs["number_quantiles"],
            embedding_dimension=kwargs["embedding_dimension"],
        )
        self.quantile = QuantileStream(
            action_dimension=kwargs["action_dimension"],
            embedding_dimension=kwargs["embedding_dimension"],
        )

    def get_quantiles(self, x: torch.Tensor):
        features = self.get_features(x)
        taus, tau_hats, q_probs = self.fpn(features)
        quantiles = self.qvn(features, tau_hats)
        return quantiles, taus, tau_hats, q_probs

    def get_q(self, x: torch.Tensor) -> torch.Tensor:
        quantiles, _, _, q_probs = self.get_quantiles(x)
        q_values = (quantiles * q_probs.unsqueeze(-1)).sum(dim=1)
        return q_values

    def value_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, tau_hats: torch.Tensor
    ) -> torch.Tensor:
        td_error = targets.unsqueeze(1) - predictions.unsqueeze(2)
        huber = torch.where(
            td_error.abs() < 1.0, 0.5 * td_error.pow(2), td_error.abs() - 0.5
        )
        tau = tau_hats.unsqueeze(2)
        loss = (torch.abs(tau - (td_error.detach() < 0).float()) * huber).mean()
        return loss

    @torch.no_grad()
    def fpn_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        quantiles_at_tau_hat: torch.Tensor,
        taus: torch.Tensor,
        q_probs: torch.Tensor,
    ) -> torch.Tensor:
        inner_taus = taus[:, 1:-1]
        quantiles_at_tau = self.qvn(features.detach(), inner_taus)
        action_idx = (
            actions.unsqueeze(-1).unsqueeze(-1).expand(-1, inner_taus.size(1), 1)
        )
        q_tau = quantiles_at_tau.gather(dim=-1, index=action_idx).squeeze(-1)
        action_idx_hat = (
            actions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, quantiles_at_tau_hat.size(1), 1)
        )
        q_tau_hat = quantiles_at_tau_hat.gather(dim=-1, index=action_idx_hat).squeeze(
            -1
        )
        gradient_term = 2 * q_tau - q_tau_hat[:, 1:] - q_tau_hat[:, :-1]
        loss = (q_probs[:, :-1] * gradient_term).sum(dim=1).mean()
        return loss

    def loss(self, mini_batch_observations, mini_batch_targets, mini_batch_actions):
        features = self._network.get_features(mini_batch_observations)
        taus, tau_hats, q_probs = self._network.fpn(features)
        quantiles = self._network.qvn(features, tau_hats)
        action_idx = (
            mini_batch_actions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, quantiles.size(1), 1)
        )
        predictions = quantiles.gather(dim=-1, index=action_idx).squeeze(-1)
        val_loss = self._network.value_loss(predictions, mini_batch_targets, tau_hats)
        fpn_loss = self._network.fpn_loss(
            features, mini_batch_actions, quantiles, taus, q_probs
        )
        return val_loss + fpn_loss
