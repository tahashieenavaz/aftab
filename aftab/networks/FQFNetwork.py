import torch
import torch.nn.functional as F
from .DuellingNetwork import BaseNetwork
from ..modules import QuantileStream, FractionProposalStream


class FQFNetwork(BaseNetwork):
    def __init__(
        self,
        entropy_coefficient: float = 1e-3,
        fraction_proposal_coefficient: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.entropy_coefficient = entropy_coefficient
        self.fraction_proposal_coefficient = fraction_proposal_coefficient

        self.fraction_proposal = FractionProposalStream(
            number_quantiles=kwargs["number_quantiles"],
            embedding_dimension=kwargs["embedding_dimension"],
        )
        self.quantile_value = QuantileStream(
            action_dimension=kwargs["action_dimension"],
            embedding_dimension=kwargs["embedding_dimension"],
        )

    def forward(self, x: torch.Tensor):
        features = self.get_features(x)
        taus, tau_hats, q_probs, entropy = self.fraction_proposal(features)
        quantiles = self.quantile_value(features, tau_hats)
        return {
            "features": features,
            "quantiles": quantiles,
            "taus": taus,
            "tau_hats": tau_hats,
            "q_probs": q_probs,
            "entropy": entropy,
        }

    def get_q(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        quantiles = output["quantiles"]
        q_probs = output["q_probs"]
        return (quantiles * q_probs.unsqueeze(-1)).sum(dim=1)

    def quantile_value_loss(self, predictions, targets, tau_hats):
        td_error = targets.unsqueeze(1) - predictions.unsqueeze(2)
        huber = torch.where(
            td_error.abs() < 1.0,
            0.5 * td_error.pow(2),
            td_error.abs() - 0.5,
        )
        tau = tau_hats.unsqueeze(2)
        loss = torch.abs(tau - (td_error.detach() < 0).float()) * huber
        return loss.mean()

    def fraction_proposal_loss(self, features, actions, quantiles, taus, q_probs):
        inner_taus = taus[:, 1:-1]
        quantiles_tau = self.quantile_value(features.detach(), inner_taus)
        action_idx = actions.unsqueeze(-1).unsqueeze(-1)
        q_tau = torch.take_along_dim(
            quantiles_tau,
            action_idx.expand(-1, inner_taus.size(1), 1),
            dim=-1,
        ).squeeze(-1)
        q_tau_hat = torch.take_along_dim(
            quantiles,
            action_idx.expand(-1, quantiles.size(1), 1),
            dim=-1,
        ).squeeze(-1)
        gradient_term = torch.abs(2 * q_tau - q_tau_hat[:, 1:] - q_tau_hat[:, :-1])
        loss = (q_probs[:, :-1] * gradient_term).sum(dim=1).mean()
        return loss

    def compute_losses(self, observations, targets, actions):
        output = self.forward(observations)

        features = output["features"]
        quantiles = output["quantiles"]
        taus = output["taus"]
        tau_hats = output["tau_hats"]
        q_probs = output["q_probs"]
        entropy = output["entropy"]

        action_idx = actions.unsqueeze(-1).unsqueeze(-1)
        predictions = torch.take_along_dim(
            quantiles,
            action_idx.expand(-1, quantiles.size(1), 1),
            dim=-1,
        ).squeeze(-1)
        val_loss = self.value_loss(predictions, targets, tau_hats)
        fpn_loss = self.fpn_loss(features, actions, quantiles, taus, q_probs)
        entropy_loss = -entropy.mean()
        total_loss = (
            val_loss
            + self.fraction_proposal_coefficient * fpn_loss
            + self.entropy_coefficient * entropy_loss
        )

        return {
            "total_loss": total_loss,
            "value_loss": val_loss,
            "fpn_loss": fpn_loss,
            "entropy_loss": entropy_loss,
        }
