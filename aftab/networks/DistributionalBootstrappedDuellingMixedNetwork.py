import torch
from hl_gauss_pytorch import HLGaussLoss
from typing import Optional
from itertools import cycle
from aftab.constants import ActivationPool
from aftab.modules import Stream
from .BaseNetwork import BaseNetwork

_ADVANTAGE_ACTIVATION_POOL = cycle(ActivationPool)
_VALUE_ACTIVATION_POOL = cycle(ActivationPool)


class DistributionalBootstrappedDuellingMixedNetwork(BaseNetwork):
    def __init__(
        self,
        *,
        distributional_bins: int,
        distributional_min_value: float,
        distributional_max_value: float,
        distributional_sigma: float,
        bootstrap_heads: int = 10,
        training_perturbation_std: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bootstrap_heads <= 0:
            raise ValueError("Expected `bootstrap_heads` to be positive.")
        if training_perturbation_std < 0:
            raise ValueError("Expected `training_perturbation_std` to be non-negative.")

        self.distributional = True
        self.bootstrapped = True
        self.bootstrap_heads = bootstrap_heads
        self.training_perturbation_std = training_perturbation_std
        self.distributional_bins = distributional_bins
        self.hl_gauss_loss = HLGaussLoss(
            min_value=distributional_min_value,
            max_value=distributional_max_value,
            num_bins=distributional_bins,
            sigma=distributional_sigma,
            clamp_to_range=True,
        )
        self.advantage_heads = torch.nn.ModuleList(
            [
                Stream(
                    input_dimension=self.feature_dimension,
                    hidden_dimension=self.embedding_dimension,
                    output_dimension=self.action_dimension * self.distributional_bins,
                    activation=next(_ADVANTAGE_ACTIVATION_POOL),
                    normalization=True,
                )
                for _ in range(self.bootstrap_heads)
            ]
        )
        self.value_heads = torch.nn.ModuleList(
            [
                Stream(
                    input_dimension=self.feature_dimension,
                    hidden_dimension=self.embedding_dimension,
                    output_dimension=self.distributional_bins,
                    activation=next(_VALUE_ACTIVATION_POOL),
                    normalization=True,
                )
                for _ in range(self.bootstrap_heads)
            ]
        )

    def get_value_logits_heads(self, features: torch.Tensor) -> torch.Tensor:
        values = [head(features) for head in self.value_heads]
        return torch.stack(values, dim=1).unsqueeze(2)

    def get_advantage_logits_heads(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        advantages = torch.stack(
            [head(features) for head in self.advantage_heads], dim=1
        )
        advantages = advantages.view(
            batch_size,
            self.bootstrap_heads,
            self.action_dimension,
            self.distributional_bins,
        )
        return advantages - advantages.mean(dim=2, keepdim=True)

    def perturb_training_features(self, features: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or not torch.is_grad_enabled()
            or self.training_perturbation_std == 0.0
        ):
            return features
        return features + torch.randn_like(features) * self.training_perturbation_std

    def get_q_logits_heads(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x=states)
        features = self.perturb_training_features(features)
        value_logits = self.get_value_logits_heads(features=features)
        advantage_logits = self.get_advantage_logits_heads(features=features)
        return value_logits + advantage_logits

    def gather_q_heads(
        self,
        q_heads: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices = torch.arange(q_heads.size(0), device=q_heads.device)
        return q_heads[batch_indices, head_indices]

    def gather_q_logits_heads(
        self,
        q_logits_heads: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices = torch.arange(
            q_logits_heads.size(0), device=q_logits_heads.device
        )
        return q_logits_heads[batch_indices, head_indices]

    def get_q_logits(
        self,
        states: torch.Tensor,
        head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_logits_heads = self.get_q_logits_heads(states)
        if head_indices is None:
            return q_logits_heads.mean(dim=1)
        return self.gather_q_logits_heads(
            q_logits_heads=q_logits_heads,
            head_indices=head_indices,
        )

    def get_q_heads(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.get_q_logits_heads(states)
        q_values = self.hl_gauss_loss(
            logits.reshape(-1, self.distributional_bins),
        )
        return q_values.reshape(logits.shape[:-1])

    def get_q(
        self,
        states: torch.Tensor,
        head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_heads = self.get_q_heads(states)
        if head_indices is None:
            return q_heads.mean(dim=1)
        return self.gather_q_heads(q_heads=q_heads, head_indices=head_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
