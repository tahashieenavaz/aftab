import torch
from hl_gauss_pytorch import HLGaussLoss
from typing import Optional
from itertools import cycle
from torch.nn import SiLU, CELU, SELU, Mish, PReLU
from aftab.modules import GatedLinearUnit
from aftab.modules import GoLU
from .BaseNetwork import BaseNetwork

_ACTIVATION_POOL = cycle(
    [
        SiLU,
        CELU,
        SELU,
        Mish,
        GoLU,
        PReLU,
        PReLU,
        PReLU,
        PReLU,
        PReLU,
    ]
)
_ADVANTAGE_ACTIVATION_POOL = cycle(_ACTIVATION_POOL)
_VALUE_ACTIVATION_POOL = cycle(_ACTIVATION_POOL)


class DistributionalBootstrappedDuellingMixedNetwork(BaseNetwork):
    def __init__(
        self,
        *,
        distributional_bins: int,
        distributional_min_value: float,
        distributional_max_value: float,
        distributional_sigma: float,
        bootstrap_heads: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if bootstrap_heads <= 0:
            raise ValueError("Expected `bootstrap_heads` to be positive.")

        self.distributional = True
        self.bootstrapped = True
        self.bootstrap_heads = bootstrap_heads
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
                GatedLinearUnit(
                    input_dimension=self.feature_dimension,
                    hidden_dimension=self.embedding_dimension // 2,
                    output_dimension=self.action_dimension * self.distributional_bins,
                    activation=next(_ADVANTAGE_ACTIVATION_POOL),
                    normalization=True,
                )
                for _ in range(self.bootstrap_heads)
            ]
        )
        self.value_heads = torch.nn.ModuleList(
            [
                GatedLinearUnit(
                    input_dimension=self.feature_dimension,
                    hidden_dimension=self.embedding_dimension // 2,
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
        advantages = [
            head(features).reshape(
                batch_size,
                self.action_dimension,
                self.distributional_bins,
            )
            for head in self.advantage_heads
        ]
        advantages = torch.stack(advantages, dim=1)
        return advantages - advantages.mean(dim=2, keepdim=True)

    def get_q_logits_heads(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x=states)
        value_logits = self.get_value_logits_heads(features=features)
        advantage_logits = self.get_advantage_logits_heads(features=features)
        return value_logits + advantage_logits

    def gather_q_heads(
        self,
        q_heads: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        action_dimension = q_heads.shape[-1]
        indices = head_indices.reshape(-1, 1, 1).expand(-1, 1, action_dimension)
        return q_heads.gather(1, indices).squeeze(1)

    def gather_q_logits_heads(
        self,
        q_logits_heads: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        action_dimension = q_logits_heads.shape[-2]
        bins = q_logits_heads.shape[-1]
        indices = head_indices.reshape(-1, 1, 1, 1).expand(
            -1,
            1,
            action_dimension,
            bins,
        )
        return q_logits_heads.gather(1, indices).squeeze(1)

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
