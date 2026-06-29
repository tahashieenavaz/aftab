import torch
from typing import Optional
from aftab.modules import Stream, forward_stream_heads
from .BaseNetwork import BaseNetwork


class BootstrappedDuellingNetwork(BaseNetwork):
    def __init__(self, *, bootstrap_heads: int = 10, **kwargs):
        super().__init__(**kwargs)
        if bootstrap_heads <= 0:
            raise ValueError("Expected `bootstrap_heads` to be positive.")

        self.bootstrapped = True
        self.bootstrap_heads = bootstrap_heads
        self.advantage_heads = torch.nn.ModuleList(
            [
                Stream(
                    input_dimension=self.feature_dimension,
                    hidden_dimension=self.embedding_dimension,
                    output_dimension=self.action_dimension,
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
                    output_dimension=1,
                    normalization=True,
                )
                for _ in range(self.bootstrap_heads)
            ]
        )

    def get_value_heads(self, features: torch.Tensor) -> torch.Tensor:
        return forward_stream_heads(heads=self.value_heads, x=features)

    def get_advantage_heads(self, features: torch.Tensor) -> torch.Tensor:
        advantages = forward_stream_heads(heads=self.advantage_heads, x=features)
        return advantages - advantages.mean(dim=2, keepdim=True)

    def get_q_heads(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        value = self.get_value_heads(features=features)
        advantage = self.get_advantage_heads(features=features)
        return value + advantage

    def gather_q_heads(
        self,
        q_heads: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        action_dimension = q_heads.shape[-1]
        indices = head_indices.reshape(-1, 1, 1).expand(-1, 1, action_dimension)
        return q_heads.gather(1, indices).squeeze(1)

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
