import torch
from .QuantileStream import QuantileStream


class DuellingQuantileStream(torch.nn.Module):
    def __init__(self, action_dimension: int, embedding_dimension: int):
        super().__init__()
        self.value_stream = QuantileStream(
            action_dimension=1,
            embedding_dimension=embedding_dimension,
        )
        self.advantage_stream = QuantileStream(
            action_dimension=action_dimension,
            embedding_dimension=embedding_dimension,
        )

    def forward(self, features: torch.Tensor, tau_hats: torch.Tensor):
        value_quantiles = self.value_stream(features, tau_hats)
        advantage_quantiles = self.advantage_stream(features, tau_hats)
        return (
            value_quantiles
            + advantage_quantiles
            - advantage_quantiles.mean(dim=2, keepdim=True)
        )
