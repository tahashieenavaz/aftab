import torch
from .Stream import Stream
from .CosineEmbeddingModule import CosineEmbeddingModule


class QuantileStream(torch.nn.Module):
    def __init__(
        self, *, action_dimension: int, embedding_dimension: int, feature_dimension: int
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.mu = CosineEmbeddingModule(embedding_dimension=embedding_dimension)
        self.nu = Stream(
            input_dimension=embedding_dimension,
            embedding_dimension=embedding_dimension,
            output_dimension=action_dimension,
        )
        self.xi = torch.nn.Linear(embedding_dimension, feature_dimension)

    def forward(
        self, state_features: torch.Tensor, fractions: torch.Tensor
    ) -> torch.Tensor:
        chi = self.mu(fractions)
        chi = self.xi(chi)
        psi = state_features.unsqueeze(1)
        merged = psi * chi
        quantiles = self.nu(merged)
        return quantiles
