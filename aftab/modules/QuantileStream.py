import torch
from .Stream import Stream
from .CosineEmbeddingModule import CosineEmbeddingModule


class QuantileStream(torch.nn.Module):
    def __init__(
        self,
        *,
        action_dimension: int,
        embedding_dimension: int,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.mu = CosineEmbeddingModule(embedding_dimension=embedding_dimension)
        self.nu = Stream(
            input_dimension=embedding_dimension,
            hidden_dimension=embedding_dimension,
            output_dimension=action_dimension,
        )
        self.xi = torch.nn.Sequential(
            torch.nn.LazyLinear(embedding_dimension),
            torch.nn.ReLU(),
        )

    def forward(self, state_features, fractions):
        psi = self.xi(state_features)
        phi = self.mu(fractions)
        B, N, D = phi.shape
        psi = psi.view(B, 1, D)
        merged = psi * phi
        quantiles = self.nu(merged)
        return quantiles
