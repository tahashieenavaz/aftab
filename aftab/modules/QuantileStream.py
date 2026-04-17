import torch
from ..modules import CosineEmbeddingModule, Stream


class QuantileStream(torch.nn.Module):
    def __init__(
        self,
        action_dimension: int,
        embedding_dimension: int = 256,
    ):
        super().__init__()
        self.mu = CosineEmbeddingModule(embedding_dimension)
        self.nu = Stream(
            input_dimension=embedding_dimension,
            hidden_dimension=embedding_dimension,
            output_dimension=action_dimension,
        )
        self.xi = torch.nn.Sequential(
            torch.nn.LazyLinear(embedding_dimension), torch.nn.ReLU()
        )

    def forward(self, state_features, fractions):
        proj_features = self.xi(state_features)
        phi = self.mu(fractions)
        merged_features = proj_features.unsqueeze(1) * phi
        return self.nu(merged_features)
