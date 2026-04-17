import torch
import math
from ..constants import ModuleType
from .Stream import Stream


class CosineEmbeddingModule(torch.nn.Module):
    def __init__(
        self,
        *,
        embedding_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.cosine_network = Stream(
            input_dimension=embedding_dimension,
            hidden_dimension=embedding_dimension,
            output_dimension=embedding_dimension,
        )
        pi_indices = math.pi * torch.arange(0, embedding_dimension).float()
        self.register_buffer("pi_indices", pi_indices)
        self.activation = activation()

    def forward(self, fractions: torch.Tensor):
        cos = torch.cos(fractions.unsqueeze(-1) * self.pi_indices.view(1, 1, -1))
        cos = cos * math.sqrt(1.0 / self.embedding_dimension)
        embeddings = self.cosine_network(cos)
        return self.activation(embeddings)
