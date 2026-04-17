import torch
import math


class CosineEmbeddingModule(torch.nn.Module):
    def __init__(self, *, embedding_dimension: int = 256, activation=torch.nn.ReLU):
        super().__init__()
        self.cosine_network = torch.nn.Linear(embedding_dimension, embedding_dimension)
        pi_indices = math.pi * torch.arange(0, embedding_dimension).float()
        self.register_buffer("pi_indices", pi_indices)
        self.activation = activation()

    def forward(self, fractions: torch.Tensor):
        cosine_embeddings = torch.cos(
            fractions.unsqueeze(-1) * self.pi_indices.view(1, 1, -1)
        )
        return self.activation(self.cosine_network(cosine_embeddings))
