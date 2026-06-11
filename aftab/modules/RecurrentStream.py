import torch
from .Stream import Stream


class RecurrentStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        stream_hidden_dimension: int,
        stream_output_dimension: int,
        heads: int = 8,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dimension,
            num_heads=heads,
            batch_first=True,
        )
        self.attention_projection = torch.nn.Linear(input_dimension, hidden_dimension)
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=stream_output_dimension,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_projection(x)
        features, _ = self.attention(x, x, x)
        return self.stream(features)
