import torch
from typing import Type


class LowRankStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        rank: int,
        activation: Type[torch.nn.Module],
        normalization: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.first_linear_down = torch.nn.Linear(input_dimension, rank, bias=False)
        self.first_linear_up = torch.nn.Linear(
            rank, hidden_dimension, bias=not normalization
        )
        self.normalization_layer = (
            torch.nn.LayerNorm(hidden_dimension)
            if normalization
            else torch.nn.Identity()
        )
        self.activation = activation()
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.second_linear_down = torch.nn.Linear(hidden_dimension, rank, bias=False)
        self.second_linear_up = torch.nn.Linear(rank, output_dimension, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear_down(x)
        x = self.first_linear_up(x)
        x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.second_linear_down(x)
        x = self.second_linear_up(x)
        return x
