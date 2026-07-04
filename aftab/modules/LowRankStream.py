import torch
from aftab.constants import ModuleType


class LowRankStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        rank: int,
        activation: ModuleType,
        normalization: bool,
    ):
        super().__init__()
        self.normalization = normalization
        self.first_linear_down = torch.nn.Linear(input_dimension, rank, bias=False)
        self.first_linear_up = torch.nn.Linear(rank, hidden_dimension)
        self.second_linear_down = torch.nn.Linear(hidden_dimension, rank, bias=False)
        self.second_linear_up = torch.nn.Linear(rank, output_dimension)
        self.activation = activation()
        if normalization:
            self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear_down(x)
        x = self.first_linear_up(x)
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.second_linear_down(x)
        x = self.second_linear_up(x)
        return x
