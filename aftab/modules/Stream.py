import torch
from aftab.constants import ModuleType
from .LearnableGELU import LearnableGELU


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization

        self.first_linear = torch.nn.Linear(input_dimension, hidden_dimension)
        self.second_linear = torch.nn.Linear(hidden_dimension, output_dimension)

        self.activation = (
            activation(hidden_dimension, dim=-1)
            if activation is LearnableGELU
            else activation()
        )

        if normalization:
            self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear(x)
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.second_linear(x)
        return x
