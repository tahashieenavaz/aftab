import torch
from aftab.constants import ModuleType


class GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization

        if self.normalization:
            self.normalization_layer = torch.nn.LayerNorm(input_dimension)

        self.projection = torch.nn.Linear(input_dimension, hidden_dimension * 2)
        self.output = torch.nn.Linear(hidden_dimension, output_dimension)
        self.activation = (
            activation(hidden_dimension)
            if isinstance(activation, torch.nn.PReLU)
            else activation()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization:
            x = self.normalization_layer(x)

        x1, x2 = self.projection(x).chunk(2, dim=-1)
        return self.output(self.activation(x1) * x2)
