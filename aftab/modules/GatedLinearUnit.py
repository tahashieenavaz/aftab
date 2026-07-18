import torch
from aftab.typing import ModuleType


class GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType = torch.nn.SiLU,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization

        if self.normalization:
            self.normalization_layer = torch.nn.LayerNorm(input_dimension)

        self.projection = torch.nn.Linear(input_dimension, hidden_dimension * 2)
        self.output = torch.nn.Linear(hidden_dimension, output_dimension)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization:
            x = self.normalization_layer(x)

        pre_gate, hidden = self.projection(x).chunk(2, dim=-1)
        # self.activation(pre_gate) is often called gate. I removed the allocation here to enhance memory efficiency
        return self.output(self.activation(pre_gate) * hidden)
