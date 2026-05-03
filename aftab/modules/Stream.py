import torch
import inspect
from ..constants import ModuleType


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

        self.activation = self.__initiate_activation(activation_class=activation)

        if normalization:
            self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)

    def __initiate_activation(self, activation_class: ModuleType) -> torch.nn.Module:
        args = inspect.signature(activation_class).parameters
        if "inplace" in args:
            return activation_class(inplace=True)
        return activation_class()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear(x)
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.second_linear(x)
        return x
