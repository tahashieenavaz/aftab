import torch
import math
from aftab.constants import ModuleType
from .Stream import Stream


class MaskedStream(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType,
        normalization: bool,
        p: float,
    ):
        super().__init__()
        mask_size = int(p * input_dimension)
        new_hidden_dimension = math.ceil(hidden_dimension * (1 / p))
        indices = torch.randperm(input_dimension)[:mask_size]
        self.register_buffer("mask", indices)
        self.stream = Stream(
            input_dimension=mask_size,
            hidden_dimension=new_hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_x = x[:, self.mask]
        return self.stream(masked_x)
