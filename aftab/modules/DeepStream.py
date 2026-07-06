import torch
import random
from typing import Literal
from aftab.functions import calculate_matched_width
from aftab.constants import ModuleType


class DeepStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType | Literal["random"],
        normalization: bool,
        depth: int = 4,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(
                "DeepStream depth must be at least 2. Use Stream for depth=2."
            )

        self.internal_width = calculate_matched_width(
            input_dimension, hidden_dimension, output_dimension, depth, normalization
        )

        self.layers = torch.nn.ModuleList()

        current_dim = input_dimension
        for _ in range(depth - 1):
            self.layers.append(torch.nn.Linear(current_dim, self.internal_width))
            if normalization:
                self.layers.append(torch.nn.LayerNorm(self.internal_width))
            if issubclass(activation, torch.nn.Module):
                self.layers.append(activation())
            elif activation == "random":
                random_activation = random.choice(
                    [torch.nn.GELU, torch.nn.ReLU, torch.nn.SiLU]
                )
                self.layers.append(random_activation())
            current_dim = self.internal_width

        self.output_layer = torch.nn.Linear(current_dim, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
