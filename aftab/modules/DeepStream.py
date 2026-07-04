import torch
import math
from aftab.constants import ModuleType


def calculate_matched_width(
    input_dim: int, target_hidden_dim: int, output_dim: int, depth: int, norm: bool
) -> int:
    if depth <= 2:
        return target_hidden_dim
    target_params = (input_dim * target_hidden_dim + target_hidden_dim) + (
        target_hidden_dim * output_dim + output_dim
    )
    if norm:
        target_params += 2 * target_hidden_dim
    a = depth - 2
    b = input_dim + output_dim + (depth - 2)
    if norm:
        b += 2 * (depth - 1)
    c = output_dim - target_params
    W = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return int(round(W))


class DeepStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType,
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
            self.layers.append(activation())
            current_dim = self.internal_width

        self.output_layer = torch.nn.Linear(current_dim, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
