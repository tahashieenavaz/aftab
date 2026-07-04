import torch
from aftab.constants import ModuleType
from .Stream import Stream


class WeightedMaskedStream(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType,
        normalization: bool,
    ):
        super().__init__()
        beta_dist = torch.distributions.Beta(0.5, 0.5)
        importance_weights = beta_dist.sample((input_dimension,))
        self.register_buffer("importance_weights", importance_weights)
        self.stream = Stream(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_x = x * self.importance_weights
        return self.stream(weighted_x)
