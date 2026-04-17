import torch
from ..constants import ModuleType


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        # this is just here to keep the consistency. it doesn't do anything in this block.
        input_dimension: int = 3136,
        hidden_dimension: int = 512,
        output_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
        normalization: bool = True,
    ):
        super().__init__()
        self.first_linear = torch.nn.LazyLinear(hidden_dimension)
        self.second_linear = torch.nn.Linear(hidden_dimension, output_dimension)
        self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)
        self.activation = activation()
        self.normalization = normalization

    def forward(self, features):
        features = self.first_linear(features)
        if self.normalization:
            features = self.normalization_layer(features)
        features = self.activation(features)
        features = self.second_linear(features)
        return features
