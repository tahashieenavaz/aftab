import torch
from typing import Type
from .Stream import Stream


class RecurrentStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        stream_hidden_dimension: int,
        num_layers: int = 1,
        normalization: bool = True,
        downsample_activation: Type[torch.nn.Module] = torch.nn.GELU,
        stream_activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.normalization = normalization
        self.downsample = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            downsample_activation(),
            torch.nn.LayerNorm(hidden_dimension),
        )
        self.recurrent = torch.nn.GRU(
            hidden_dimension,
            hidden_dimension,
            num_layers=num_layers,
            batch_first=True,
        )
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
            activation=stream_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x, _ = self.recurrent(x)
        features = x.mean(dim=1)
        return self.stream(features)
