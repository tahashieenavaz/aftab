import torch
from typing import Type
from aftab.constants import ModuleType
from .Stream import Stream

_DEFAULT_DOWNSAMPLE_ACTIVATION = torch.nn.GELU
_DEFAULT_STREAM_ACTIVATION = torch.nn.ReLU


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
        batch_first: bool = True,
        downsample_activation: ModuleType = _DEFAULT_DOWNSAMPLE_ACTIVATION,
        stream_activation: ModuleType = _DEFAULT_STREAM_ACTIVATION,
    ):
        super().__init__()
        self.normalization = normalization

        self.__initialize_downsample(
            normalization=normalization,
            activation=downsample_activation,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
        )

        self.recurrent = torch.nn.GRU(
            hidden_dimension,
            hidden_dimension,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
            activation=stream_activation,
        )

    def __initialize_downsample(
        self,
        *,
        normalization: bool,
        activation: ModuleType,
        input_dimension: int,
        hidden_dimension: int,
    ):
        self.downsample = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            activation(),
        )
        if normalization:
            self.downsample = torch.nn.Sequential(
                *list(self.downsample), torch.nn.LayerNorm(hidden_dimension)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x, _ = self.recurrent(x)
        features = x.mean(dim=1)
        return self.stream(features)
