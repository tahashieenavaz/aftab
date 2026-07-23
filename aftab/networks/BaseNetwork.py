import inspect

import torch
from aftab.typing import ModuleType
from aftab.common import LinearEpsilon


class BaseNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        action_dimension: int,
        embedding_dimension: int,
        encoder: ModuleType,
        channels_last: bool,
        observation_shape: tuple[int, int, int] = (4, 84, 84),
    ):
        super().__init__()
        self.channels_last = channels_last
        self.observation_shape = tuple(observation_shape)
        if len(self.observation_shape) != 3 or any(
            dimension <= 0 for dimension in self.observation_shape
        ):
            raise ValueError(
                "Expected `observation_shape` to contain three positive dimensions."
            )

        encoder_parameters = inspect.signature(encoder).parameters.values()
        accepts_in_channels = any(
            parameter.name == "in_channels"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in encoder_parameters
        )
        observation_channels = self.observation_shape[0]
        if accepts_in_channels:
            self.phi = encoder(in_channels=observation_channels)
        else:
            if observation_channels != 4:
                raise TypeError(
                    f"{encoder.__name__} must accept an `in_channels` argument "
                    f"to process {observation_channels}-channel observations."
                )
            self.phi = encoder()

        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()

        self.action_dimension = action_dimension
        self.embedding_dimension = embedding_dimension
        dummy_input = self.__as_channels_last(
            torch.randn(2, *self.observation_shape)
        )
        with torch.inference_mode():
            self.feature_dimension = self.phi(dummy_input).flatten(1).size(1)

    def __as_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        if not self.channels_last or x.ndim != 4:
            return x
        return x.contiguous(memory_format=torch.channels_last)

    def no_epsilon_greedy(self) -> None:
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__as_channels_last(x)
        x = self.normalize_observations(x)
        x = self.__as_channels_last(x)
        features = self.phi(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
