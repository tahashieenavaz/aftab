import torch
from ..constants import ModuleType
from ..common import LinearEpsilon


class BaseNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        action_dimension: int,
        embedding_dimension: int,
        encoder: ModuleType,
        channels_last: bool,
    ):
        super().__init__()
        self.channels_last = channels_last
        self.phi = encoder()

        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()

        self.action_dimension = action_dimension
        self.embedding_dimension = embedding_dimension
        dummy_input = self.__as_channels_last(torch.randn(1, 4, 84, 84))
        with torch.no_grad():
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
