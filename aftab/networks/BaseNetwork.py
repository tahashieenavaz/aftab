import torch
from ..constants import ModuleType
from ..common import LinearEpsilon


class BaseNetwork(torch.nn.Module):
    def __init__(
        self, *, action_dimension: int, hidden_dimension: int, encoder: ModuleType
    ):
        super().__init__()
        self.phi = encoder()

        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()

        self.action_dimension = action_dimension
        self.hidden_dimension = hidden_dimension
        dummy_input = torch.randn(1, 4, 84, 84)
        with torch.no_grad():
            self.feature_dimension = self.phi(dummy_input).flatten(1).size(1)

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_observations(x)
        features = self.phi(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
