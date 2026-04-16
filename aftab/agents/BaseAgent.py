import torch
from ..constants import ModuleType
from ..common import LinearEpsilon
from ..functions import mse_loss


class BaseAgent(torch.nn.Module):
    def __init__(self, *, action_dimension: int, encoder: ModuleType):
        self.epsilon_greedy = True
        self.action_dimension = action_dimension

        self.epsilon = LinearEpsilon()
        self.phi = encoder()

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_observations(x)
        features = self.phi(x)
        return features

    def loss(self, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_loss(q, target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
