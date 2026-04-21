import torch
from ..constants import ModuleType
from ..common import LinearEpsilon
from ..modules.augmentation import RandomShift, ColorIntensity


class BaseNetwork(torch.nn.Module):
    def __init__(
        self, *, action_dimension: int, augmentation: str, encoder: ModuleType
    ):
        super().__init__()

        self.epsilon_greedy = True
        self.action_dimension = action_dimension
        self.epsilon = LinearEpsilon()

        if augmentation == "all":
            self.phi = torch.nn.Sequential(RandomShift(), ColorIntensity(), encoder())
        elif augmentation == "intensity":
            self.phi = torch.nn.Sequential(ColorIntensity(), encoder())
        elif augmentation == "shift":
            self.phi = torch.nn.Sequential(RandomShift(), encoder())
        elif augmentation in ["none", "off"]:
            self.phi = encoder()
        else:
            raise ValueError(
                f"Augmentation pipeline expected among all, intensity, shift, none, off. Got {augmentation}."
            )

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_observations(x)
        features = self.phi(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
