import torch
from ..constants import ModuleType
from ..common import LinearEpsilon
from ..modules.augmentation import RandomShift, ColorIntensity


class BaseNetwork(torch.nn.Module):
    def __init__(
        self, *, action_dimension: int, augmentation: str, encoder: ModuleType
    ):
        super().__init__()

        self.phi = encoder()
        self.epsilon_greedy = True
        self.epsilon = LinearEpsilon()
        self.action_dimension = action_dimension

        if augmentation == "all":
            self.chi = torch.nn.Sequential(RandomShift(), ColorIntensity())
        elif augmentation == "intensity":
            self.chi = ColorIntensity()
        elif augmentation == "shift":
            self.chi = RandomShift()
        elif augmentation in ["none", "off"]:
            self.chi = torch.nn.Identity()
        else:
            raise ValueError(
                f"Augmentation pipeline expected among all, intensity, shift, none, off. Got {augmentation}."
            )

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    def get_features(self, x: torch.Tensor, augment: bool = True) -> torch.Tensor:
        x = self.normalize_observations(x)

        if augment:
            x = self.chi(x)

        features = self.phi(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
