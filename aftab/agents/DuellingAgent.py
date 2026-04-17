import torch
from ..constants import ModuleType
from ..encoders import NatureDQNEncoder
from ..modules import Stream
from .BaseAgent import BaseAgent


class PQNAgent(BaseAgent):
    def __init__(
        self,
        *,
        action_dimension: int,
        augmentation: bool,
        encoder: ModuleType = NatureDQNEncoder,
    ):
        super().__init__(
            action_dimension=action_dimension,
            encoder=encoder,
            augmentation=augmentation,
        )
        self.advantage = Stream(output_dimension=action_dimension)
        self.value = Stream(output_dimension=1)

    def get_value(self, features):
        return self.value(features)

    def get_advantage(self, features):
        advantage = self.advantage(features)
        advantage = advantage - advantage.mean(dim=1)
        return advantage

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        value = self.get_value(features=features)
        advantage = self.get_advantage(features=features)
        return value + advantage
