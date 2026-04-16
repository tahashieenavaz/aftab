import torch
from ..constants import ModuleType
from ..encoders import NatureDQNEncoder
from ..modules import Stream
from .BaseAgent import BaseAgent


class DuellingAgent(BaseAgent):
    def __init__(
        self,
        action_dimension: int,
        encoder: ModuleType = NatureDQNEncoder,
    ):
        super().__init__(encoder=encoder)
        self.q = Stream(output_dimension=action_dimension)

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        return self.q(features)
