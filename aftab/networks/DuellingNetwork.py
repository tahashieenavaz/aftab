import torch
from ..modules import Stream
from .BaseNetwork import BaseNetwork


class DuellingNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.advantage = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=self.action_dimension,
            normalization=True,
        )
        self.value = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=1,
            normalization=True,
        )

    def get_value(self, features):
        return self.value(features)

    def get_advantage(self, features):
        advantage = self.advantage(features)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return advantage

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        value = self.get_value(features=features)
        advantage = self.get_advantage(features=features)
        return value + advantage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
