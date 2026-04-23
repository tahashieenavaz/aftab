import torch
from ..modules import Stream
from .BaseNetwork import BaseNetwork


class PQNNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.hidden_dimension,
            output_dimension=self.action_dimension,
        )

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        return self.q(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
