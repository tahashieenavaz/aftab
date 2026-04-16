import torch
from typing import Type
from ..encoders import NatureDQNEncoder
from ..common import LinearEpsilon
from ..modules import Stream
from ..functions import mse_loss


class PQNAgent(torch.nn.Module):
    def __init__(
        self,
        action_dimension,
        encoder_instance: Type[torch.nn.Module] = NatureDQNEncoder,
    ):
        super().__init__()
        self.epsilon_greedy = True

        self.epsilon = LinearEpsilon()
        self.phi = encoder_instance()
        self.q = Stream(output_dim=action_dimension)

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def normalize_features(self, x):
        return x / 255.0

    def get_features(self, x):
        x = self.normalize_features(x)
        features = self.phi(x)
        return features

    def get_q(self, states):
        features = self.get_features(states)
        return self.q(features)

    def loss(self, q, target):
        return mse_loss(q, target)

    def forward(self, x):
        return self.get_q(x)
