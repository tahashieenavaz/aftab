import torch
from typing import Type
from ..encoders import DQNEncoder
from ..common import LinearEpsilon, mse_loss
from ..modules import Stream


class PQNAgent(torch.nn.Module):
    def __init__(
        self, action_dimension, encoder_instance: Type[torch.nn.Module] = DQNEncoder
    ):
        super().__init__()
        self.phi = encoder_instance()
        self.q = Stream(output_dim=action_dimension)
        self.epsilon = LinearEpsilon()
        self.epsilon_greedy = True

    def no_epsilon_greedy(self):
        self.epsilon_greedy = False

    def get_features(self, x):
        x = x / 255.0
        return self.phi(x)

    def get_q(self, states):
        features = self.get_features(states)
        return self.q(features)

    def loss(self, q, target):
        return mse_loss(q, target)

    def forward(self, x):
        return self.get_q(x)
