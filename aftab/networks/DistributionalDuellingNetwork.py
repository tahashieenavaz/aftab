import torch
from ..modules import Stream
from .BaseNetwork import BaseNetwork


class DistributionalDuellingNetwork(BaseNetwork):
    def __init__(
        self,
        *,
        distributional_bins: int,
        distributional_min_value: float,
        distributional_max_value: float,
        distributional_sigma: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from hl_gauss_pytorch import HLGaussLoss

        self.distributional = True
        self.distributional_bins = distributional_bins
        self.hl_gauss_loss = HLGaussLoss(
            min_value=distributional_min_value,
            max_value=distributional_max_value,
            num_bins=distributional_bins,
            sigma=distributional_sigma,
            clamp_to_range=True,
        )
        self.advantage_logits = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=self.action_dimension * self.distributional_bins,
            normalization=True,
        )
        self.value_logits = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=self.distributional_bins,
            normalization=True,
        )

    def get_value_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.value_logits(features)
        return logits.unsqueeze(1)

    def get_advantage_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.advantage_logits(features)
        logits = logits.reshape(
            features.shape[0],
            self.action_dimension,
            self.distributional_bins,
        )
        return logits - logits.mean(dim=1, keepdim=True)

    def get_q_logits(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        value_logits = self.get_value_logits(features=features)
        advantage_logits = self.get_advantage_logits(features=features)
        return value_logits + advantage_logits

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.get_q_logits(states)
        return self.hl_gauss_loss(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
