import torch
from ..modules import Stream
from .BaseNetwork import BaseNetwork


class DistributionalPQNNetwork(BaseNetwork):
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
        self.q_logits = Stream(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.embedding_dimension,
            output_dimension=self.action_dimension * self.distributional_bins,
        )

    def get_q_logits(self, states: torch.Tensor) -> torch.Tensor:
        features = self.get_features(states)
        logits = self.q_logits(features)
        return logits.reshape(
            states.shape[0],
            self.action_dimension,
            self.distributional_bins,
        )

    def get_q(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.get_q_logits(states)
        return self.hl_gauss_loss(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q(x)
