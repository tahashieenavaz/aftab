import torch


class QuantileStream(torch.nn.Module):
    def __init__(
        self, feature_dimension: int, action_dimension: int, number_quantiles: int
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            feature_dimension, action_dimension * number_quantiles
        )

    def forward(self, x, tau_hats):
        B, N = tau_hats.shape
        return self.linear(x).view(B, N, -1)
