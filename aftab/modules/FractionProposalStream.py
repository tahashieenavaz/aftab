import torch
from .Stream import Stream


class FractionProposalStream(torch.nn.Module):
    def __init__(
        self,
        *,
        features_dimension: int,
        number_quantiles: int = 32,
        embedding_dimension: int = 256,
    ):
        super().__init__()
        self.mu = Stream(
            input_dimension=features_dimension,
            output_dimension=number_quantiles,
            hidden_dimension=embedding_dimension,
        )

    def forward(self, features):
        q_logits = self.mu(features)
        q_probs = torch.nn.functional.softmax(q_logits, dim=-1)
        tau_0 = torch.zeros((features.size(0), 1), device=features.device)
        tau_1_to_N = torch.cumsum(q_probs, dim=-1)
        tau = torch.cat([tau_0, tau_1_to_N], dim=-1)
        tau_hat = (tau[:, :-1] + tau[:, 1:]) / 2.0
        entropy = -torch.sum(q_probs * torch.log(q_probs + 1e-8), dim=-1, keepdim=True)
        return tau, tau_hat, q_probs, entropy
