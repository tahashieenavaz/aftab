import torch
from .Stream import Stream


class FractionProposalStream(torch.nn.Module):
    def __init__(
        self,
        *,
        number_quantiles: int,
        embedding_dimension: int,
        probability_cap: float = 0.98,
    ):
        super().__init__()
        self.number_quantiles = number_quantiles
        if probability_cap is not None and not (0.0 < probability_cap <= 1.0):
            raise ValueError("`probability_cap` must be in (0, 1].")
        self.probability_cap = probability_cap
        self.stream = Stream(
            input_dimension=None,
            embedding_dimension=embedding_dimension,
            output_dimension=number_quantiles,
            normalization=True,
        )

    def forward(self, features):
        q_logits = self.stream(features)
        q_probs = torch.nn.functional.softmax(q_logits, dim=-1)
        if self.probability_cap is not None:
            q_probs = q_probs.clamp(max=self.probability_cap)
            q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q_log_probabilities = torch.log(q_probs.clamp(min=1e-8))
        tau_0 = torch.zeros((features.size(0), 1), device=features.device)
        tau_1_to_N = torch.cumsum(q_probs, dim=-1)
        tau = torch.cat([tau_0, tau_1_to_N], dim=-1)
        tau_hat = (tau[:, :-1] + tau[:, 1:]) / 2.0
        entropy = -torch.sum(
            q_probs * q_log_probabilities,
            dim=-1,
            keepdim=True,
        )
        return tau, tau_hat, q_probs, entropy
