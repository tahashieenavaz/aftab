import torch


class FractionProposalStream(torch.nn.Module):
    def __init__(self, *, number_quantiles: int):
        super().__init__()
        self.number_quantiles = number_quantiles
        self.linear = torch.nn.LazyLinear(number_quantiles)

    def forward(self, features):
        q_logits = self.linear(features)
        q_probs = torch.nn.functional.softmax(q_logits, dim=-1)
        tau_0 = torch.zeros((features.size(0), 1), device=features.device)
        tau_1_to_N = torch.cumsum(q_probs, dim=-1)
        tau = torch.cat([tau_0, tau_1_to_N], dim=-1)
        tau_hat = (tau[:, :-1] + tau[:, 1:]) / 2.0
        inner_tau = tau[:, 1:-1]
        entropy = -torch.sum(inner_tau * torch.log(inner_tau), dim=-1)
        return tau, tau_hat, q_probs, entropy
