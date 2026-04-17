import torch
from .BaseAgent import BaseAgent
from ..modules import QuantileStream, FractionProposalStream


class FQFAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fpn = QuantileStream()
        self.qvn = FractionProposalStream()

    def get_q(self, x):
        quantiles, _, _ = self.get_quantiles(x)
        return quantiles.mean(dim=1)

    def loss(self, predictions, targets, tau_hats):
        td_error = targets.unsqueeze(1) - predictions.unsqueeze(2)
        huber = torch.where(
            td_error.abs() < 1.0, 0.5 * td_error.pow(2), td_error.abs() - 0.5
        )
        tau = tau_hats.unsqueeze(2)
        loss = (torch.abs(tau - (td_error.detach() < 0).float()) * huber).mean()
        return loss
