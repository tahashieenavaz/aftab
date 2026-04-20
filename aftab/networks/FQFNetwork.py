import torch
from typing import Dict
from .BaseNetwork import BaseNetwork
from ..modules import QuantileStream, FractionProposalStream


class FQFNetwork(BaseNetwork):
    def __init__(
        self,
        quantile_embedding_dimension: int,
        number_quantiles: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fraction_proposal = FractionProposalStream(
            number_quantiles=number_quantiles,
            embedding_dimension=quantile_embedding_dimension,
        )
        self.quantile_value = QuantileStream(
            action_dimension=kwargs["action_dimension"],
            embedding_dimension=quantile_embedding_dimension,
        )

    def forward(self, x: torch.Tensor) -> Dict:
        features = self.get_features(x)
        taus, tau_hats, q_probs, entropy = self.fraction_proposal(features)
        quantiles = self.quantile_value(features, tau_hats)
        return {
            "features": features,
            "quantiles": quantiles,
            "taus": taus,
            "tau_hats": tau_hats,
            "q_probs": q_probs,
            "entropy": entropy,
        }

    def get_q(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        quantiles = output["quantiles"]
        q_probs = output["q_probs"]
        return (quantiles * q_probs.unsqueeze(-1)).sum(dim=1)
