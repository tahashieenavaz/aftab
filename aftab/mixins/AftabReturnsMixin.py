import torch
from aftab.functions import lambda_returns
from .AftabBaseMixin import AftabBaseMixin


class AftabReturnsMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def get_returns(
        self,
        *,
        batch_rewards: torch.Tensor,
        batch_terminations: torch.Tensor,
        next_q: torch.Tensor,
    ) -> torch.Tensor:
        return lambda_returns(
            rewards=batch_rewards,
            terminations=batch_terminations,
            next_q=next_q,
            gamma=getattr(self, "gamma"),
            return_lambda=getattr(self, "return_lambda"),
        )
