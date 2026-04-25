import torch
from ..functions import lambda_returns


class LambdaReturnsMixin:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
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
            lmbda=getattr(self, "lmbda"),
        )
