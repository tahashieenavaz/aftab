import torch
from ..functions import lambda_returns


class LambdaReturnsMixin:
    def __init__(self):
        super().__init__()

    def get_returns(
        self,
        float_observations,
        batch_q,
        batch_rewards,
        batch_terminations,
        next_q_values=None,
    ):
        with torch.no_grad():
            if next_q_values is not None:
                next_q = next_q_values.max(dim=-1).values
            else:
                next_q = self._network(float_observations).max(dim=-1).values

            max_q_seq = batch_q.max(dim=-1).values
            q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])

            returns = lambda_returns(
                batch_rewards,
                batch_terminations,
                q_seq_for_lambda[1:],
                self.gamma,
                self.lmbda,
            )
        return returns
