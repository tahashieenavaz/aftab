import torch
from ..functions import lambda_returns


class TargetMixin:
    def __init__(self):
        super().__init__()

    def get_returns(
        self, float_observations, batch_q, batch_rewards, batch_terminations
    ):
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=torch.float16),
        ):
            next_q = self._network(float_observations).max(dim=-1).values
            max_q_seq = batch_q.max(dim=-1).values
            q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])
            targets = lambda_returns(
                batch_rewards,
                batch_terminations,
                q_seq_for_lambda[1:],
                self.gamma,
                self.lmbda,
            )
        return targets
