import torch
from typing import Optional
from ..functions import mse_loss


class LossMixin:
    def __init__(self):
        super().__init__()

    def __get_distributional_logits_taken(
        self,
        *,
        q_logits: torch.Tensor,
        mini_batch_actions: torch.Tensor,
    ) -> torch.Tensor:
        action_indices = mini_batch_actions.reshape(-1, 1, 1).expand(
            -1,
            1,
            q_logits.shape[-1],
        )
        return q_logits.gather(1, action_indices).squeeze(1)

    def __get_distributional_loss(
        self,
        *,
        mini_batch_observations: torch.Tensor,
        mini_batch_actions: torch.Tensor,
        mini_batch_targets: torch.Tensor,
        mini_batch_old_q_values: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q_logits = self._network.get_q_logits(mini_batch_observations)
        q_logits_taken = self.__get_distributional_logits_taken(
            q_logits=q_logits,
            mini_batch_actions=mini_batch_actions,
        )
        hl_gauss = self._network.hl_gauss_loss
        loss = hl_gauss(q_logits_taken, mini_batch_targets, reduction="none")

        value_clip = float(getattr(self, "distributional_value_clip", 0.0))
        if mini_batch_old_q_values is None or value_clip <= 0.0:
            return loss.mean()

        scalar_q_taken = hl_gauss(q_logits_taken)
        scalar_q_clipped = mini_batch_old_q_values + (
            scalar_q_taken - mini_batch_old_q_values
        ).clamp(-value_clip, value_clip)
        q_clipped_logprobs = hl_gauss.transform_to_logprobs(scalar_q_clipped)
        clipped_loss = hl_gauss(
            q_clipped_logprobs,
            mini_batch_targets,
            reduction="none",
        )
        return torch.max(loss, clipped_loss).mean()

    def get_loss(
        self,
        mini_batch_observations,
        mini_batch_actions,
        mini_batch_targets,
        mini_batch_old_q_values=None,
    ) -> torch.Tensor:
        mini_batch_observations = mini_batch_observations.float()
        if bool(getattr(self._network, "distributional", False)):
            return self.__get_distributional_loss(
                mini_batch_observations=mini_batch_observations,
                mini_batch_actions=mini_batch_actions,
                mini_batch_targets=mini_batch_targets,
                mini_batch_old_q_values=mini_batch_old_q_values,
            )

        q_values = self._network.get_q(mini_batch_observations)
        q_taken = q_values.gather(1, mini_batch_actions.unsqueeze(1)).squeeze(1)
        return mse_loss(q_taken, mini_batch_targets)
