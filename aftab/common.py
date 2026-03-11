import torch
import numpy as np
from baloot import acceleration_device


class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.ln = torch.nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


@torch.jit.script
def lambda_returns(
    rew,
    done,
    next_q,
    gamma: float,
    lmbda: float,
):
    T = rew.size(0)
    out = torch.zeros_like(rew)
    ret = rew[-1] + gamma * next_q[-1] * (1 - done[-1])
    out[-1] = ret
    for t in range(T - 2, -1, -1):
        bootstrap = next_q[t]
        td_target = rew[t] + gamma * (1 - done[t]) * bootstrap
        ret = td_target + gamma * lmbda * (1 - done[t]) * (ret - bootstrap)
        out[t] = ret
    return out


def epsilon_greedy_vectorized(q_values, eps):
    if isinstance(eps, (np.ndarray, list)):
        eps = torch.tensor(eps, device=q_values.device, dtype=torch.float32)
    elif isinstance(eps, (float, int)):
        eps = torch.tensor([eps], device=q_values.device, dtype=torch.float32)
    elif isinstance(eps, torch.Tensor):
        eps = eps.to(q_values.device)

    if eps.ndim == 1 and eps.shape[0] != q_values.shape[0]:
        eps = eps[0]

    num_envs, action_dim = q_values.shape
    greedy_actions = torch.argmax(q_values, dim=-1)
    random_actions = torch.randint(0, action_dim, (num_envs,), device=q_values.device)

    mask = torch.rand(num_envs, device=q_values.device) < eps
    final_actions = torch.where(mask, random_actions, greedy_actions)
    return final_actions.cpu().numpy()


class LinearEpsilon:
    def __init__(self, ratio: float = 0.1, target=0.001):
        self.top = 1.0
        self.target = target
        self.ratio = ratio

    def get(self, frames, total_frames, all_rewards, episode_returns):
        target = self.target
        top = self.top
        decay_duration = total_frames * self.ratio
        if decay_duration == 0:
            return top
        return max(target, top - (frames / decay_duration) * (top - target))


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        # this is just here to keep the consistency. it doesn't do anything in this block.
        input_dim: int = 3136,
        hidden_dim: int = 512,
        output_dim,
        activation: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.stream(features)


def mse_loss(A, B):
    _device = acceleration_device()
    A = A.to(_device)
    B = B.to(_device)
    return 0.5 * torch.nn.functional.mse_loss(A, B)
