import numpy
import torch


def epsilon_greedy_vectorized(q_values, eps):
    if isinstance(eps, (numpy.ndarray, list)):
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
    return final_actions
