import numpy
import torch


def epsilon_greedy_vectorized(q_values, eps):
    if isinstance(eps, numpy.ndarray):
        if eps.dtype != numpy.float32:
            eps = eps.astype(numpy.float32, copy=False)
        if not eps.flags.c_contiguous:
            eps = numpy.ascontiguousarray(eps)
        eps = torch.from_numpy(eps).to(
            device=q_values.device,
            non_blocking=q_values.device.type == "cuda",
        )
    elif isinstance(eps, list):
        eps = torch.from_numpy(numpy.asarray(eps, dtype=numpy.float32)).to(
            device=q_values.device,
            non_blocking=q_values.device.type == "cuda",
        )
    elif isinstance(eps, (float, int, numpy.number)):
        eps = float(eps)
    elif isinstance(eps, torch.Tensor):
        eps = eps.to(q_values.device)

    num_envs, action_dim = q_values.shape
    if isinstance(eps, float):
        if eps <= 0.0:
            return torch.argmax(q_values, dim=-1)
        if eps >= 1.0:
            return torch.randint(0, action_dim, (num_envs,), device=q_values.device)

    if (
        isinstance(eps, torch.Tensor)
        and eps.ndim == 1
        and eps.shape[0] != q_values.shape[0]
    ):
        eps = eps[0]

    greedy_actions = torch.argmax(q_values, dim=-1)
    random_actions = torch.randint(0, action_dim, (num_envs,), device=q_values.device)

    mask = torch.rand(num_envs, device=q_values.device) < eps
    final_actions = torch.where(mask, random_actions, greedy_actions)
    return final_actions
