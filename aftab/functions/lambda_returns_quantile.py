import torch


@torch.jit.script
def lambda_returns_quantile(
    rewards,
    terminations,
    next_quantiles,
    gamma: float,
    lmbda: float,
):
    T = next_quantiles.size(0)
    output = torch.zeros_like(next_quantiles)
    not_done_last = (1.0 - terminations[-1]).unsqueeze(-1)
    accumulated = rewards[-1].unsqueeze(-1) + gamma * not_done_last * next_quantiles[-1]
    output[-1] = accumulated
    for t in range(T - 2, -1, -1):
        not_done = (1.0 - terminations[t]).unsqueeze(-1)
        bootstrap = next_quantiles[t]
        mix = (1.0 - lmbda) * bootstrap + lmbda * accumulated
        accumulated = rewards[t].unsqueeze(-1) + gamma * not_done * mix
        output[t] = accumulated
    return output
