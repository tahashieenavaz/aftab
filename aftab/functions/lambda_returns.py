import torch


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
