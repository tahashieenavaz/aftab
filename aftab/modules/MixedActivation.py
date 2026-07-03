import torch
import torch.nn.functional as F


class MixedActivation(torch.nn.Module):
    def __init__(self, hidden_dimension: int):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(hidden_dimension, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=-1)
        w_gelu, w_silu, w_relu = weights[:, 0], weights[:, 1], weights[:, 2]

        if x.dim() > 1:
            broadcast_shape = [1, x.size(1)] + [1] * (x.dim() - 2)
        else:
            broadcast_shape = [x.size(0)]

        w_gelu = w_gelu.view(*broadcast_shape)
        w_silu = w_silu.view(*broadcast_shape)
        w_relu = w_relu.view(*broadcast_shape)

        return w_gelu * F.gelu(x) + w_silu * F.silu(x) + w_relu * F.relu(x)
