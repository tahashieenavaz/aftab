import torch
import torch.nn.functional as F


class MixedActivation(torch.nn.Module):
    def __init__(self, hidden_dimension: int, dim: int = -1):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.dim = dim
        self.logits = torch.nn.Parameter(torch.zeros(hidden_dimension, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=-1)
        w_gelu, w_silu, w_relu = torch.unbind(weights, dim=-1)

        if self.dim != -1 and self.dim != (x.dim() - 1):
            shape = [1] * x.dim()
            shape[self.dim] = self.hidden_dimension
            w_gelu = w_gelu.view(*shape)
            w_silu = w_silu.view(*shape)
            w_relu = w_relu.view(*shape)

        out = F.gelu(x)
        out.mul_(w_gelu)
        temp_silu = F.silu(x)
        out.addcmul_(temp_silu, w_silu)
        temp_relu = F.relu(x)
        out.addcmul_(temp_relu, w_relu)
        return out
