import torch


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
