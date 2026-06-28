import torch
import random


class RandomGELUSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = random.choice([torch.nn.functional.gelu, torch.nn.functional.silu])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return torch.nn.functional.gelu(x)

        return self.gate(x)
