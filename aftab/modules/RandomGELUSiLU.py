import torch
import random


class RandomGELUSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = random.choice([torch.nn.functional.gelu, torch.nn.functional.silu])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)
