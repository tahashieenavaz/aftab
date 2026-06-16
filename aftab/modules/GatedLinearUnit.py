import torch
from aftab.constants import ModuleType


class GatedLinearUnit(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, activation: ModuleType):
        super().__init__()
        self.projection = torch.nn.Linear(dim_in, dim_hidden * 2)
        self.output = torch.nn.Linear(dim_hidden, dim_out)
        self.activation = activation()

    def forward(self, x):
        x1, x2 = self.projection(x).chunk(2, dim=-1)
        return self.output(self.activation(x1) * x2)
