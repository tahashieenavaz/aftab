import torch
from ..modules import EncoderBlock
from ..constants import ModuleType


class DeltaEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            EncoderBlock(
                4, 24, kernel_size=9, stride=4, padding=0, activation=activation
            ),
            EncoderBlock(
                24, 48, kernel_size=5, stride=2, padding=0, activation=activation
            ),
            EncoderBlock(
                48, 96, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x):
        return self.stream(x)
