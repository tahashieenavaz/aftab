import torch
from aftab.modules import EncoderBlock
from aftab.constants import ModuleType


class EpsilonEncoder(torch.nn.Module):
    def __init__(self, *, activation: ModuleType = torch.nn.ReLU):
        super().__init__()
        self.stream = torch.nn.Sequential(
            EncoderBlock(
                4, 32, kernel_size=3, stride=2, padding=1, activation=activation
            ),
            EncoderBlock(
                32, 48, kernel_size=3, stride=2, padding=1, activation=activation
            ),
            EncoderBlock(
                48, 64, kernel_size=3, stride=2, padding=0, activation=activation
            ),
            EncoderBlock(
                64, 64, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
