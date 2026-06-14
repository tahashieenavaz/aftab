import torch
from aftab.modules import EncoderBlock
from aftab.constants import ModuleType

_DEFAULT_ACTIVATION = torch.nn.ReLU


class ZetaEncoder(torch.nn.Module):
    def __init__(
        self, *, activation: ModuleType = _DEFAULT_ACTIVATION, in_channels: int = 4
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            EncoderBlock(
                in_channels,
                48,
                kernel_size=4,
                stride=2,
                padding=1,
                activation=activation,
            ),
            EncoderBlock(
                48, 48, kernel_size=4, stride=2, padding=1, activation=activation
            ),
            EncoderBlock(
                48, 48, kernel_size=4, stride=2, padding=1, activation=activation
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
