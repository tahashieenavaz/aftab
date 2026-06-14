import torch
from aftab.modules import EncoderBlock
from aftab.constants import ModuleType

_DEFAULT_ACTIVATION = torch.nn.ReLU


class EtaEncoder(torch.nn.Module):
    def __init__(
        self, *, activation: ModuleType = _DEFAULT_ACTIVATION, in_channels: int = 4
    ):
        super().__init__()
        self.stream = torch.nn.Sequential(
            EncoderBlock(
                in_channels,
                64,
                kernel_size=4,
                stride=4,
                padding=0,
                activation=activation,
            ),
            EncoderBlock(
                64, 128, kernel_size=3, stride=1, padding=0, activation=activation
            ),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
