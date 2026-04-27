import torch


class LayerNorm2d(torch.nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            eps=1e-6,
            affine=True,
        )
