import torch


class ColorIntensity(torch.nn.Module):
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if not self.training:
            return x
        noise = 1.0 + self.scale * torch.randn((x.shape[0], 1, 1, 1), device=x.device)
        return x * noise
