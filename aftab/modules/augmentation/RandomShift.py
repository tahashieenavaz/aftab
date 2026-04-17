import torch


class RandomShift(torch.nn.Module):
    def __init__(self, padding: int = 4):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        n, c, h, w = x.shape
        device = x.device
        x = torch.nn.functional.pad(
            x, (self.padding, self.padding, self.padding, self.padding)
        )
        eps = 1.0 / (h + 2 * self.padding)
        arange = torch.linspace(-1 + eps, 1 - eps, h, device=device)
        base_grid = torch.stack(torch.meshgrid(arange, arange, indexing="ij"), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.padding + 1, (n, 2), device=device)
        shift = shift * 2.0 / (h + 2 * self.padding)
        grid = base_grid + shift.view(n, 1, 1, 2)
        return torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
