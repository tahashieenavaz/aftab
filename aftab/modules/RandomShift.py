import torch


class RandomShift(torch.nn.Module):
    def __init__(self, *, padding: int):
        super().__init__()
        if padding < 0:
            raise ValueError("Expected `padding` to be non-negative.")

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.padding] * 4)
        x = torch.nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.padding)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.padding, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.padding + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.padding)
        grid = base_grid + shift
        return torch.nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )
