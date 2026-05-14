import torch


class HadamaxLayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2 * num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(2 * num_channels))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, total_channels, height, width = x.shape
        C = total_channels // 2
        reshaped = x.view(batch_size, 2, C, height, width)
        variance, mean = torch.var_mean(reshaped, dim=2, keepdim=True, correction=0)
        normalized = (reshaped - mean) / torch.sqrt(variance + self.epsilon)
        weight = self.weight.view(1, 2, C, 1, 1)
        bias = self.bias.view(1, 2, C, 1, 1)
        normalized = normalized * weight + bias
        return normalized.view(batch_size, total_channels, height, width)
