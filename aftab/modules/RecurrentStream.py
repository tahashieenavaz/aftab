import torch
from .Stream import Stream


class RecurrentStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        stream_hidden_dimension: int,
        num_layers: int = 2,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.recurrent = torch.nn.GRU(
            input_dimension, hidden_dimension, num_layers=num_layers, batch_first=True
        )
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.recurrent(x)
        features = x.mean(dim=1)
        return self.stream(features)
