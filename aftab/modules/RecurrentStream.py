import torch
from .Stream import Stream


class RecurrentStream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        stream_hidden_dimension: int,
        stream_output_dimension: int,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.recurrent = torch.nn.GRU(
            input_dimension, hidden_dimension, batch_first=True
        )
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=stream_output_dimension,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.recurrent(x)
        features = recurrent_output.mean(dim=1)
        return self.stream(features)
