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
        heads: int = 8,
        encoders: int = 2,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.encoder_projection = torch.nn.Linear(input_dimension, hidden_dimension)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dimension,
                nhead=heads,
                batch_first=True,
            ),
            num_layers=encoders,
        )
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=stream_output_dimension,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_projection(x)
        x = self.encoder(x)
        features = x.mean(dim=1)
        return self.stream(features)
