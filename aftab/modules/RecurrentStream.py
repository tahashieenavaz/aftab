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
        heads: int = 8,
        num_encoder_layers: int = 2,
        normalization: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.normalization = normalization
        self.encoder_projection = torch.nn.Linear(input_dimension, hidden_dimension)
        self.encoder = self.__create_encoder(
            dimension=hidden_dimension,
            heads=heads,
            batch_first=batch_first,
            num_layers=num_encoder_layers,
        )
        self.stream = Stream(
            input_dimension=hidden_dimension,
            hidden_dimension=stream_hidden_dimension,
            output_dimension=output_dimension,
            normalization=normalization,
        )

    def __create_encoder_layer(
        self, *, heads: int, dimension: int, batch_first: bool = True
    ):
        return torch.nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            batch_first=batch_first,
        )

    def __create_encoder(
        self, *, dimension: int, heads: int, batch_first: bool, num_layers: int
    ):
        return torch.nn.TransformerEncoder(
            self.__create_encoder_layer(
                dimension=dimension, heads=heads, batch_first=batch_first
            ),
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_projection(x)
        x = self.encoder(x)
        features = x.mean(dim=1)
        return self.stream(features)
