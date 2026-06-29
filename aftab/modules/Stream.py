import torch
import torch.nn.functional as F
import inspect
from aftab.constants import ModuleType


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType = torch.nn.ReLU,
        normalization: bool = True,
    ):
        super().__init__()
        self.normalization = normalization

        self.first_linear = torch.nn.Linear(input_dimension, hidden_dimension)
        self.second_linear = torch.nn.Linear(hidden_dimension, output_dimension)

        self.activation = self.__initiate_activation(
            activation_class=activation, channels=hidden_dimension
        )

        if normalization:
            self.normalization_layer = torch.nn.LayerNorm(hidden_dimension)

    def __initiate_activation(
        self, activation_class: ModuleType, channels: int
    ) -> torch.nn.Module:
        activation_args = {}
        args = inspect.signature(activation_class).parameters

        if "inplace" in args:
            activation_args.update({"inplace": True})

        if "channels" in args:
            activation_args.update({"channels": channels})
        elif "in_channels" in args:
            activation_args.update({"in_channels": channels})

        return activation_class(**activation_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_linear(x)
        if self.normalization:
            x = self.normalization_layer(x)
        x = self.activation(x)
        x = self.second_linear(x)
        return x


def _can_parallelize_stream_heads(heads: torch.nn.ModuleList) -> bool:
    first_head = heads[0]
    if not isinstance(first_head, Stream):
        return False

    normalization = first_head.normalization
    hidden_dimension = first_head.first_linear.out_features
    input_dimension = first_head.first_linear.in_features
    output_dimension = first_head.second_linear.out_features

    for head in heads:
        if not isinstance(head, Stream):
            return False
        if head.normalization != normalization:
            return False
        if head.first_linear.in_features != input_dimension:
            return False
        if head.first_linear.out_features != hidden_dimension:
            return False
        if head.second_linear.in_features != hidden_dimension:
            return False
        if head.second_linear.out_features != output_dimension:
            return False

    return True


def _apply_stream_head_activations(
    x: torch.Tensor,
    heads: torch.nn.ModuleList,
) -> torch.Tensor:
    activation = heads[0].activation
    same_activation = all(
        type(head.activation) is type(activation)
        and repr(head.activation) == repr(activation)
        and len(list(head.activation.parameters())) == 0
        and len(list(head.activation.buffers())) == 0
        for head in heads
    )
    if same_activation:
        return activation(x)

    activated = torch.empty_like(x)
    for index, head in enumerate(heads):
        activated[:, index] = head.activation(x[:, index].clone())
    return activated


def forward_stream_heads(
    heads: torch.nn.ModuleList,
    x: torch.Tensor,
) -> torch.Tensor:
    if len(heads) == 0:
        raise ValueError("Expected at least one stream head.")

    if not _can_parallelize_stream_heads(heads=heads):
        return torch.stack([head(x) for head in heads], dim=1)

    first_weights = torch.stack(
        [head.first_linear.weight for head in heads],
        dim=0,
    )
    first_biases = torch.stack(
        [head.first_linear.bias for head in heads],
        dim=0,
    )
    x = torch.einsum("bi,hoi->bho", x, first_weights)
    x = x + first_biases.unsqueeze(0)

    if heads[0].normalization:
        eps = heads[0].normalization_layer.eps
        x = F.layer_norm(x, (x.size(-1),), eps=eps)
        weights = torch.stack(
            [head.normalization_layer.weight for head in heads],
            dim=0,
        )
        biases = torch.stack(
            [head.normalization_layer.bias for head in heads],
            dim=0,
        )
        x = x * weights.unsqueeze(0) + biases.unsqueeze(0)

    x = _apply_stream_head_activations(x=x, heads=heads)

    second_weights = torch.stack(
        [head.second_linear.weight for head in heads],
        dim=0,
    )
    second_biases = torch.stack(
        [head.second_linear.bias for head in heads],
        dim=0,
    )
    x = torch.einsum("bhi,hoi->bho", x, second_weights)
    return x + second_biases.unsqueeze(0)
