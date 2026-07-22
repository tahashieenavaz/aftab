import torch
from typing import Optional
from aftab.typing import ModuleType


class Stream(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: ModuleType,
        normalization: bool,
        perturbation_std: Optional[float] = None,
    ):
        super().__init__()
        if perturbation_std is not None and perturbation_std < 0.0:
            raise ValueError("Expected `perturbation_std` to be non-negative.")

        self.first_linear = torch.nn.Linear(input_dimension, hidden_dimension)
        self.second_linear = torch.nn.Linear(hidden_dimension, output_dimension)
        self.activation = activation()
        self.normalization_layer = (
            torch.nn.LayerNorm(hidden_dimension)
            if normalization
            else torch.nn.Identity()
        )
        self.register_buffer(
            "perturbation_std",
            None if perturbation_std is None else torch.tensor(float(perturbation_std)),
        )
        self.register_buffer(
            "first_input_perturbation",
            None if perturbation_std is None else torch.empty(input_dimension),
            persistent=False,
        )
        self.register_buffer(
            "first_output_perturbation",
            None if perturbation_std is None else torch.empty(hidden_dimension),
            persistent=False,
        )
        self.register_buffer(
            "second_input_perturbation",
            None if perturbation_std is None else torch.empty(hidden_dimension),
            persistent=False,
        )
        self.register_buffer(
            "second_output_perturbation",
            None if perturbation_std is None else torch.empty(output_dimension),
            persistent=False,
        )
        if perturbation_std is not None:
            self.resample_perturbation()

    @torch.no_grad()
    def resample_perturbation(self) -> None:
        if self.perturbation_std is None:
            return

        perturbations = (
            self.first_input_perturbation,
            self.first_output_perturbation,
            self.second_input_perturbation,
            self.second_output_perturbation,
        )
        for perturbation in perturbations:
            perturbation.normal_()
            perturbation.div_(perturbation.square().mean().sqrt().clamp_min(1e-12))

    @torch.no_grad()
    def set_perturbation_std(self, std: float) -> None:
        if self.perturbation_std is None:
            raise RuntimeError("This stream was created without weight perturbation.")
        if std < 0.0:
            raise ValueError(
                "Expected perturbation standard deviation to be non-negative."
            )
        self.perturbation_std.fill_(std)

    def _linear_with_perturbation(
        self,
        x: torch.Tensor,
        linear: torch.nn.Linear,
        input_perturbation: torch.Tensor,
        output_perturbation: torch.Tensor,
    ) -> torch.Tensor:
        output = linear(x)
        projected_noise = torch.matmul(x, input_perturbation).unsqueeze(-1)
        return output + (self.perturbation_std * projected_noise * output_perturbation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.perturbation_std is None:
            x = self.first_linear(x)
        else:
            x = self._linear_with_perturbation(
                x,
                self.first_linear,
                self.first_input_perturbation,
                self.first_output_perturbation,
            )
        x = self.normalization_layer(x)
        x = self.activation(x)
        if self.perturbation_std is None:
            x = self.second_linear(x)
        else:
            x = self._linear_with_perturbation(
                x,
                self.second_linear,
                self.second_input_perturbation,
                self.second_output_perturbation,
            )
        return x
