import torch
from typing import Type
from typing import Literal, Annotated, TypeAlias

ModuleType: TypeAlias = Type[torch.nn.Module]
EncoderStringType: TypeAlias = Annotated[str, "must be a valid encoder key"]
OptimizerStringType: TypeAlias = Literal["adam", "adamw" "radam", "nadam"]
NetworkStringType: TypeAlias = Literal[
    "q",
    "duelling",
    "bootstrapped",
    "bootstrapped-duelling",
    "distributional",
    "distributional-duelling",
    "distributional-bootstrapped-duelling",
    "distributional-bootstrapped-mixed-duelling",
]
