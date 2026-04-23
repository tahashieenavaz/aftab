import torch
import math
import os
from typing import Type
from typing import Literal
from types import SimpleNamespace
from baloot import acceleration_device
from baloot import seed_everything
from .maps import encoders_map
from .maps import acceptable_frames_map
from .functions import flush
from .mixins import TrainingResultsMixin
from .mixins import EnvironmentMixin
from .mixins import ActionsMixin
from .mixins import EpsilonMixin
from .mixins import NetworkMixin
from .mixins import OptimizerMixin
from .mixins import QValueMixin
from .mixins import LossMixin
from .mixins import LambdaReturnsMixin
from .mixins import TrainMixin


class Aftab(
    TrainingResultsMixin,
    EnvironmentMixin,
    ActionsMixin,
    EpsilonMixin,
    NetworkMixin,
    OptimizerMixin,
    QValueMixin,
    LossMixin,
    LambdaReturnsMixin,
    TrainMixin,
):
    def __init__(
        self,
        *,
        encoder: str | Type[torch.nn.Module] = "gamma",
        network: Literal["q", "duelling", "fqf", "dfqf"] = "dfqf",
        frames: int | Literal["pilot", "full", "ablation"] = "pilot",
        frame_skip: int = 4,
        mini_batches: int = 32,
        epochs: int = 2,
        gamma: float = 0.999,
        lmbda: float = 0.65,
        lr: float = 2.5e-4,
        fraction_proposal_lr: float = 2.5e-9,
        train_environments: int = 128,
        test_environments: int = 8,
        steps_per_update: int = 32,
        min_cpu_count: int = 4,
        noop: int = 30,
        frame_stack: int = 4,
        gradient_norm: float = 10.0,
        verbose: bool = False,
        verbose_interval: int = 10,
        verbose_window: int = 10,
        number_quantiles: int = 32,
        embedding_dimension: int = 512,
        quantile_embedding_dimension: int = 512,
        optimizer_instance: Type[torch.nn.Module] = torch.optim.RAdam,
        optimizer_epsilon: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
        optimizer_first_beta: float = 0.9,
        optimizer_second_beta: float = 0.999,
        should_compile: bool = True,
        train_episodic_life: bool = True,
        test_episodic_life: bool = False,
        train_reward_clip: bool = True,
        test_reward_clip: bool = True,
        reward_centering: bool = True,
        reward_centering_beta: float = 0.01,
        random_shift: bool = True,
        random_shift_padding: int = 4,
        random_shift_k: int = 1,
        random_shift_m: int = 1,
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyperparameters(**params)
        self.__initialize_frames()
        self.__initialize_derived_attributes()
        self.__initialize_constants()
        self.__initialize__encoder()
        super().__init__()

        self.buffer = SimpleNamespace()

    def __initialize_hyperparameters(self, **hyperparameters):
        for key, value in hyperparameters.items():
            setattr(self, key, value)

        self.flush_verbose("Aftab hyper-parameters initialized.")

    def __initialize_frames(self):
        if not isinstance(self.frames, str):
            return

        try:
            self.frames = acceptable_frames_map[self.frames]
        except KeyError as exc:
            raise ValueError(
                f"Invalid value for `frames`: {self.frames!r}. "
                f"Expected one of {tuple(acceptable_frames_map)}."
            ) from exc

    def __initialize_derived_attributes(self):
        self.total_environments = int(self.train_environments + self.test_environments)
        self.batch_size = int(self.train_environments * self.steps_per_update)
        self.mini_batch_size = int(self.batch_size / self.mini_batches)
        self.actual_frames = int(self.frames / self.frame_skip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)

    def __initialize__encoder(self):
        if not isinstance(self.encoder, str):
            return

        try:
            self.encoder = encoders_map[self.encoder]
        except KeyError as exc:
            raise ValueError(
                f"Unknown encoder key: {self.encoder!r}. "
                f"Expected one of: {tuple(encoders_map.keys())}"
            ) from exc

        self.flush_verbose(f"Encoder {self.encoder.__name__} was initialized.")

    def __initialize_constants(self):
        self.device = acceleration_device()
        self.cpu_count = os.cpu_count() or 1

    def set_precision(self):
        torch.set_float32_matmul_precision("high")

    def set_seed(self, seed: int):
        seed_everything(seed)

    def set_buffer(self, key, value):
        setattr(self.buffer, key, value)

    def flush_results(self):
        self.results = SimpleNamespace()
        self.results.rewards = SimpleNamespace()
        self.results.rewards.train = []
        self.results.rewards.test = []
        self.results.loss = []
        self.results.duration = 0.0

    def flush_verbose(self, message: str):
        if self.verbose:
            flush(message=message)

    def train(self, *, environment: str, seed: int = 42):
        self.set_buffer("seed", seed)
        self.set_buffer("environment", environment)
        self.train_loop(environment=environment, seed=seed)
