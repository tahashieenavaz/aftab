from __future__ import annotations

import torch
import math
import os
from typing import Literal
from types import SimpleNamespace
from baloot import acceleration_device
from baloot import seed_everything
from .constants import ModuleType
from .constants import EncoderStringType
from .constants import OptimizerStringType
from .maps import encoders_map
from .maps import acceptable_frames_map
from .functions import flush
from .modules import RandomShift
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
        encoder: ModuleType | EncoderStringType = "gammahadamaxv1",
        network: Literal[
            "q",
            "duelling",
            "distributional",
            "distributional-duelling",
        ] = "distributional-duelling",
        frames: int | Literal["pilot", "full", "ablation"] = "full",
        frame_skip: int = 4,
        mini_batches: int = 32,
        epochs: int = 2,
        gamma: float = 0.995,
        lmbda: float = 0.65,
        lr: float = 25e-5,
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
        embedding_dimension: int = 512,
        optimizer_instance: ModuleType | OptimizerStringType = torch.optim.RAdam,
        optimizer_epsilon: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
        optimizer_first_beta: float = 0.9,
        optimizer_second_beta: float = 0.999,
        should_compile: bool = True,
        autocast_float16: bool = True,
        train_episodic_life: bool = True,
        test_episodic_life: bool = False,
        train_reward_clip: bool = True,
        test_reward_clip: bool = True,
        reward_centering: bool = True,
        reward_centering_beta: float = 0.01,
        random_shift: bool = True,
        random_shift_padding: int = 4,
        v_min: float = -10.0,
        v_max: float = 10.0,
        bins: int = 51,
        hl_gauss_smoothing_ratio: float = 0.75,
    ):
        params = locals()
        params.pop("self")
        self.__initialize_hyperparameters(**params)
        self.__initialize_frames()
        self.__initialize_derived_attributes()
        self.__initialize_constants()
        self.__initialize__encoder()
        self.__initialize_augmentation_pipeline()
        super().__init__()
        self.buffer = SimpleNamespace()

    def __initialize_hyperparameters(self, **hyperparameters):
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def __initialize_augmentation_pipeline(self):
        self.augmentation_pipeline = RandomShift(padding=self.random_shift_padding)

    # TODO
    def __initialize_optimizer(self):
        pass

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

        self.flush_verbose(f"Encoder: {self.encoder.__name__}")

    def __initialize_constants(self):
        self.device = acceleration_device()
        self.cpu_count = os.cpu_count() or 1
        self.flush_verbose(f"Acceleration device: {self.device}")
        self.flush_verbose(f"CPU Count: {self.cpu_count}")

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

    def train(self, *, environment: str, seed: int):
        self.flush_verbose(f"Environment: {environment}")
        self.flush_verbose(f"Seed: {seed}")
        self.set_buffer("seed", seed)
        self.set_buffer("environment", environment)
        self.train_loop(environment=environment, seed=seed)
