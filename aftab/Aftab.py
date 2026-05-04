from __future__ import annotations

import torch
import math
import os
from .common import _make_sure_directory_exists
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
from .mixins import *


class Aftab(
    AftabOptimizerMixin,
    AftabTrainingResultsMixin,
    AftabEnvironmentMixin,
    AftabActionsMixin,
    AftabEpsilonMixin,
    AftabNetworkMixin,
    AftabQValueMixin,
    AftabLossMixin,
    AftabReturnsMixin,
    AftabTrainMixin,
):
    def __init__(
        self,
        *,
        encoder: ModuleType | EncoderStringType = "gammahadamaxv1",
        network: Literal[
            "q",
            "duelling",
            "bootstrapped",
            "bootstrapped-duelling",
            "distributional",
            "distributional-duelling",
        ] = "distributional-duelling",
        frames: int | Literal["pilot", "full", "ablation"] = "full",
        frame_skip: int = 4,
        mini_batches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        return_lambda: float = 0.65,
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
        verbose_window: int = 100,
        embedding_dimension: int = 512,
        optimizer: OptimizerStringType = "radam",
        optimizer_epsilon: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
        optimizer_first_beta: float = 0.9,
        optimizer_second_beta: float = 0.999,
        torch_compile: bool = True,
        autocast_float16: bool = True,
        channels_last: bool = True,
        train_episodic_life: bool = True,
        test_episodic_life: bool = False,
        train_reward_clip: bool = True,
        test_reward_clip: bool = True,
        distributional_bins: int = 51,
        distributional_min_value: float = -10.0,
        distributional_max_value: float = 10.0,
        distributional_sigma: float | None = None,
        distributional_sigma_ratio: float = 0.75,
        distributional_value_clip: float = 0.0,
        bootstrap_heads: int = 10,
        bootstrap_probability: float = 1.0,
    ):
        self.buffer = SimpleNamespace()

        params = locals()
        params.pop("self")
        self.__initialize_hyperparameters(**params)
        self.__initialize_frames()
        self.__initialize_derived_attributes()
        self.__initialize_constants()
        self.__initialize__encoder()
        super().__init__()

    def __initialize_hyperparameters(self, **hyperparameters):
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def __initialize_frames(self):
        if isinstance(self.frames, int):
            return

        if self.frames not in acceptable_frames_map:
            raise ValueError(f"Frames was provided a wrong value `{self.frames}`")

        self.frames = acceptable_frames_map[self.frames]

    def __initialize_derived_attributes(self):
        self.total_environments = int(self.train_environments + self.test_environments)
        self.batch_size = int(self.train_environments * self.steps_per_update)
        self.mini_batch_size = int(self.batch_size / self.mini_batches)
        self.actual_frames = int(self.frames / self.frame_skip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)

    def __initialize__encoder(self):
        if isinstance(self.encoder, str):
            if self.encoder not in encoders_map:
                raise ValueError(f"Encoder was provided a wrong value `{self.encoder}`")
            self.encoder = encoders_map[self.encoder]

        self.flush_verbose(f"Encoder: {self.encoder.__name__}")

    def __initialize_constants(self):
        self.device = acceleration_device()
        self.cpu_count = os.cpu_count() or 1
        self.flush_verbose(f"Acceleration device: {self.device}")
        self.flush_verbose(f"CPU Count: {self.cpu_count}")

    def __set_precision(self):
        torch.set_float32_matmul_precision("high")

    def __set_seed(self, seed: int):
        seed_everything(seed)

    def __set_buffer(self, key, value):
        setattr(self.buffer, key, value)

    def __flush_results(self):
        self.results = SimpleNamespace()
        self.results.rewards = SimpleNamespace()
        self.results.rewards.train = []
        self.results.rewards.test = []
        self.results.loss = []
        self.results.duration = 0.0

    def flush_verbose(self, message: str):
        if not self.verbose:
            return

        flush(message=message)

    def save(self, directory: str = "models"):
        directory_path = _make_sure_directory_exists(directory).strip("/").strip()

    def log(self, directory: str = "results") -> None:
        self._log(directory=directory)

    def train(self, *, environment: str, seed: int):
        self.__set_precision()
        self.__set_seed(seed)
        self.__flush_results()
        self.__set_buffer("seed", seed)
        self.__set_buffer("environment", environment)

        self.flush_verbose(f"Environment: {environment}")
        self.flush_verbose(f"Seed: {seed}")
        self._train(environment=environment, seed=seed)
