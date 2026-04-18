import torch
import math
from typing import Type, Literal
from .mixins import (
    EncoderRefinementMixin,
    ConstantsMixin,
    TrainingResultsMixin,
    MatrixPrecisionMixin,
    ReproducibilityMixin,
    EnvironmentSetupMixin,
    ActionsMixin,
    BatchMixin,
    EpsilonMixin,
    NetworkMixin,
    OptimizerMixin,
    QValueMixin,
    CheckFramesMixin,
    LossMixin,
    LambdaReturnsMixin,
    TrainMixin,
)


class Aftab(
    EncoderRefinementMixin,
    ConstantsMixin,
    TrainingResultsMixin,
    EnvironmentSetupMixin,
    ReproducibilityMixin,
    MatrixPrecisionMixin,
    ActionsMixin,
    BatchMixin,
    EpsilonMixin,
    NetworkMixin,
    OptimizerMixin,
    QValueMixin,
    CheckFramesMixin,
    LossMixin,
    LambdaReturnsMixin,
    TrainMixin,
):
    def __init__(
        self,
        *,
        encoder: str | Type[torch.nn.Module] = "gamma",
        network: Literal["regression", "duelling", "fqf"] = "regression",
        frames: int | Literal["pilot", "full", "ablation"] = "pilot",
        augmentation: Literal["all", "intensity", "shift", "none"] = "all",
        q_value_iterations: int = 4,
        frame_skip: int = 4,
        num_minibatches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.65,
        lr: float = 2.5e-4,
        fraction_proposal_lr: float = 2.5e-9,
        num_train_environments: int = 128,
        num_test_environments: int = 8,
        steps_per_update: int = 32,
        min_test_cpu_count: int = 4,
        noop: int = 30,
        frame_stack: int = 4,
        gradient_norm: float = 10.0,
        log_interval: int = 10,
        number_quantiles: int = 32,
        quantile_embedding_dimension: int = 256,
        verbose: bool = False,
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
    ):
        params = locals()
        params.pop("self")
        self._init_hyperparameters(**params)
        super().__init__()
        self._calculate_derived_attributes()

    def _init_hyperparameters(self, **hyperparameters):
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def _calculate_derived_attributes(self):
        self.total_environments = int(
            self.num_train_environments + self.num_test_environments
        )
        self.batch_size = int(self.num_train_environments * self.steps_per_update)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.actual_frames = int(self.frames / self.frame_skip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)

    def train(self, environment: str, seed: int = 42):
        self._train_loop(environment=environment, seed=seed)
