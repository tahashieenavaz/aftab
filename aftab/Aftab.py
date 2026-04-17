import torch
import numpy
import math
import time
from typing import Type, Literal
from .functions import flush
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
):
    def __init__(
        self,
        encoder: str | Type[torch.nn.Module] = "gamma",
        network: Literal["regression", "duelling", "fqf"] = "regression",
        frames: int | Literal["pilot", "full", "ablation"] = "pilot",
        frame_skip: int = 4,
        num_minibatches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.65,
        lr: float = 0.00025,
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
        train_episodic_life: bool = True,
        train_reward_clip: bool = True,
        test_episodic_life: bool = False,
        test_reward_clip: bool = True,
        should_compile: bool = True,
        augmentation: Literal["all", "intensity", "shift"] = "all",
    ):
        self.frame_skip = frame_skip
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.num_minibatches = num_minibatches
        self.encoder = encoder
        self.num_train_environments = num_train_environments
        self.num_test_environments = num_test_environments
        self.total_environments = int(num_train_environments + num_test_environments)
        self.steps_per_update = steps_per_update
        self.batch_size = int(num_train_environments * steps_per_update)
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.frames = frames
        self.actual_frames = int(self.frames / self.frame_skip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)
        self.train_reward_clip = train_reward_clip
        self.train_episodic_life = train_episodic_life
        self.test_reward_clip = test_reward_clip
        self.test_episodic_life = test_episodic_life
        self.min_test_cpu_count = min_test_cpu_count
        self.noop = noop
        self.gradient_norm = gradient_norm
        self.log_interval = log_interval
        self.verbose = verbose
        self.should_compile = should_compile
        self.frame_stack = frame_stack
        self.optimizer_instance = optimizer_instance
        self.optimizer_epsilon = optimizer_epsilon
        self.optimizer_first_beta = optimizer_first_beta
        self.optimizer_second_beta = optimizer_second_beta
        self.optimizer_weight_decay = optimizer_weight_decay
        self.augmentation = augmentation
        self.network = network
        self.number_quantiles = number_quantiles
        self.quantile_embedding_dimension = quantile_embedding_dimension

    def train(self, environment, seed: int = 42):
        self.flush_results()
        self.set_precision()
        self.set_seed(seed)

        all_train_rewards = []
        all_test_rewards = []
        all_loss = []

        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        train_environment, test_environment = self.make_environments(
            environment=environment, seed=seed
        )
        action_dimension = train_environment.action_space.n
        self.prepare_network(action_dimension=action_dimension)

        observation_train, _ = train_environment.reset()
        observation_test, _ = test_environment.reset()
        observation = numpy.concatenate([observation_train, observation_test], axis=0)
        observation = torch.as_tensor(
            observation, dtype=torch.uint8, device=self.device
        )
        optimizer = self.make_optimizer()
        observation_shape = train_environment.observation_space.shape
        (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
        ) = self.make_batches(
            observation_shape=observation_shape, action_dimension=action_dimension
        )
        scaler = torch.amp.GradScaler("cuda")
        training_start_time = time.time()

        for update in range(1, self.total_updates + 1):
            self._network.eval()
            for step in range(self.steps_per_update):
                float_observations = observation.float()
                epsilon_value = self._network.epsilon.get(
                    frame_count,
                    self.actual_frames,
                )
                q_values = self.get_q_values(
                    float_observations=float_observations, gradient=False
                )
                actions = self.get_actions(
                    q_values=q_values, epsilon_value=epsilon_value
                )
                actions_train, actions_test = self.split_actions(actions)

                (
                    next_observation_train,
                    reward_train,
                    termination_train,
                    truncation_train,
                    info_train,
                ) = train_environment.step(actions_train)
                (
                    next_observation_test,
                    reward_test,
                    termination_test,
                    truncation_test,
                    info_test,
                ) = test_environment.step(actions_test)
                next_observation = numpy.concatenate(
                    [next_observation_train, next_observation_test], axis=0
                )
                rewards = numpy.concatenate([reward_train, reward_test], axis=0)
                terminations = numpy.concatenate(
                    [termination_train, termination_test], axis=0
                )
                truncations = numpy.concatenate(
                    [truncation_train, truncation_test], axis=0
                )

                infos = {}
                for k, v_train in info_train.items():
                    if k not in info_test:
                        continue
                    v_test = info_test[k]

                    if isinstance(v_train, numpy.ndarray) and isinstance(
                        v_test, numpy.ndarray
                    ):
                        if v_train.ndim == 0:
                            infos[k] = numpy.stack([v_train, v_test])
                        else:
                            infos[k] = numpy.concatenate([v_train, v_test], axis=0)
                    else:
                        continue

                dones = numpy.logical_or(terminations, truncations)

                if "reward" in infos:
                    episode_returns += infos["reward"]

                if numpy.any(dones):
                    done_indices = numpy.where(dones)[0]
                    finished_scores = episode_returns[dones]
                    for idx, score in zip(done_indices, finished_scores):
                        if idx < self.num_train_environments:
                            all_train_rewards.append(score)
                        else:
                            all_test_rewards.append(score)
                    episode_returns[dones] = 0

                batch_observations[step] = observation
                batch_actions[step] = torch.as_tensor(actions, device=self.device)
                batch_rewards[step] = torch.as_tensor(rewards, device=self.device)
                batch_terminations[step] = torch.as_tensor(dones, device=self.device)
                batch_q[step] = q_values

                observation = torch.as_tensor(
                    next_observation, dtype=torch.uint8, device=self.device
                )
                frame_count += self.num_train_environments

            targets = self.get_returns(
                float_observations=observation.float(),
                batch_q=batch_q,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
            )
            flattened_observations = batch_observations[
                :, : self.num_train_environments
            ].reshape((-1,) + observation_shape)
            flattened_actions = batch_actions[:, : self.num_train_environments].reshape(
                -1
            )
            flattened_targets = targets[:, : self.num_train_environments].reshape(-1)

            self._network.train()
            total_loss = 0.0

            for _ in range(self.epochs):
                indices = torch.randperm(self.batch_size, device=self.device)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]
                    mini_batch_observations = flattened_observations[mb_idx]
                    mini_batch_actions = flattened_actions[mb_idx]
                    mini_batch_targets = flattened_targets[mb_idx]

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.get_loss(
                        mini_batch_observations=mini_batch_observations,
                        mini_batch_targets=mini_batch_targets,
                        mini_batch_actions=mini_batch_actions,
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(), self.gradient_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()
                    all_loss.append(loss.item())

            avg_loss = total_loss / (self.epochs * self.num_minibatches)
            test_score = (
                0.0
                if len(all_test_rewards) < 10
                else numpy.mean(all_test_rewards[-10:])
            )

            if self.verbose and update % self.log_interval == 0:
                flush(f"Update {update} | Frames: {frame_count} | Loss: {avg_loss:.4f}")
                flush(
                    f"Test Score: {test_score:.4f}",
                )

        train_environment.close()
        test_environment.close()

        self.results.rewards.train = all_train_rewards
        self.results.rewards.test = all_test_rewards
        self.results.loss = all_loss
        self.results.duration = time.time() - training_start_time

        if self.verbose:
            flush(f"Training finished.")
