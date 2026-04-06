import torch
import numpy
import os
import math
import envpool
from baloot import acceleration_device, seed_everything, funnel
from typing import Type
from .maps import AftabMapEncoder
from .agents import PQNAgent
from .common import epsilon_greedy_vectorized, lambda_returns, flush


class Aftab:
    def __init__(
        self,
        encoder: str | Type[torch.nn.Module] = "gamma",
        frame_skip: int = 4,
        num_minibatches: int = 32,
        epochs: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.65,
        lr: float = 0.00025,
        num_train_environments: int = 128,
        num_test_environments: int = 8,
        steps_per_update: int = 32,
        total_frames: int = 200_000_000,
        min_test_cpu_count: int = 4,
        noop: int = 30,
        gradient_norm: float = 10.0,
        log_interval: int = 10,
        verbose: bool = False,
        optimizer_epsilon: float = 1e-5,
        train_episodic_life: bool = True,
        train_reward_clip: bool = True,
        test_episodic_life: bool = False,
        test_reward_clip: bool = True,
        optimizer_instance: Type[torch.nn.Module] = torch.optim.RAdam,
        should_compile: bool = False,
    ):
        self.device = acceleration_device()
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
        self.cpu_count = os.cpu_count()
        self.steps_per_update = steps_per_update
        self.batch_size = int(num_train_environments * steps_per_update)
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.actual_frames = int(total_frames / self.frame_skip)
        self.total_updates = math.ceil(self.actual_frames / self.batch_size)
        self.train_reward_clip = train_reward_clip
        self.train_episodic_life = train_episodic_life
        self.test_reward_clip = test_reward_clip
        self.test_episodic_life = test_episodic_life
        self.min_test_cpu_count = min_test_cpu_count
        self.noop = noop
        self.optimizer_epsilon = optimizer_epsilon
        self.gradient_norm = gradient_norm
        self.log_interval = log_interval
        self.verbose = verbose
        self.optimizer_instance = optimizer_instance
        self.should_compile = should_compile

        ######
        # this line ensures users can pass a string (predefined) or their defined encoder to the system.
        ######
        if isinstance(encoder, str):
            module = AftabMapEncoder.get(encoder)
            self.encoder = module

        ######
        # these will be filled right after the training is completed
        ######
        self.final_training_rewards = None
        self.final_test_rewards = None
        self.final_loss_evolution = None

    def make_network(
        action_dimension: int, encoder_instance: Type[torch.nn.Module]
    ) -> Type[torch.nn.Module]:
        return PQNAgent(
            action_dimension=action_dimension, encoder_instance=encoder_instance
        )

    def set_precision(self):
        torch.set_float32_matmul_precision("high")

    def set_seed(self, seed: int):
        seed_everything(seed)

    def make_environments(self, environment: str):
        train_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.num_train_environments,
            seed=self.seed,
            num_threads=self.cpu_count,
            thread_affinity_offset=0,
            noop_max=self.noop,
            reward_clip=self.reward_clip,
            episodic_life=self.episodic_life,
            frame_skip=self.frame_skip,
        )

        test_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.num_test_environments,
            seed=self.seed + 1000,
            num_threads=min(self.min_test_cpu_count, self.cpu_count),
            thread_affinity_offset=0,
            noop_max=self.noop,
            reward_clip=self.test_reward_clip,
            episodic_life=self.test_episodic_life,
            frame_skip=self.frame_skip,
        )

        return train_environment, test_environment

    def make_batches(self, observation_shape, action_dimension):
        batch_observations = torch.empty(
            (self.steps_per_update, self.total_environments) + observation_shape,
            dtype=torch.uint8,
            device=self._device,
        )
        batch_actions = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.int64,
            device=self.device,
        )
        batch_rewards = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        batch_terminations = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        batch_q = torch.empty(
            (self.steps_per_update, self.total_environments, action_dimension),
            dtype=torch.float32,
            device=self.device,
        )

        return (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
        )

    @torch.no_grad()
    def _perform_dummy_pass(self):
        self._network(torch.randn(1, 4, 84, 84).to(self.device))

    def train(self, environment, seed: int = 42):
        self.set_precision()
        self.set_seed(seed)
        all_train_rewards = []
        all_test_rewards = []
        all_loss = []
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        train_environment, test_environment = self.make_environments(environment)
        action_dimension = train_environment.action_space.n
        self._network = self.make_network(action_dimension, self.encoder)

        if self.should_compile:
            self._network = torch.compile(self._network)

        self._perform_dummy_pass()

        observation_train, _ = train_environment.reset()
        observation_test, _ = test_environment.reset()
        observation = numpy.concatenate([observation_train, observation_test], axis=0)
        observation = torch.as_tensor(
            observation, dtype=torch.uint8, device=self.device
        )
        optimizer = self.optimizer_instance(
            self._network.parameters(), lr=self.lr, eps=self.optimizer_epsilon
        )
        frame_count = 0
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

        for update in range(1, self.total_updates + 1):
            self._network.eval()
            for step in range(self.steps_per_update):
                float_observations = observation.float()
                training_epsilon_value = self._network.epsilon.get(
                    frame_count,
                    self.actual_frames,
                    all_train_rewards,
                    episode_returns[: self.num_train_environments],
                )
                epsilon_vector = torch.full(
                    (self.num_train_environments,),
                    training_epsilon_value,
                    device=self.device,
                    dtype=torch.float32,
                )
                test_epsilon_vector = torch.zeros(
                    (self.num_test_environments,),
                    device=self.device,
                    dtype=torch.float32,
                )
                full_epsilon_vector = torch.cat([epsilon_vector, test_epsilon_vector])

                with (
                    torch.no_grad(),
                    torch.autocast(device_type=self.device.type, dtype=torch.float16),
                ):
                    q_values = self._network(float_observations)
                    if self._network.epsilon_greedy:
                        actions = epsilon_greedy_vectorized(
                            q_values, full_epsilon_vector
                        )
                    else:
                        actions = q_values.argmax(dim=-1).cpu().numpy()

                act_train = actions[: self.num_train_environments]
                act_test = actions[self.num_train_environments :]

                (
                    next_observation_train,
                    reward_train,
                    termination_train,
                    truncation_train,
                    info_train,
                ) = train_environment.step(act_train)
                (
                    next_observation_test,
                    reward_test,
                    termination_test,
                    truncation_test,
                    info_test,
                ) = test_environment.step(act_test)
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
                        # Skip non-array keys (like nested dicts or raw python types)
                        continue

                dones = numpy.logical_or(terminations, truncations)

                # Since we filter infos, ensure 'reward' (which is always an array in EnvPool) exists
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

            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                next_q = self._network(observation.float()).max(dim=-1).values
                max_q_seq = batch_q.max(dim=-1).values
                q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])
                targets = lambda_returns(
                    batch_rewards,
                    batch_terminations,
                    q_seq_for_lambda[1:],
                    self.gamma,
                    self.lmbda,
                )

            flat_obs = batch_observations[:, : self.num_train_environments].reshape(
                (-1,) + observation_shape
            )
            flat_act = batch_actions[:, : self.num_train_environments].reshape(-1)
            flat_tgt = targets[:, : self.num_train_environments].reshape(-1)

            self._network.train()
            total_loss = 0.0

            for _ in range(self.epochs):
                indices = torch.randperm(self.batch_size, device=self.device)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]

                    mbatch_observations = flat_obs[mb_idx]
                    mb_act = flat_act[mb_idx]
                    mb_tgt = flat_tgt[mb_idx]

                    optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        q_values = self._network(mbatch_observations.float())
                        q_taken = q_values.gather(1, mb_act.unsqueeze(1)).squeeze()
                        loss = self._network.loss(q_taken, mb_tgt)

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
                    f"Test Score: {test_score:.4f} | Epsilon: {training_epsilon_value}",
                )

        train_environment.close()
        test_environment.close()

        self.final_training_rewards = all_train_rewards
        self.final_test_rewards = all_test_rewards
        self.final_loss_evolution = all_loss

        if self.verbose:
            flush(f"Training finished.")

    def make_filename(self, **arguments):
        dynamic_part = "_".join(f"{k}-{v}" for k, v in arguments.items())
        static_part = f"environment-{self.environment}"
        return f"{static_part}_{dynamic_part}"

    def save(self, **arguments) -> None:
        funnel(
            self.make_filename(**arguments),
            {
                "training_reward": self.final_training_rewards,
                "test_reward": self.final_test_rewards,
                "loss": self.final_loss_evolution,
            },
        )
