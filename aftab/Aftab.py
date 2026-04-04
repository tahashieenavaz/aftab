import torch
import numpy
import os
import math
import envpool
from baloot import acceleration_device, seed_everything
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
        logging_interval: int = 10,
        num_train_environments: int = 128,
        num_test_environments: int = 8,
        steps_per_update: int = 32,
        total_frames: int = 200_000_000,
        seed: int = 42,
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
    ):
        self.device = acceleration_device()
        self.frame_skip = frame_skip
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.num_minibatches = num_minibatches
        self.logging_interval = logging_interval
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
        self.seed = seed
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

        ######
        # these will be filled right after the training is completed
        ######
        self.final_training_rewards = None
        self.final_test_rewards = None
        self.final_loss_evolution = None

        ######
        # this line ensures users can pass a string (predefined) or their defined encoder to the system.
        ######
        if isinstance(encoder, str):
            module = AftabMapEncoder.get(encoder)
            self.encoder = module

    def make_network(
        action_dimension: int, encoder_instance: Type[torch.nn.Module]
    ) -> Type[torch.nn.Module]:
        return PQNAgent(
            action_dimension=action_dimension, encoder_instance=encoder_instance
        )

    def set_precision(self):
        torch.set_float32_matmul_precision("high")

    def set_seed(self):
        seed_everything(self.seed)

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

    def train(self, environment):
        self.set_precision()
        self.set_seed()
        all_train_rewards = []
        all_test_rewards = []
        all_loss = []
        episode_returns = numpy.zeros(self.total_envs, dtype=numpy.float32)
        train_environment, test_environment = self.make_environments(environment)
        action_dimension = train_environment.action_space.n
        self._network = self.make_network(action_dimension, self.encoder)

        ######
        # a dummy pass ensure all the lazy layer are initialized
        ######
        with torch.no_grad():
            self._network(torch.randn(1, 4, 84, 84).to(self.device))

        obs_train, _ = train_environment.reset()
        obs_test, _ = test_environment.reset()
        observation = numpy.concatenate([obs_train, obs_test], axis=0)
        observation = torch.as_tensor(
            observation, dtype=torch.uint8, device=self.device
        )
        optimizer = torch.optim.RAdam(
            self._network.parameters(), lr=self.lr, eps=self.optimizer_epsilon
        )
        frame_count = 0
        observation_shape = train_environment.observation_space.shape
        b_obs = torch.empty(
            (self.steps_per_update, self.total_envs) + observation_shape,
            dtype=torch.uint8,
            device=self._device,
        )
        b_act = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.int64,
            device=self.device,
        )
        b_rew = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        b_done = torch.empty(
            (self.steps_per_update, self.total_environments),
            dtype=torch.float32,
            device=self.device,
        )
        b_q = torch.empty(
            (self.steps_per_update, self.total_environments, action_dimension),
            dtype=torch.float32,
            device=self.device,
        )

        scaler = torch.amp.GradScaler("cuda")

        for update in range(1, self.total_updates + 1):
            self._network.eval()
            for step in range(self.steps_per_update):
                float_observations = observation.float()
                train_eps_val = self._network.epsilon.get(
                    frame_count,
                    self.actual_frames,
                    all_train_rewards,
                    episode_returns[: self.num_train_environments],
                )
                epsilon_vector = torch.full(
                    (self.num_train_environments,),
                    train_eps_val,
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

                next_obs_train, rew_train, term_train, trunc_train, info_train = (
                    train_environment.step(act_train)
                )
                next_obs_test, rew_test, term_test, trunc_test, info_test = (
                    test_environment.step(act_test)
                )

                next_obs = numpy.concatenate([next_obs_train, next_obs_test], axis=0)
                rewards = numpy.concatenate([rew_train, rew_test], axis=0)
                terms = numpy.concatenate([term_train, term_test], axis=0)
                truncs = numpy.concatenate([trunc_train, trunc_test], axis=0)

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

                dones = numpy.logical_or(terms, truncs)

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

                b_obs[step] = obs
                b_act[step] = torch.as_tensor(actions, device=self.device)
                b_rew[step] = torch.as_tensor(rewards, device=self.device)
                b_done[step] = torch.as_tensor(dones, device=self.device)
                b_q[step] = q_values

                obs = torch.as_tensor(next_obs, dtype=torch.uint8, device=self.device)
                frame_count += self.num_train_environments

            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device.type, dtype=torch.float16),
            ):
                next_q = self._network(obs.float()).max(dim=-1).values
                max_q_seq = b_q.max(dim=-1).values
                q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])
                targets = lambda_returns(
                    b_rew, b_done, q_seq_for_lambda[1:], self.gamma, self.lmbda
                )

            flat_obs = b_obs[:, : self.num_train_environments].reshape(
                (-1,) + observation_shape
            )
            flat_act = b_act[:, : self.num_train_environments].reshape(-1)
            flat_tgt = targets[:, : self.num_train_environments].reshape(-1)

            self._network.train()
            total_loss = 0.0

            for _ in range(self.epochs):
                indices = torch.randperm(self.batch_size, device=self.device)

                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]

                    mb_obs = flat_obs[mb_idx]
                    mb_act = flat_act[mb_idx]
                    mb_tgt = flat_tgt[mb_idx]

                    optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        q_values = self._network(mb_obs.float())
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
                    f"Test Score: {test_score:.4f} | Epsilon: {train_eps_val}",
                )

        train_environment.close()
        test_environment.close()

        self.final_training_rewards = all_train_rewards
        self.final_test_rewards = all_test_rewards
        self.final_loss_evolution = all_loss

        if self.verbose:
            flush(f"Training finished.")

    def save(name: str):
        pass
