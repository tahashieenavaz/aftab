import torch
import numpy
import os
import math
import envpool
from baloot import acceleration_device, seed_everything
from typing import Type
from .maps import AftabMapEncoder
from .agents import PQNAgent


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

        ######
        # This line ensures users can pass a string (predefined) or their defined encoder to the system.
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

    def save(name: str):
        pass
