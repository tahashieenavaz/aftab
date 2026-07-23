import envpool
from .AftabBaseMixin import AftabBaseMixin


class AftabEnvironmentMixin(AftabBaseMixin):
    _OPTIONAL_ENVIRONMENT_KWARGS = (
        "noop_max",
        "frame_skip",
        "stack_num",
        "reward_clip",
        "episodic_life",
    )

    def _allocate_threads(self) -> tuple[int, int]:
        if self.cpu_count <= 1:
            return 1, 1
        total_envs = self.train_environments + self.test_environments
        train_ratio = self.train_environments / total_envs if total_envs > 0 else 0.8
        train_threads = max(1, int(self.cpu_count * train_ratio))
        test_threads = max(1, self.cpu_count - train_threads)
        return min(16, train_threads), min(16, test_threads)

    def _environment_config_keys(self, environment: str) -> set[str]:
        config = envpool.make_spec(environment).config
        return set(config)

    def _environment_kwargs(
        self,
        *,
        config_keys: set[str],
        reward_clip: bool,
        episodic_life: bool,
    ) -> dict:
        candidates = {
            "noop_max": self.noop,
            "frame_skip": self.frame_skip,
            "stack_num": self.frame_stack,
            "reward_clip": reward_clip,
            "episodic_life": episodic_life,
        }
        return {
            key: candidates[key]
            for key in self._OPTIONAL_ENVIRONMENT_KWARGS
            if key in config_keys
        }

    def _configure_frame_accounting(self, config_keys: set[str]) -> None:
        self.environment_frame_skip = (
            self.frame_skip if "frame_skip" in config_keys else 1
        )
        self.effective_frames = int(self.frames / self.environment_frame_skip)
        self.total_updates = (self.effective_frames + self.batch_size - 1) // (
            self.batch_size
        )

    def make_environments(self, environment: str, seed: int):
        train_threads, test_threads = self._allocate_threads()
        config_keys = self._environment_config_keys(environment)
        self._configure_frame_accounting(config_keys)

        train_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.train_environments,
            seed=seed,
            num_threads=train_threads,
            thread_affinity_offset=0,
            **self._environment_kwargs(
                config_keys=config_keys,
                reward_clip=self.train_reward_clip,
                episodic_life=self.train_episodic_life,
            ),
        )

        test_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.test_environments,
            seed=seed + self.seed_offset,
            num_threads=test_threads,
            thread_affinity_offset=train_threads if self.cpu_count > 1 else 0,
            **self._environment_kwargs(
                config_keys=config_keys,
                reward_clip=self.test_reward_clip,
                episodic_life=self.test_episodic_life,
            ),
        )

        return (
            train_environment,
            test_environment,
            train_environment.action_space.n,
            train_environment.observation_space.shape,
        )
