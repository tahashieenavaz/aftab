import envpool
from .AftabBaseMixin import AftabBaseMixin


class AftabEnvironmentMixin(AftabBaseMixin):
    def _allocate_threads(self) -> tuple[int, int]:
        if self.cpu_count <= 1:
            return 1, 1
        total_envs = self.train_environments + self.test_environments
        train_ratio = self.train_environments / total_envs if total_envs > 0 else 0.8
        train_threads = max(1, int(self.cpu_count * train_ratio))
        test_threads = max(1, self.cpu_count - train_threads)
        return min(16, train_threads), min(16, test_threads)

    def make_environments(self, environment: str, seed: int):
        train_threads, test_threads = self._allocate_threads()

        common_kwargs = {
            "env_type": "gymnasium",
            "noop_max": self.noop,
            "frame_skip": self.frame_skip,
            "stack_num": self.frame_stack,
        }

        train_environment = envpool.make(
            environment,
            **common_kwargs,
            num_envs=self.train_environments,
            seed=seed,
            num_threads=train_threads,
            thread_affinity_offset=0,
            reward_clip=self.train_reward_clip,
            episodic_life=self.train_episodic_life,
        )

        test_environment = envpool.make(
            environment,
            **common_kwargs,
            num_envs=self.test_environments,
            seed=seed + self.seed_offset,
            num_threads=test_threads,
            thread_affinity_offset=train_threads if self.cpu_count > 1 else 0,
            reward_clip=self.test_reward_clip,
            episodic_life=self.test_episodic_life,
        )

        return (
            train_environment,
            test_environment,
            train_environment.action_space.n,
            train_environment.observation_space.shape,
        )
