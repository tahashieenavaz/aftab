import envpool
from .AftabBaseMixin import AftabBaseMixin


class AftabEnvironmentMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def make_environments(self, environment: str, seed: int):
        test_cpu_count = min(self.min_cpu_count, max(1, self.cpu_count - 1))
        train_cpu_count = max(1, self.cpu_count - test_cpu_count)

        train_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.train_environments,
            seed=seed,
            num_threads=train_cpu_count,
            thread_affinity_offset=0,
            noop_max=self.noop,
            reward_clip=self.train_reward_clip,
            episodic_life=self.train_episodic_life,
            frame_skip=self.frame_skip,
            stack_num=self.frame_stack,
        )

        test_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.test_environments,
            seed=seed + 1000,
            num_threads=test_cpu_count,
            thread_affinity_offset=train_cpu_count if self.cpu_count > 1 else 0,
            noop_max=self.noop,
            reward_clip=self.test_reward_clip,
            episodic_life=self.test_episodic_life,
            frame_skip=self.frame_skip,
            stack_num=self.frame_stack,
        )
        action_dimension = train_environment.action_space.n
        observation_shape = train_environment.observation_space.shape
        return train_environment, test_environment, action_dimension, observation_shape
