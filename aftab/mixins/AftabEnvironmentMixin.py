import envpool
import numpy
from .AftabBaseMixin import AftabBaseMixin


class _SplitEnvPool:
    def __init__(self, environment, train_environments: int, test_environments: int):
        self.environment = environment
        self.train_environments = train_environments
        self.test_environments = test_environments
        self.total_environments = train_environments + test_environments
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self._actions = numpy.empty(self.total_environments, dtype=numpy.int64)

    def __split_info(self, info, start: int, end: int):
        if info is None:
            return {}

        split_info = {}
        for key, value in info.items():
            if (
                hasattr(value, "shape")
                and len(value.shape) > 0
                and value.shape[0] == self.total_environments
            ):
                split_info[key] = value[start:end]
            elif (
                isinstance(value, (list, tuple))
                and len(value) == self.total_environments
            ):
                split_info[key] = value[start:end]
            else:
                split_info[key] = value
        return split_info

    def reset_split(self):
        observation, info = self.environment.reset()
        train_slice = slice(0, self.train_environments)
        test_slice = slice(self.train_environments, self.total_environments)
        return (
            (
                observation[train_slice],
                self.__split_info(info, train_slice.start, train_slice.stop),
            ),
            (
                observation[test_slice],
                self.__split_info(info, test_slice.start, test_slice.stop),
            ),
        )

    def step_split(self, train_actions, test_actions):
        self._actions[: self.train_environments] = train_actions
        if self.test_environments > 0:
            self._actions[self.train_environments :] = test_actions

        observation, reward, termination, truncation, info = self.environment.step(
            self._actions
        )
        train_slice = slice(0, self.train_environments)
        test_slice = slice(self.train_environments, self.total_environments)
        return (
            (
                observation[train_slice],
                reward[train_slice],
                termination[train_slice],
                truncation[train_slice],
                self.__split_info(info, train_slice.start, train_slice.stop),
            ),
            (
                observation[test_slice],
                reward[test_slice],
                termination[test_slice],
                truncation[test_slice],
                self.__split_info(info, test_slice.start, test_slice.stop),
            ),
        )

    def close(self):
        self.environment.close()


class AftabEnvironmentMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def __make_shared_environment(self, environment: str, seed: int):
        combined_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.total_environments,
            seed=seed,
            num_threads=max(1, self.cpu_count),
            thread_affinity_offset=0,
            noop_max=self.noop,
            reward_clip=self.train_reward_clip,
            episodic_life=self.train_episodic_life,
            frame_skip=self.frame_skip,
            stack_num=self.frame_stack,
        )
        split_environment = _SplitEnvPool(
            combined_environment,
            train_environments=self.train_environments,
            test_environments=self.test_environments,
        )
        action_dimension = split_environment.action_space.n
        observation_shape = split_environment.observation_space.shape
        return split_environment, None, action_dimension, observation_shape

    def make_environments(self, environment: str, seed: int):
        if bool(getattr(self, "shared_envpool", True)):
            return self.__make_shared_environment(environment=environment, seed=seed)

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
