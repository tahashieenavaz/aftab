import envpool
import numpy
from .AftabBaseMixin import AftabBaseMixin


class _SplitEnvPool:
    def __init__(self, environment, train_environments: int, test_environments: int):
        self.environment = environment
        self.train_envs = train_environments
        self.total_envs = train_environments + test_environments
        self.action_space = environment.action_space
        self.observation_space = environment.observation_space
        self._actions = numpy.empty(self.total_envs, dtype=numpy.int64)
        self.train_slice = slice(0, self.train_envs)
        self.test_slice = slice(self.train_envs, self.total_envs)

    def _split_info(self, info, slc: slice):
        if not info:
            return {}

        return {
            k: (
                v[slc]
                if (getattr(v, "shape", ()) and v.shape[0] == self.total_envs)
                or (isinstance(v, (list, tuple)) and len(v) == self.total_envs)
                else v
            )
            for k, v in info.items()
        }

    def reset_split(self):
        observation, info = self.environment.reset()
        return (
            (observation[self.train_slice], self._split_info(info, self.train_slice)),
            (observation[self.test_slice], self._split_info(info, self.test_slice)),
        )

    def step_split(self, train_actions, test_actions):
        self._actions[self.train_slice] = train_actions
        if self.total_envs > self.train_envs:  # Only assign if test environments exist
            self._actions[self.test_slice] = test_actions

        observation, reward, termination, truncation, info = self.environment.step(
            self._actions
        )

        return (
            (
                observation[self.train_slice],
                reward[self.train_slice],
                termination[self.train_slice],
                truncation[self.train_slice],
                self._split_info(info, self.train_slice),
            ),
            (
                observation[self.test_slice],
                reward[self.test_slice],
                termination[self.test_slice],
                truncation[self.test_slice],
                self._split_info(info, self.test_slice),
            ),
        )

    def close(self):
        self.environment.close()


class AftabEnvironmentMixin(AftabBaseMixin):
    def __init__(self):
        super().__init__()

    def _get_optimal_threads(self, allocation: int) -> int:
        return max(1, min(16, allocation))

    def __make_shared_environment(self, environment: str, seed: int):
        num_threads = self._get_optimal_threads(self.cpu_count)

        combined_environment = envpool.make(
            environment,
            env_type=(
                "grid" if "gymnasium" not in envpool.list_all_envs() else "gymnasium"
            ),
            num_envs=self.total_environments,
            seed=seed,
            num_threads=num_threads,
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
        return (
            split_environment,
            None,
            split_environment.action_space.n,
            split_environment.observation_space.shape,
        )

    def make_environments(self, environment: str, seed: int):
        if getattr(self, "shared_envpool"):
            return self.__make_shared_environment(environment=environment, seed=seed)

        raw_test_cpu = min(self.min_cpu_count, max(1, self.cpu_count - 1))
        raw_train_cpu = max(1, self.cpu_count - raw_test_cpu)

        train_threads = self._get_optimal_threads(raw_train_cpu)
        test_threads = self._get_optimal_threads(raw_test_cpu)

        train_environment = envpool.make(
            environment,
            env_type="gymnasium",
            num_envs=self.train_environments,
            seed=seed,
            num_threads=train_threads,
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
            num_threads=test_threads,
            thread_affinity_offset=train_threads if self.cpu_count > 1 else 0,
            noop_max=self.noop,
            reward_clip=self.test_reward_clip,
            episodic_life=self.test_episodic_life,
            frame_skip=self.frame_skip,
            stack_num=self.frame_stack,
        )
        return (
            train_environment,
            test_environment,
            train_environment.action_space.n,
            train_environment.observation_space.shape,
        )
