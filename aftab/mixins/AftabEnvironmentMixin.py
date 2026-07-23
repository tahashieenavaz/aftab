from collections.abc import Mapping

import envpool
import numpy

from .AftabBaseMixin import AftabBaseMixin


class _ProcgenFrameStack:
    def __init__(self, environment, stack_num: int):
        observation_shape = environment.observation_space.shape
        if len(observation_shape) != 3 or observation_shape[0] != 3:
            raise ValueError(
                "Expected Procgen observations with shape (3, height, width)."
            )
        if stack_num <= 0:
            raise ValueError("Expected `frame_stack` to be positive.")

        self.environment = environment
        self.observation_shape = (stack_num, *observation_shape[1:])
        self.stack_num = stack_num
        self.frames = None
        self.needs_reset = None

    def __getattr__(self, name):
        return getattr(self.environment, name)

    @staticmethod
    def _grayscale(observation):
        rgb = observation.astype(numpy.uint16, copy=False)
        return ((77 * rgb[:, 0] + 150 * rgb[:, 1] + 29 * rgb[:, 2] + 128) >> 8).astype(
            numpy.uint8
        )

    def reset(self, *args, **kwargs):
        observation, info = self.environment.reset(*args, **kwargs)
        grayscale = self._grayscale(observation)
        self.frames = numpy.repeat(grayscale[:, None], self.stack_num, axis=1)
        self.needs_reset = numpy.zeros(grayscale.shape[0], dtype=numpy.bool_)
        return self.frames, info

    def step(self, actions):
        observation, reward, termination, truncation, info = self.environment.step(
            actions
        )
        grayscale = self._grayscale(observation)

        if self.frames is None:
            self.frames = numpy.repeat(grayscale[:, None], self.stack_num, axis=1)
            self.needs_reset = numpy.zeros(grayscale.shape[0], dtype=numpy.bool_)
        else:
            self.frames[:, :-1] = self.frames[:, 1:]
            self.frames[:, -1] = grayscale
            self.frames[self.needs_reset] = grayscale[self.needs_reset, None]

        self.needs_reset = numpy.logical_or(termination, truncation)
        return self.frames, reward, termination, truncation, info


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
        fields = getattr(config, "_fields", None)
        if fields is not None:
            return set(fields)
        if isinstance(config, Mapping):
            return set(config.keys())
        raise TypeError(
            "Expected EnvPool's environment config to be a named tuple or mapping, "
            f"but received {type(config).__name__}."
        )

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
        procgen = {"env_name", "distribution_mode"}.issubset(config_keys)

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

        if procgen:
            train_environment = _ProcgenFrameStack(
                train_environment,
                stack_num=self.frame_stack,
            )
            test_environment = _ProcgenFrameStack(
                test_environment,
                stack_num=self.frame_stack,
            )
            observation_shape = train_environment.observation_shape
        else:
            observation_shape = train_environment.observation_space.shape

        return (
            train_environment,
            test_environment,
            train_environment.action_space.n,
            observation_shape,
        )
