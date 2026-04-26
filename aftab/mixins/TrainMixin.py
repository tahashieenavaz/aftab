import torch
import numpy
import time
from typing import Optional


class TrainMixin:
    def __init__(self):
        super().__init__()

    def __autocast_float16_enabled(self):
        return (
            bool(getattr(self, "autocast_float16", False))
            and self.device.type != "cpu"
            and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )

    def __autocast_float16(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.__autocast_float16_enabled(),
        )

    def __distributional_value_clip_enabled(self):
        return bool(getattr(self._network, "distributional", False)) and (
            float(getattr(self, "distributional_value_clip", 0.0)) > 0.0
        )

    def __initialize_training(self, environment: str, seed: int):
        self.flush_results()
        self.set_precision()
        self.set_seed(seed)

        train_environment, test_environment, action_dimension, observation_shape = (
            self.make_environments(environment=environment, seed=seed)
        )
        self.prepare_network(action_dimension=action_dimension)
        optimizer = self.make_optimizer()
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        train_observation, _ = train_environment.reset()
        train_observation = (
            torch.from_numpy(train_observation).to(torch.uint8).to(self.device)
        )

        test_observation, _ = test_environment.reset()
        test_observation = (
            torch.from_numpy(test_observation).to(torch.uint8).to(self.device)
        )
        observation = torch.cat([train_observation, test_observation], dim=0)
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        return (
            train_environment,
            test_environment,
            action_dimension,
            observation_shape,
            optimizer,
            scaler,
            observation,
            episode_returns,
        )

    def __allocate_buffers(self, *, observation_shape: tuple):
        batch_observations = torch.empty(
            (self.steps_per_update, self.total_environments) + observation_shape,
            dtype=torch.uint8,
            device=self.device,
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
        batch_old_q_values = None
        if self.__distributional_value_clip_enabled():
            batch_old_q_values = torch.empty(
                (self.steps_per_update, self.total_environments),
                dtype=torch.float32,
                device=self.device,
            )
        return (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_old_q_values,
        )

    def __collect_trajectories(
        self,
        *,
        frame_count: int,
        train_environment,
        test_environment,
        episode_returns,
        observation: torch.Tensor,
        batch_observations: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_terminations: torch.Tensor,
        batch_old_q_values: Optional[torch.Tensor],
    ):
        for step in range(self.steps_per_update):
            train_observation = observation[: self.train_environments]
            test_observation = observation[self.train_environments :]
            float_train_observations = train_observation.float()
            float_test_observations = test_observation.float()
            epsilon_value = self._network.epsilon.get(
                frame_count,
                self.actual_frames,
            )

            with self.__autocast_float16():
                q_values = self.get_q_values(
                    float_train_observations=float_train_observations,
                    float_test_observations=float_test_observations,
                    gradient=False,
                )

            actions_train, actions_test = self.get_actions(
                q_values_train=q_values["train"],
                q_values_test=q_values["test"],
                epsilon_value=epsilon_value,
            )
            actions = numpy.concatenate([actions_train, actions_test], axis=0)
            actions_tensor = torch.from_numpy(actions).to(self.device)
            if batch_old_q_values is not None:
                q_values_all = torch.cat([q_values["train"], q_values["test"]], dim=0)
                batch_old_q_values[step] = q_values_all.gather(
                    1,
                    actions_tensor.unsqueeze(1),
                ).squeeze(1)

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
            rewards = numpy.concatenate([reward_train, reward_test], axis=0).astype(
                numpy.float32,
                copy=False,
            )
            terminations = numpy.concatenate(
                [termination_train, termination_test], axis=0
            )
            truncations = numpy.concatenate([truncation_train, truncation_test], axis=0)
            terminations = numpy.logical_or(terminations, truncations)

            score_rewards = rewards
            score_reward_train = info_train.get("reward", None)
            score_reward_test = info_test.get("reward", None)
            if score_reward_train is not None and score_reward_test is not None:
                if isinstance(score_reward_train, numpy.ndarray) and isinstance(
                    score_reward_test, numpy.ndarray
                ):
                    if score_reward_train.ndim == 0:
                        score_rewards = numpy.stack(
                            [score_reward_train, score_reward_test]
                        )
                    else:
                        score_rewards = numpy.concatenate(
                            [score_reward_train, score_reward_test],
                            axis=0,
                        )
            score_rewards = score_rewards.astype(numpy.float32, copy=False)
            episode_returns += score_rewards

            done_mask = terminations
            if numpy.any(done_mask):
                idx = numpy.nonzero(done_mask)[0]
                scores = episode_returns[done_mask]
                split = idx < self.train_environments
                self.results.rewards.train.extend(scores[split].tolist())
                self.results.rewards.test.extend(scores[~split].tolist())
                episode_returns[done_mask] = 0

            batch_observations[step] = observation
            batch_actions[step] = actions_tensor
            batch_rewards[step] = torch.from_numpy(rewards).to(self.device)
            batch_terminations[step] = torch.from_numpy(terminations).to(self.device)

            observation = (
                torch.from_numpy(next_observation).to(torch.uint8).to(self.device)
            )
            frame_count += self.train_environments

        return observation, frame_count

    @torch.no_grad()
    def __compute_targets(
        self,
        *,
        batch_observations: torch.Tensor,
        batch_actions: torch.Tensor,
        observation: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_terminations: torch.Tensor,
    ):
        next_observations = torch.cat(
            [batch_observations[1:], observation.unsqueeze(0)], dim=0
        )
        sequence_length, environment_count = next_observations.shape[:2]
        flat_next_observations = next_observations.reshape(
            (-1,) + next_observations.shape[2:]
        )

        with self.__autocast_float16():
            next_q_values = self.get_q_values(
                float_observations=flat_next_observations,
                gradient=False,
            )

        next_q_values = next_q_values.float()
        next_q = next_q_values.max(dim=-1).values.reshape(
            sequence_length,
            environment_count,
        )

        return self.get_returns(
            batch_rewards=batch_rewards,
            batch_terminations=batch_terminations,
            next_q=next_q,
        )

    def __flatten_batches(
        self,
        *,
        batch_observations: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_old_q_values: Optional[torch.Tensor],
        targets: torch.Tensor,
        observation_shape: tuple,
    ):
        train_slice = slice(0, self.train_environments)
        flattened_observations = batch_observations[
            :, : self.train_environments
        ].reshape((-1,) + observation_shape)
        flattened_actions = batch_actions[:, train_slice].reshape(-1)
        flattened_old_q_values = None
        if batch_old_q_values is not None:
            flattened_old_q_values = batch_old_q_values[:, train_slice].reshape(-1)
        flattened_targets = targets[:, train_slice].reshape(-1)
        return (
            flattened_observations,
            flattened_actions,
            flattened_old_q_values,
            flattened_targets,
        )

    def __update_network(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        flattened_observations: torch.Tensor,
        flattened_actions: torch.Tensor,
        flattened_old_q_values: Optional[torch.Tensor],
        flattened_targets: torch.Tensor,
        scaler,
    ):
        self._network.train()
        for _ in range(self.epochs):
            indices = torch.randperm(self.batch_size, device=self.device)

            for range_start in range(0, self.batch_size, self.mini_batch_size):
                range_end = range_start + self.mini_batch_size
                mini_batch_idx = indices[range_start:range_end]
                mini_batch_observations = flattened_observations[mini_batch_idx]
                mini_batch_actions = flattened_actions[mini_batch_idx]
                mini_batch_old_q_values = None
                if flattened_old_q_values is not None:
                    mini_batch_old_q_values = flattened_old_q_values[mini_batch_idx]
                mini_batch_targets = flattened_targets[mini_batch_idx]

                optimizer.zero_grad(set_to_none=True)
                with self.__autocast_float16():
                    loss = self.get_loss(
                        mini_batch_observations=mini_batch_observations,
                        mini_batch_targets=mini_batch_targets,
                        mini_batch_actions=mini_batch_actions,
                        mini_batch_old_q_values=mini_batch_old_q_values,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(), self.gradient_norm
                )
                scaler.step(optimizer)
                scaler.update()
                self.results.loss.append(loss.item())

    def __log_progress(self, *, update: int, frame_count: int):
        verbose_window = self.verbose_window
        test_score = (
            0.0
            if len(self.results.rewards.test) < verbose_window
            else numpy.mean(self.results.rewards.test[-verbose_window:])
        )

        if update % self.verbose_interval == 0:
            self.flush_verbose(f"Update {update} | Frames: {frame_count:,}")
            self.flush_verbose(f"Test Score: {test_score:.4f}")

    def train_loop(self, *, environment: str, seed: int):
        frame_count = 0
        (
            train_environment,
            test_environment,
            action_dimension,
            observation_shape,
            optimizer,
            scaler,
            observation,
            episode_returns,
        ) = self.__initialize_training(environment=environment, seed=seed)

        (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_old_q_values,
        ) = self.__allocate_buffers(
            observation_shape=observation_shape,
        )

        training_start_time = time.time()

        for update in range(1, self.total_updates + 1):
            self._network.eval()

            observation, frame_count = self.__collect_trajectories(
                frame_count=frame_count,
                observation=observation,
                train_environment=train_environment,
                test_environment=test_environment,
                episode_returns=episode_returns,
                batch_observations=batch_observations,
                batch_actions=batch_actions,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
                batch_old_q_values=batch_old_q_values,
            )

            targets = self.__compute_targets(
                batch_observations=batch_observations,
                batch_actions=batch_actions,
                observation=observation,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
            )

            (
                flattened_observations,
                flattened_actions,
                flattened_old_q_values,
                flattened_targets,
            ) = self.__flatten_batches(
                batch_observations=batch_observations,
                batch_actions=batch_actions,
                batch_old_q_values=batch_old_q_values,
                targets=targets,
                observation_shape=observation_shape,
            )

            self.__update_network(
                optimizer=optimizer,
                scaler=scaler,
                flattened_observations=flattened_observations,
                flattened_actions=flattened_actions,
                flattened_old_q_values=flattened_old_q_values,
                flattened_targets=flattened_targets,
            )

            self.__log_progress(update=update, frame_count=frame_count)

        train_environment.close()
        test_environment.close()
        self.results.duration = time.time() - training_start_time
