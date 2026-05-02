import torch
import numpy
import time
from typing import Optional
from ..common import RolloutBuffer


class TrainMixin:
    def __init__(self):
        super().__init__()

    def __autocast_float16_enabled(self) -> bool:
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

    def __tensor(self, value, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def __distributional_value_clip_enabled(self) -> bool:
        return bool(getattr(self._network, "distributional", False)) and (
            float(getattr(self, "distributional_value_clip")) > 0.0
        )

    def __bootstrapped_enabled(self) -> bool:
        return bool(getattr(self._network, "bootstrapped", False))

    def __get_bootstrap_heads(self) -> int:
        return int(getattr(self._network, "bootstrap_heads", 1))

    def __sample_bootstrap_heads(self, size: int) -> torch.Tensor:
        return torch.randint(self.__get_bootstrap_heads(), (size,), device=self.device)

    def __sample_bootstrap_masks(self) -> torch.Tensor:
        probability = float(getattr(self, "bootstrap_probability", 1.0))
        if probability <= 0.0 or probability > 1.0:
            raise ValueError("Expected `bootstrap_probability` to be in (0, 1].")

        mask_shape = (self.train_environments, self.__get_bootstrap_heads())
        if probability == 1.0:
            return torch.ones(mask_shape, dtype=torch.float32, device=self.device)
        return (torch.rand(mask_shape, device=self.device) < probability).float()

    def __get_bootstrapped_vote_q_values(self, q_heads: torch.Tensor) -> torch.Tensor:
        head_actions = q_heads.argmax(dim=-1)
        votes = torch.nn.functional.one_hot(
            head_actions,
            num_classes=q_heads.shape[-1],
        )
        return votes.sum(dim=1).to(dtype=q_heads.dtype)

    def __resample_terminated_bootstrap_heads(
        self,
        *,
        active_heads: Optional[torch.Tensor],
        terminations,
    ) -> None:
        if active_heads is None:
            return

        done_mask = torch.as_tensor(
            terminations,
            dtype=torch.bool,
            device=self.device,
        )
        done_count = int(done_mask.sum().item())
        if done_count == 0:
            return
        active_heads[done_mask] = self.__sample_bootstrap_heads(done_count)

    def __get_step_q_values(
        self,
        *,
        float_observations: torch.Tensor,
        active_heads: Optional[torch.Tensor],
    ):
        if not self.__bootstrapped_enabled():
            q_values_all = self.get_q_values(
                float_observations=float_observations,
                gradient=False,
            )
            q_values_train = q_values_all[: self.train_environments]
            q_values_test = q_values_all[self.train_environments :]
            state_q_values = q_values_train.float().max(dim=-1).values
            return q_values_train, q_values_test, state_q_values

        if active_heads is None:
            raise RuntimeError("Expected active bootstrapped heads.")

        q_heads_all = self._network.get_q_heads(float_observations)
        q_heads_train = q_heads_all[: self.train_environments]
        q_values_train = self._network.gather_q_heads(
            q_heads=q_heads_train,
            head_indices=active_heads[: self.train_environments],
        )
        q_values_test = self.__get_bootstrapped_vote_q_values(
            q_heads_all[self.train_environments :]
        )
        state_q_values = q_heads_train.float().max(dim=-1).values
        return q_values_train, q_values_test, state_q_values

    def __initialize_training(self, environment: str, seed: int):
        train_environment, test_environment, action_dimension, observation_shape = (
            self.make_environments(environment=environment, seed=seed)
        )
        self.prepare_network(action_dimension=action_dimension)
        scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and self.__autocast_float16_enabled()
        )

        train_observation, _ = train_environment.reset()
        test_observation, _ = test_environment.reset()
        observation = torch.cat(
            [
                self.__tensor(train_observation, torch.uint8),
                self.__tensor(test_observation, torch.uint8),
            ],
            dim=0,
        )
        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        return (
            train_environment,
            test_environment,
            observation_shape,
            scaler,
            observation,
            episode_returns,
        )

    def __allocate_buffers(self, *, observation_shape: tuple):
        return RolloutBuffer(
            observation_shape=observation_shape,
            steps_per_update=self.steps_per_update,
            train_environments=self.train_environments,
            device=self.device,
            bootstrapped=self.__bootstrapped_enabled(),
            bootstrap_heads=self.__get_bootstrap_heads(),
            store_old_q_values=self.__distributional_value_clip_enabled(),
        )

    def __get_score_rewards(self, *, rewards, info_train, info_test):
        score_reward_train = info_train.get("reward", None)
        score_reward_test = info_test.get("reward", None)
        if not isinstance(score_reward_train, numpy.ndarray) or not isinstance(
            score_reward_test, numpy.ndarray
        ):
            return rewards.astype(numpy.float32, copy=False)

        if score_reward_train.ndim == 0:
            score_rewards = numpy.stack([score_reward_train, score_reward_test])
        else:
            score_rewards = numpy.concatenate(
                [score_reward_train, score_reward_test],
                axis=0,
            )
        return score_rewards.astype(numpy.float32, copy=False)

    def __record_completed_episodes(self, *, episode_returns, terminations):
        if not numpy.any(terminations):
            return

        idx = numpy.nonzero(terminations)[0]
        scores = episode_returns[terminations]
        split = idx < self.train_environments
        self.results.rewards.train.extend(scores[split].tolist())
        self.results.rewards.test.extend(scores[~split].tolist())
        episode_returns[terminations] = 0

    @torch.no_grad()
    def __collect_trajectories(
        self,
        *,
        frame_count: int,
        train_environment,
        test_environment,
        episode_returns,
        observation: torch.Tensor,
        rollout_buffer: RolloutBuffer,
        active_heads: Optional[torch.Tensor],
    ):
        for step in range(self.steps_per_update):
            train_observation = observation[: self.train_environments]
            float_observations = observation.float()
            epsilon_value = self._network.epsilon.get(
                frame_count,
                self.actual_frames,
            )

            with self.__autocast_float16():
                q_values_train, q_values_test, state_q_values = (
                    self.__get_step_q_values(
                        float_observations=float_observations,
                        active_heads=active_heads,
                    )
                )
            actions_train_tensor, actions_test_tensor = self.get_action_tensors(
                q_values_train=q_values_train,
                q_values_test=q_values_test,
                epsilon_value=epsilon_value,
            )
            actions_train = actions_train_tensor.cpu().numpy()
            actions_test = actions_test_tensor.cpu().numpy()
            old_q_values = None
            if rollout_buffer.old_q_values is not None:
                old_q_values = q_values_train.gather(
                    1,
                    actions_train_tensor.unsqueeze(1),
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
            reward_train = reward_train.astype(numpy.float32, copy=False)
            reward_test = reward_test.astype(numpy.float32, copy=False)
            rewards = numpy.concatenate([reward_train, reward_test], axis=0)
            termination_train = numpy.logical_or(termination_train, truncation_train)
            termination_test = numpy.logical_or(termination_test, truncation_test)
            terminations = numpy.concatenate(
                [termination_train, termination_test],
                axis=0,
            )

            score_rewards = self.__get_score_rewards(
                rewards=rewards,
                info_train=info_train,
                info_test=info_test,
            )
            episode_returns += score_rewards
            self.__record_completed_episodes(
                episode_returns=episode_returns,
                terminations=terminations,
            )
            self.__resample_terminated_bootstrap_heads(
                active_heads=active_heads,
                terminations=terminations,
            )

            bootstrap_masks = None
            if rollout_buffer.bootstrap_masks is not None:
                bootstrap_masks = self.__sample_bootstrap_masks()

            rollout_buffer.insert(
                step=step,
                observation=train_observation,
                action=actions_train_tensor,
                reward=self.__tensor(reward_train, torch.float32),
                termination=self.__tensor(termination_train, torch.float32),
                state_q_values=state_q_values,
                old_q_values=old_q_values,
                bootstrap_masks=bootstrap_masks,
            )

            observation = self.__tensor(next_observation, torch.uint8)
            frame_count += self.train_environments

        return observation, frame_count

    @torch.no_grad()
    def __compute_targets(
        self,
        *,
        observation: torch.Tensor,
        rollout_buffer: RolloutBuffer,
    ):
        last_train_observation = observation[: self.train_environments]
        returns_rewards = rollout_buffer.rewards
        returns_terminations = rollout_buffer.terminations

        if self.__bootstrapped_enabled():
            with self.__autocast_float16():
                last_next_q_heads = self._network.get_q_heads(
                    last_train_observation.float()
                )
            last_next_q = last_next_q_heads.float().max(dim=-1).values
            returns_rewards = rollout_buffer.rewards.unsqueeze(-1).expand_as(
                rollout_buffer.state_q_values
            )
            returns_terminations = rollout_buffer.terminations.unsqueeze(-1).expand_as(
                rollout_buffer.state_q_values
            )
        else:
            with self.__autocast_float16():
                last_next_q_values = self.get_q_values(
                    float_observations=last_train_observation,
                    gradient=False,
                )
            last_next_q = last_next_q_values.float().max(dim=-1).values

        next_q = torch.empty_like(rollout_buffer.state_q_values)
        next_q[:-1] = rollout_buffer.state_q_values[1:]
        next_q[-1] = last_next_q

        return self.get_returns(
            batch_rewards=returns_rewards,
            batch_terminations=returns_terminations,
            next_q=next_q,
        )

    def __get_bootstrapped_loss(
        self,
        *,
        mini_batch_observations: torch.Tensor,
        mini_batch_actions: torch.Tensor,
        mini_batch_targets: torch.Tensor,
        mini_batch_bootstrap_masks: torch.Tensor,
    ) -> torch.Tensor:
        mini_batch_observations = mini_batch_observations.float()
        q_heads = self._network.get_q_heads(mini_batch_observations)
        action_indices = mini_batch_actions.reshape(-1, 1, 1).expand(
            -1,
            self.__get_bootstrap_heads(),
            1,
        )
        q_taken = q_heads.gather(2, action_indices).squeeze(2).float()
        loss = 0.5 * (q_taken - mini_batch_targets.float()).pow(2)
        masks = mini_batch_bootstrap_masks.to(dtype=loss.dtype)
        return (loss * masks).sum() / masks.sum().clamp_min(1.0)

    def __slice_optional(
        self,
        tensor: Optional[torch.Tensor],
        indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return None if tensor is None else tensor[indices]

    def __update_network(
        self,
        *,
        flattened_observations: torch.Tensor,
        flattened_actions: torch.Tensor,
        flattened_old_q_values: Optional[torch.Tensor],
        flattened_bootstrap_masks: Optional[torch.Tensor],
        flattened_targets: torch.Tensor,
        flattened_target_probs: Optional[torch.Tensor],
        scaler,
    ):
        self._network.train()
        update_losses = []
        clip_grad_foreach = self.device.type in {"cpu", "cuda"}
        for _ in range(self.epochs):
            indices = torch.randperm(self.batch_size, device=self.device)

            for mini_batch_idx in indices.split(self.mini_batch_size):
                mini_batch_observations = flattened_observations[mini_batch_idx]
                mini_batch_actions = flattened_actions[mini_batch_idx]
                mini_batch_targets = flattened_targets[mini_batch_idx]
                mini_batch_old_q_values = self.__slice_optional(
                    flattened_old_q_values,
                    mini_batch_idx,
                )
                mini_batch_target_probs = self.__slice_optional(
                    flattened_target_probs,
                    mini_batch_idx,
                )
                mini_batch_bootstrap_masks = self.__slice_optional(
                    flattened_bootstrap_masks,
                    mini_batch_idx,
                )

                self._optimizer.zero_grad(set_to_none=True)
                with self.__autocast_float16():
                    if mini_batch_bootstrap_masks is None:
                        loss = self.get_loss(
                            mini_batch_observations=mini_batch_observations,
                            mini_batch_targets=mini_batch_targets,
                            mini_batch_actions=mini_batch_actions,
                            mini_batch_old_q_values=mini_batch_old_q_values,
                            mini_batch_target_probs=mini_batch_target_probs,
                        )
                    else:
                        loss = self.__get_bootstrapped_loss(
                            mini_batch_observations=mini_batch_observations,
                            mini_batch_targets=mini_batch_targets,
                            mini_batch_actions=mini_batch_actions,
                            mini_batch_bootstrap_masks=mini_batch_bootstrap_masks,
                        )
                scaler.scale(loss).backward()
                scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    self.gradient_norm,
                    foreach=clip_grad_foreach,
                )
                scaler.step(self._optimizer)
                scaler.update()
                update_losses.append(loss.detach())

        if update_losses:
            self.results.loss.extend(torch.stack(update_losses).float().cpu().tolist())

    def __log_progress(self, *, update: int, frame_count: int):
        if update % self.verbose_interval != 0:
            return

        verbose_window = self.verbose_window
        test_score = (
            0.0
            if len(self.results.rewards.test) < verbose_window
            else numpy.mean(self.results.rewards.test[-verbose_window:])
        )

        self.flush_verbose(f"Update {update} | Frames: {frame_count * self.frame_skip:,}")
        self.flush_verbose(f"Test Score: {test_score:.4f}")

    def train_loop(self, *, environment: str, seed: int):
        frame_count = 0
        (
            train_environment,
            test_environment,
            observation_shape,
            scaler,
            observation,
            episode_returns,
        ) = self.__initialize_training(environment=environment, seed=seed)

        rollout_buffer = self.__allocate_buffers(
            observation_shape=observation_shape,
        )
        active_heads = (
            self.__sample_bootstrap_heads(self.total_environments)
            if self.__bootstrapped_enabled()
            else None
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
                rollout_buffer=rollout_buffer,
                active_heads=active_heads,
            )

            targets = self.__compute_targets(
                observation=observation,
                rollout_buffer=rollout_buffer,
            )

            (
                flattened_observations,
                flattened_actions,
                flattened_old_q_values,
                flattened_targets,
                flattened_bootstrap_masks,
            ) = rollout_buffer.flatten(targets)
            flattened_target_probs = None
            if bool(getattr(self._network, "distributional", False)):
                flattened_target_probs = self._network.hl_gauss_loss.transform_to_probs(
                    flattened_targets
                )

            self.__update_network(
                scaler=scaler,
                flattened_observations=flattened_observations,
                flattened_actions=flattened_actions,
                flattened_old_q_values=flattened_old_q_values,
                flattened_bootstrap_masks=flattened_bootstrap_masks,
                flattened_targets=flattened_targets,
                flattened_target_probs=flattened_target_probs,
            )

            self.__log_progress(update=update, frame_count=frame_count)

        train_environment.close()
        test_environment.close()
        self.results.duration = time.time() - training_start_time
