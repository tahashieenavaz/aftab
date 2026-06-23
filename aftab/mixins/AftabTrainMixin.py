import torch
import numpy
import time
from typing import Optional
from .AftabBaseMixin import AftabBaseMixin
from ..common import RolloutBuffer

_TORCH_TO_NUMPY_DTYPE = {
    torch.bool: numpy.bool_,
    torch.uint8: numpy.uint8,
    torch.int8: numpy.int8,
    torch.int16: numpy.int16,
    torch.int32: numpy.int32,
    torch.int64: numpy.int64,
    torch.float16: numpy.float16,
    torch.float32: numpy.float32,
    torch.float64: numpy.float64,
}


class AftabTrainMixin(AftabBaseMixin):
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

    def __numpy_array(self, value, dtype: torch.dtype) -> numpy.ndarray:
        numpy_dtype = _TORCH_TO_NUMPY_DTYPE.get(dtype)
        if isinstance(value, numpy.ndarray):
            array = value
            if numpy_dtype is not None and array.dtype != numpy_dtype:
                array = array.astype(numpy_dtype, copy=False)
        else:
            array = numpy.asarray(value, dtype=numpy_dtype)

        if not array.flags.c_contiguous:
            array = numpy.ascontiguousarray(array)
        return array

    def __tensor(self, value, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=self.device)

        tensor = torch.from_numpy(self.__numpy_array(value, dtype))
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if tensor.device == self.device:
            return tensor
        return tensor.to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        )

    def __copy_array_to_tensor(
        self,
        target: torch.Tensor,
        value,
        dtype: torch.dtype,
    ) -> None:
        if isinstance(value, torch.Tensor):
            source = value.to(dtype=dtype, device=target.device)
        else:
            source = torch.from_numpy(self.__numpy_array(value, dtype))
            if source.dtype != dtype:
                source = source.to(dtype=dtype)
        target.copy_(source, non_blocking=target.device.type == "cuda")

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

        done_indices = numpy.nonzero(terminations)[0]
        done_count = int(done_indices.size)
        if done_count == 0:
            return

        done_indices = torch.from_numpy(done_indices).to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        )
        active_heads[done_indices] = self.__sample_bootstrap_heads(done_count)

    def __get_step_q_values(
        self,
        *,
        observations: torch.Tensor,
        active_heads: Optional[torch.Tensor],
    ):
        if not self.__bootstrapped_enabled():
            q_values_all = self.get_q_values(
                float_observations=observations,
                gradient=False,
            )
            q_values_train = q_values_all[: self.train_environments]
            q_values_test = q_values_all[self.train_environments :]
            state_q_values = q_values_train.float().max(dim=-1).values
            return q_values_train, q_values_test, state_q_values, None

        if active_heads is None:
            raise RuntimeError("Expected active bootstrapped heads.")

        q_heads_all = self._network.get_q_heads(observations)
        q_heads_train = q_heads_all[: self.train_environments]
        q_values_train = self._network.gather_q_heads(
            q_heads=q_heads_train,
            head_indices=active_heads[: self.train_environments],
        )
        q_values_test = self.__get_bootstrapped_vote_q_values(
            q_heads_all[self.train_environments :]
        )
        state_q_values = q_heads_train.float().max(dim=-1).values
        return q_values_train, q_values_test, state_q_values, q_heads_train

    def __initialize_training(self, environment: str, seed: int):
        train_environment, test_environment, action_dimension, observation_shape = (
            self.make_environments(environment=environment, seed=seed)
        )

        self._initialize_network(action_dimension=action_dimension)
        self._initialize_optimizer()

        scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and self.__autocast_float16_enabled()
        )

        if test_environment is None:
            (train_observation, _), (test_observation, _) = (
                train_environment.reset_split()
            )
        else:
            train_observation, _ = train_environment.reset()
            test_observation, _ = test_environment.reset()
        observation = torch.empty(
            (self.total_environments, *observation_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.__copy_array_to_tensor(
            observation[: self.train_environments],
            train_observation,
            torch.uint8,
        )
        self.__copy_array_to_tensor(
            observation[self.train_environments :],
            test_observation,
            torch.uint8,
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

    def __allocate_buffers(self, *, observation_shape: tuple) -> RolloutBuffer:
        return RolloutBuffer(
            observation_shape=observation_shape,
            steps_per_update=self.steps_per_update,
            train_environments=self.train_environments,
            device=self.device,
            bootstrapped=self.__bootstrapped_enabled(),
            bootstrap_heads=self.__get_bootstrap_heads(),
            store_old_q_values=self.__distributional_value_clip_enabled(),
        )

    def __get_score_rewards(
        self,
        *,
        reward_train,
        reward_test,
        info_train,
        info_test,
        output,
    ):
        score_reward_train = info_train.get("reward", None)
        score_reward_test = info_test.get("reward", None)
        if not isinstance(score_reward_train, numpy.ndarray) or not isinstance(
            score_reward_test, numpy.ndarray
        ):
            output[: self.train_environments] = reward_train
            output[self.train_environments :] = reward_test
            return output

        if score_reward_train.ndim == 0:
            return numpy.stack([score_reward_train, score_reward_test]).astype(
                numpy.float32,
                copy=False,
            )
        else:
            output[: self.train_environments] = score_reward_train
            output[self.train_environments :] = score_reward_test
        return output

    def __get_episode_terminations(
        self,
        *,
        info_train,
        info_test,
        fallback_terminations,
        output,
    ):
        episode_done_train = info_train.get("terminated", None)
        episode_done_test = info_test.get("terminated", None)
        if not isinstance(episode_done_train, numpy.ndarray) or not isinstance(
            episode_done_test, numpy.ndarray
        ):
            return fallback_terminations

        if episode_done_train.ndim == 0:
            return fallback_terminations

        output[: self.train_environments] = episode_done_train
        output[self.train_environments :] = episode_done_test
        return output

    def __record_completed_episodes(self, *, episode_returns, terminations):
        idx = numpy.nonzero(terminations)[0]
        if idx.size == 0:
            return

        scores = episode_returns[idx]
        split = idx < self.train_environments
        self.results.rewards.train.extend(scores[split].tolist())
        self.results.rewards.test.extend(scores[~split].tolist())
        episode_returns[idx] = 0

    @torch.inference_mode()
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
        actions_train_cpu = torch.empty(self.train_environments, dtype=torch.int64)
        actions_test_cpu = torch.empty(self.test_environments, dtype=torch.int64)
        actions_train = actions_train_cpu.numpy()
        actions_test = actions_test_cpu.numpy()
        terminations = numpy.empty(self.total_environments, dtype=numpy.bool_)
        episode_terminations = numpy.empty(self.total_environments, dtype=numpy.bool_)
        score_rewards_output = numpy.empty(
            self.total_environments,
            dtype=numpy.float32,
        )
        full_bootstrap_masks = None
        if (
            rollout_buffer.bootstrap_masks is not None
            and float(getattr(self, "bootstrap_probability", 1.0)) == 1.0
        ):
            full_bootstrap_masks = torch.ones(
                (self.train_environments, self.__get_bootstrap_heads()),
                dtype=torch.float32,
                device=self.device,
            )

        for step in range(self.steps_per_update):
            train_observation = observation[: self.train_environments]
            epsilon_value = self._network.epsilon.get(
                frame_count,
                self.effective_frames,
            )

            with self.__autocast_float16():
                q_values_train, q_values_test, state_q_values, q_heads_train = (
                    self.__get_step_q_values(
                        observations=observation,
                        active_heads=active_heads,
                    )
                )
            actions_train_tensor, actions_test_tensor = self.get_action_tensors(
                q_values_train=q_values_train,
                q_values_test=q_values_test,
                epsilon_value=epsilon_value,
            )
            actions_train_cpu.copy_(actions_train_tensor)
            actions_test_cpu.copy_(actions_test_tensor)
            old_q_values = None
            if rollout_buffer.old_q_values is not None:
                if q_heads_train is None:
                    old_q_values = q_values_train.gather(
                        1,
                        actions_train_tensor.unsqueeze(1),
                    ).squeeze(1)
                else:
                    action_indices = actions_train_tensor.reshape(-1, 1, 1).expand(
                        -1,
                        self.__get_bootstrap_heads(),
                        1,
                    )
                    old_q_values = q_heads_train.gather(2, action_indices).squeeze(2)

            if test_environment is None:
                train_step, test_step = train_environment.step_split(
                    actions_train,
                    actions_test,
                )
                (
                    next_observation_train,
                    reward_train,
                    termination_train,
                    truncation_train,
                    info_train,
                ) = train_step
                (
                    next_observation_test,
                    reward_test,
                    termination_test,
                    truncation_test,
                    info_test,
                ) = test_step
            else:
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

            reward_train = reward_train.astype(numpy.float32, copy=False)
            reward_test = reward_test.astype(numpy.float32, copy=False)
            numpy.logical_or(
                termination_train,
                truncation_train,
                out=terminations[: self.train_environments],
            )
            numpy.logical_or(
                termination_test,
                truncation_test,
                out=terminations[self.train_environments :],
            )
            termination_train = terminations[: self.train_environments]

            score_rewards = self.__get_score_rewards(
                reward_train=reward_train,
                reward_test=reward_test,
                info_train=info_train,
                info_test=info_test,
                output=score_rewards_output,
            )
            completed_episodes = self.__get_episode_terminations(
                info_train=info_train,
                info_test=info_test,
                fallback_terminations=terminations,
                output=episode_terminations,
            )
            episode_returns += score_rewards
            self.__record_completed_episodes(
                episode_returns=episode_returns,
                terminations=completed_episodes,
            )
            self.__resample_terminated_bootstrap_heads(
                active_heads=active_heads,
                terminations=terminations,
            )

            if self.__bootstrapped_enabled():
                self._network.replace_activations()

            bootstrap_masks = None
            if rollout_buffer.bootstrap_masks is not None:
                bootstrap_masks = (
                    full_bootstrap_masks
                    if full_bootstrap_masks is not None
                    else self.__sample_bootstrap_masks()
                )

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

            self.__copy_array_to_tensor(
                observation[: self.train_environments],
                next_observation_train,
                torch.uint8,
            )
            self.__copy_array_to_tensor(
                observation[self.train_environments :],
                next_observation_test,
                torch.uint8,
            )
            frame_count += self.train_environments

        return observation, frame_count

    @torch.inference_mode()
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
                last_next_q_heads = self._network.get_q_heads(last_train_observation)
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
        mini_batch_old_q_values: Optional[torch.Tensor] = None,
        mini_batch_target_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(getattr(self._network, "distributional", False)):
            return self.__get_distributional_bootstrapped_loss(
                mini_batch_observations=mini_batch_observations,
                mini_batch_actions=mini_batch_actions,
                mini_batch_targets=mini_batch_targets,
                mini_batch_bootstrap_masks=mini_batch_bootstrap_masks,
                mini_batch_old_q_values=mini_batch_old_q_values,
                mini_batch_target_probs=mini_batch_target_probs,
            )

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

    def __get_distributional_bootstrapped_cross_entropy(
        self,
        *,
        logits: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        bins = logits.shape[-1]
        flattened_logits = logits.reshape(-1, bins)
        flattened_target_probs = target_probs.reshape(-1, bins)
        if logits.device.type == "cuda":
            with torch.autocast("cuda", enabled=False):
                loss = torch.nn.functional.cross_entropy(
                    flattened_logits,
                    flattened_target_probs,
                    reduction="none",
                )
        else:
            loss = torch.nn.functional.cross_entropy(
                flattened_logits,
                flattened_target_probs,
                reduction="none",
            )
        return loss.reshape(logits.shape[:-1])

    def __get_distributional_bootstrapped_loss(
        self,
        *,
        mini_batch_observations: torch.Tensor,
        mini_batch_actions: torch.Tensor,
        mini_batch_targets: torch.Tensor,
        mini_batch_bootstrap_masks: torch.Tensor,
        mini_batch_old_q_values: Optional[torch.Tensor],
        mini_batch_target_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        q_logits_heads = self._network.get_q_logits_heads(mini_batch_observations)
        action_indices = mini_batch_actions.reshape(-1, 1, 1, 1).expand(
            -1,
            self.__get_bootstrap_heads(),
            1,
            q_logits_heads.shape[-1],
        )
        q_logits_taken = q_logits_heads.gather(2, action_indices).squeeze(2)

        if mini_batch_target_probs is None:
            bins = q_logits_taken.shape[-1]
            loss = self._network.hl_gauss_loss(
                q_logits_taken.reshape(-1, bins),
                mini_batch_targets.reshape(-1),
                reduction="none",
            ).reshape(q_logits_taken.shape[:-1])
        else:
            loss = self.__get_distributional_bootstrapped_cross_entropy(
                logits=q_logits_taken,
                target_probs=mini_batch_target_probs,
            )

        value_clip = float(getattr(self, "distributional_value_clip", 0.0))
        if mini_batch_old_q_values is not None and value_clip > 0.0:
            bins = q_logits_taken.shape[-1]
            scalar_q_taken = self._network.hl_gauss_loss(
                q_logits_taken.reshape(-1, bins),
            ).reshape(q_logits_taken.shape[:-1])
            scalar_q_clipped = mini_batch_old_q_values + (
                scalar_q_taken - mini_batch_old_q_values
            ).clamp(-value_clip, value_clip)
            q_clipped_logprobs = self._network.hl_gauss_loss.transform_to_logprobs(
                scalar_q_clipped.reshape(-1)
            ).reshape(*scalar_q_clipped.shape, -1)
            if mini_batch_target_probs is None:
                clipped_loss = self._network.hl_gauss_loss(
                    q_clipped_logprobs.reshape(-1, bins),
                    mini_batch_targets.reshape(-1),
                    reduction="none",
                ).reshape(q_logits_taken.shape[:-1])
            else:
                clipped_loss = self.__get_distributional_bootstrapped_cross_entropy(
                    logits=q_clipped_logprobs,
                    target_probs=mini_batch_target_probs,
                )
            loss = torch.max(loss, clipped_loss)

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
        log_batches = bool(getattr(self, "loss_log_batches", False))
        update_losses = [] if log_batches else None
        update_loss_sum = None
        update_loss_count = 0
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
                            mini_batch_old_q_values=mini_batch_old_q_values,
                            mini_batch_target_probs=mini_batch_target_probs,
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
                detached_loss = loss.detach()
                if log_batches:
                    update_losses.append(detached_loss)
                else:
                    update_loss_sum = (
                        detached_loss
                        if update_loss_sum is None
                        else update_loss_sum + detached_loss
                    )
                    update_loss_count += 1

        if log_batches and update_losses:
            self.results.loss.extend(torch.stack(update_losses).float().cpu().tolist())
        elif update_loss_sum is not None and update_loss_count > 0:
            mean_loss = update_loss_sum / update_loss_count
            self.results.loss.append(float(mean_loss.float().cpu().item()))

    def __log_progress(self, *, update: int, frame_count: int):
        if update % self.verbose_interval != 0:
            return

        verbose_window = self.verbose_window
        test_score = (
            0.0
            if len(self.results.rewards.test) < verbose_window
            else numpy.mean(self.results.rewards.test[-verbose_window:])
        )

        self.flush_verbose(
            f"Update {update} | Frames: {frame_count * self.frame_skip:,}"
        )
        self.flush_verbose(f"Test Score: {test_score:.4f}")

    def _train(self, *, environment: str, seed: int):
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
                target_shape = flattened_targets.shape
                flattened_target_probs = self._network.hl_gauss_loss.transform_to_probs(
                    flattened_targets.reshape(-1)
                ).reshape(*target_shape, -1)

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
        if test_environment is not None:
            test_environment.close()
        self.results.duration = time.time() - training_start_time
