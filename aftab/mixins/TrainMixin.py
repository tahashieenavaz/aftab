import torch
import numpy
import time
from ..functions import flush
from ..functions import random_shifts
from ..functions import lambda_returns_quantile


class TrainMixin:
    def __init__(self):
        super().__init__()

    def __initialize_training(self, environment: str, seed: int):
        self.flush_results()
        self.set_precision()
        self.set_seed(seed)

        if getattr(self, "reward_centering"):
            self._average_reward = 0.0

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

    def __allocate_buffers(
        self,
        *,
        observation_shape: tuple,
        action_dimension: int,
        is_distributional: bool,
    ):
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

        batch_q = None
        batch_quantiles = None
        if not is_distributional:
            batch_q = torch.empty(
                (self.steps_per_update, self.total_environments, action_dimension),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            batch_quantiles = torch.empty(
                (self.steps_per_update, self.total_environments, self.number_quantiles),
                dtype=torch.float32,
                device=self.device,
            )

        return (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
            batch_quantiles,
        )

    def __collect_trajectories(
        self,
        *,
        is_distributional: bool,
        frame_count: int,
        observation,
        train_environment,
        test_environment,
        episode_returns,
        batch_observations,
        batch_actions,
        batch_rewards,
        batch_terminations,
        batch_q,
        batch_quantiles,
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

            if not is_distributional:
                q_values = self.get_q_values(
                    float_train_observations=float_train_observations,
                    float_test_observations=float_test_observations,
                    gradient=False,
                )
            else:
                q_values, quantiles = self.get_q_and_quantiles(
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
            rewards = numpy.concatenate([reward_train, reward_test], axis=0)
            terminations = numpy.concatenate(
                [termination_train, termination_test], axis=0
            )
            truncations = numpy.concatenate([truncation_train, truncation_test], axis=0)
            terminations = numpy.logical_or(terminations, truncations)

            reward_train = info_train.get("reward", None)
            reward_test = info_test.get("reward", None)
            if reward_train is not None and reward_test is not None:
                if isinstance(reward_train, numpy.ndarray) and isinstance(
                    reward_test, numpy.ndarray
                ):
                    if reward_train.ndim == 0:
                        rewards = numpy.stack([reward_train, reward_test])
                    else:
                        rewards = numpy.concatenate([reward_train, reward_test], axis=0)
                    episode_returns += rewards

            if getattr(self, "reward_centering"):
                reward_centering_beta = getattr(self, "reward_centering_beta")
                mean_step_reward = numpy.mean(rewards)
                reward_difference = mean_step_reward - self._average_reward
                self._average_reward += reward_centering_beta * reward_difference
                rewards = rewards - self._average_reward

            done_mask = terminations
            if numpy.any(done_mask):
                idx = numpy.nonzero(done_mask)[0]
                scores = episode_returns[done_mask]
                split = idx < self.train_environments
                self.results.rewards.train.extend(scores[split].tolist())
                self.results.rewards.test.extend(scores[~split].tolist())
                episode_returns[done_mask] = 0

            batch_observations[step] = observation
            batch_actions[step] = torch.from_numpy(actions).to(self.device)
            batch_rewards[step] = torch.from_numpy(rewards).to(self.device)
            batch_terminations[step] = torch.from_numpy(terminations).to(self.device)

            if not is_distributional:
                batch_q[step] = torch.cat([q_values["train"], q_values["test"]], dim=0)
            else:
                action_idx = (
                    batch_actions[step]
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, self.number_quantiles, -1)
                )
                chosen_quantiles = quantiles.gather(2, action_idx).squeeze(-1)
                batch_quantiles[step] = chosen_quantiles

            observation = (
                torch.from_numpy(next_observation).to(torch.uint8).to(self.device)
            )
            frame_count += self.train_environments

        return observation, frame_count

    def __compute_targets(
        self,
        *,
        is_distributional: bool,
        observation,
        batch_q,
        batch_rewards,
        batch_terminations,
        batch_quantiles,
    ):
        float_obs = observation.float()

        random_shift_k = getattr(self, "random_shift_k")
        if not is_distributional:
            if getattr(self, "random_shift") and random_shift_k > 0:
                B = float_obs.shape[0]
                q_sum = 0
                for _ in range(random_shift_k):
                    shifts = torch.randint(
                        0,
                        2 * self.random_shift_padding + 1,
                        size=(2, B),
                        device=self.device,
                    )
                    observation_k = random_shifts(
                        observation=float_obs,
                        width_shifts=shifts[0],
                        height_shifts=shifts[1],
                        padding=self.random_shift_padding,
                    )
                    q_sum += self.get_q_values(
                        float_observations=observation_k, gradient=False
                    )
                next_q_values = q_sum / random_shift_k
            else:
                next_q_values = self.get_q_values(
                    float_observations=float_obs, gradient=False
                )

            targets = self.get_returns(
                float_observations=float_obs,
                batch_q=batch_q,
                next_q_values=next_q_values,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
            )
        else:
            if getattr(self, "random_shift") and random_shift_k > 0:
                B = float_obs.shape[0]
                q_sum = 0
                quantiles_sum = 0
                for _ in range(random_shift_k):
                    shifts = torch.randint(
                        0,
                        2 * self.random_shift_padding + 1,
                        size=(2, B),
                        device=self.device,
                    )
                    shifted_observations = random_shifts(
                        observation=float_obs,
                        width_shifts=shifts[0],
                        height_shifts=shifts[1],
                        padding=self.random_shift_padding,
                    )
                    q, quantiles = self.get_q_and_quantiles(
                        float_observations=shifted_observations
                    )
                    q_sum += q
                    quantiles_sum += quantiles

                next_q_values = q_sum / random_shift_k
                next_quantiles_all = quantiles_sum / random_shift_k
            else:
                next_q_values, next_quantiles_all = self.get_q_and_quantiles(
                    float_observations=float_obs
                )

            next_action = next_q_values.argmax(dim=-1, keepdim=True)
            next_action_idx = next_action.unsqueeze(1).expand(
                -1, self.number_quantiles, -1
            )
            next_quantiles = next_quantiles_all.gather(2, next_action_idx).squeeze(-1)

            q_seq_for_lambda = torch.cat(
                [batch_quantiles[1:], next_quantiles.unsqueeze(0)], dim=0
            )

            with torch.no_grad():
                targets = lambda_returns_quantile(
                    batch_rewards,
                    batch_terminations,
                    q_seq_for_lambda,
                    self.gamma,
                    self.lmbda,
                )
        return targets

    def __flatten_batches(
        self,
        *,
        is_distributional: bool,
        batch_observations,
        batch_actions,
        targets,
        observation_shape,
    ):
        train_slice = slice(0, self.train_environments)
        flattened_observations = batch_observations[
            :, : self.train_environments
        ].reshape((-1,) + observation_shape)
        flattened_actions = batch_actions[:, train_slice].reshape(-1)

        if not is_distributional:
            flattened_targets = targets[:, train_slice].reshape(-1)
        else:
            flattened_targets = targets[:, train_slice].reshape(
                -1, self.number_quantiles
            )
        return flattened_observations, flattened_actions, flattened_targets

    def __update_network(
        self,
        *,
        is_distributional: bool,
        optimizer,
        scaler,
        flattened_observations,
        flattened_actions,
        flattened_targets,
    ):
        self._network.train()
        for _ in range(self.epochs):
            indices = torch.randperm(self.batch_size, device=self.device)

            for range_start in range(0, self.batch_size, self.mini_batch_size):
                range_end = range_start + self.mini_batch_size
                mini_batch_idx = indices[range_start:range_end]

                mini_batch_observations = flattened_observations[mini_batch_idx]
                mini_batch_actions = flattened_actions[mini_batch_idx]
                mini_batch_targets = flattened_targets[mini_batch_idx]

                random_shift_m = getattr(self, "random_shift_m")
                if getattr(self, "random_shift") and random_shift_m > 0:
                    mini_batch_observations = mini_batch_observations.repeat(
                        random_shift_m, 1, 1, 1
                    )
                    mini_batch_actions = mini_batch_actions.repeat(random_shift_m)
                    mini_batch_targets = (
                        mini_batch_targets.repeat(random_shift_m, 1)
                        if is_distributional
                        else mini_batch_targets.repeat(random_shift_m)
                    )

                    current_mini_batch_size = mini_batch_observations.shape[0]
                    shifts = torch.randint(
                        0,
                        2 * self.random_shift_padding + 1,
                        size=(2, current_mini_batch_size),
                        device=self.device,
                    )
                    mini_batch_observations = random_shifts(
                        observation=mini_batch_observations.float(),
                        width_shifts=shifts[0],
                        height_shifts=shifts[1],
                        padding=self.random_shift_padding,
                    )

                if not is_distributional:
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.get_loss(
                        mini_batch_observations=mini_batch_observations,
                        mini_batch_targets=mini_batch_targets,
                        mini_batch_actions=mini_batch_actions,
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(), self.gradient_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    self.results.loss.append(loss.item())
                else:
                    optimizer.quantile_value.zero_grad(set_to_none=True)
                    optimizer.fraction_proposal.zero_grad(set_to_none=True)

                    quantile_loss, fraction_loss = self.get_distributional_loss(
                        mini_batch_observations=mini_batch_observations,
                        mini_batch_actions=mini_batch_actions,
                        mini_batch_targets=mini_batch_targets,
                    )

                    total_loss = quantile_loss + fraction_loss
                    scaler.scale(total_loss).backward()

                    scaler.unscale_(optimizer.quantile_value)
                    scaler.unscale_(optimizer.fraction_proposal)

                    torch.nn.utils.clip_grad_norm_(
                        list(self._network.phi.parameters())
                        + list(self._network.quantile_value.parameters()),
                        self.gradient_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self._network.fraction_proposal.parameters(), self.gradient_norm
                    )

                    scaler.step(optimizer.quantile_value)
                    scaler.step(optimizer.fraction_proposal)
                    scaler.update()

                    self.results.loss.append(quantile_loss.item())

    def __log_progress(self, *, update: int, frame_count: int):
        verbose_window = self.verbose_window
        test_score = (
            0.0
            if len(self.results.rewards.test) < verbose_window
            else numpy.mean(self.results.rewards.test[-verbose_window:])
        )

        if update % self.verbose_interval == 0:
            self.flush_verbose(f"Update {update} | Frames: {frame_count}")
            self.flush_verbose(f"Test Score: {test_score:.4f}")

    def train_loop(self, *, environment: str, seed: int):
        self.flush_verbose("Training started.")

        frame_count = 0
        is_distributional = self.network not in ["q", "duelling"]

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
            batch_q,
            batch_quantiles,
        ) = self.__allocate_buffers(
            observation_shape=observation_shape,
            action_dimension=action_dimension,
            is_distributional=is_distributional,
        )

        training_start_time = time.time()

        for update in range(1, self.total_updates + 1):
            self._network.eval()

            observation, frame_count = self.__collect_trajectories(
                is_distributional=is_distributional,
                frame_count=frame_count,
                observation=observation,
                train_environment=train_environment,
                test_environment=test_environment,
                episode_returns=episode_returns,
                batch_observations=batch_observations,
                batch_actions=batch_actions,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
                batch_q=batch_q,
                batch_quantiles=batch_quantiles,
            )

            targets = self.__compute_targets(
                is_distributional=is_distributional,
                observation=observation,
                batch_q=batch_q,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
                batch_quantiles=batch_quantiles,
            )

            (
                flattened_observations,
                flattened_actions,
                flattened_targets,
            ) = self.__flatten_batches(
                is_distributional=is_distributional,
                batch_observations=batch_observations,
                batch_actions=batch_actions,
                targets=targets,
                observation_shape=observation_shape,
            )

            self.__update_network(
                is_distributional=is_distributional,
                optimizer=optimizer,
                scaler=scaler,
                flattened_observations=flattened_observations,
                flattened_actions=flattened_actions,
                flattened_targets=flattened_targets,
            )

            self.__log_progress(update=update, frame_count=frame_count)

        train_environment.close()
        test_environment.close()
        self.results.duration = time.time() - training_start_time
