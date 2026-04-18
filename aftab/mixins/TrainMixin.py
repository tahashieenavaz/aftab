import torch
import numpy
import time
from ..functions import flush, lambda_returns_quantile


class TrainMixin:
    def __init__(self):
        super().__init__()

    def _initialize_training(self, environment: str, seed: int):
        self.flush_results()
        self.set_precision()
        self.set_seed(seed)

        if getattr(self, "reward_centering", False):
            self._average_reward = 0.0

        train_environment, test_environment, action_dimension, observation_shape = (
            self.make_environments(environment=environment, seed=seed)
        )
        self.prepare_network(action_dimension=action_dimension)
        optimizer = self.make_optimizer()
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        observation_train, _ = train_environment.reset()
        observation_test, _ = test_environment.reset()
        observation = numpy.concatenate([observation_train, observation_test], axis=0)
        observation = torch.from_numpy(observation).to(torch.uint8).to(self.device)

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

    def _allocate_buffers(self, observation_shape, action_dimension, is_regression):
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
        if is_regression:
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

    def _collect_trajectories(
        self,
        is_regression,
        frame_count,
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
            float_observations = observation.float()
            epsilon_value = self._network.epsilon.get(
                frame_count,
                self.actual_frames,
            )

            if is_regression:
                q_values = self.get_q_values(
                    float_observations=float_observations, gradient=False
                )
            else:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        features = self._network.phi(float_observations)
                        tau, tau_hat, q_probs, _ = self._network.fraction_proposal(
                            features
                        )
                        quantiles = self._network.quantile_value(features, tau_hat)
                        q_values = (q_probs.unsqueeze(-1) * quantiles).sum(dim=1)

            actions = self.get_actions(q_values=q_values, epsilon_value=epsilon_value)
            actions_train, actions_test = self.split_actions(actions)

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

            if getattr(self, "reward_centering", False):
                reward_centering_beta = getattr(self, "reward_centering_beta", 0.01)
                mean_step_reward = numpy.mean(rewards)
                reward_difference = mean_step_reward - self._average_reward
                self._average_reward += reward_centering_beta * reward_difference
                rewards = rewards - self._average_reward

            done_mask = terminations
            if numpy.any(done_mask):
                idx = numpy.nonzero(done_mask)[0]
                scores = episode_returns[done_mask]
                split = idx < self.num_train_environments
                self.results.rewards.train.extend(scores[split].tolist())
                self.results.rewards.test.extend(scores[~split].tolist())
                episode_returns[done_mask] = 0

            batch_observations[step] = observation
            batch_actions[step] = torch.from_numpy(actions).to(self.device)
            batch_rewards[step] = torch.from_numpy(rewards).to(self.device)
            batch_terminations[step] = torch.from_numpy(terminations).to(self.device)

            if is_regression:
                batch_q[step] = q_values
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
            frame_count += self.num_train_environments

        return observation, frame_count

    def _compute_targets(
        self,
        is_regression,
        observation,
        batch_q,
        batch_rewards,
        batch_terminations,
        batch_quantiles,
    ):
        if is_regression:
            targets = self.get_returns(
                float_observations=observation.float(),
                batch_q=batch_q,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
            )
        else:
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    next_float_observations = observation.float()
                    next_features = self._network.phi(next_float_observations)
                    next_tau, next_tau_hat, next_q_probs, _ = (
                        self._network.fraction_proposal(next_features)
                    )
                    next_quantiles_all = self._network.quantile_value(
                        next_features, next_tau_hat
                    )
                    next_q_values = (
                        next_q_probs.unsqueeze(-1) * next_quantiles_all
                    ).sum(dim=1)

                    next_action = next_q_values.argmax(dim=-1, keepdim=True)
                    next_action_idx = next_action.unsqueeze(1).expand(
                        -1, self.number_quantiles, -1
                    )
                    next_quantiles = next_quantiles_all.gather(
                        2, next_action_idx
                    ).squeeze(-1)

            q_seq_for_lambda = torch.cat(
                [batch_quantiles[1:], next_quantiles.unsqueeze(0)], dim=0
            )

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    targets = lambda_returns_quantile(
                        batch_rewards,
                        batch_terminations,
                        q_seq_for_lambda,
                        self.gamma,
                        self.lmbda,
                    )
        return targets

    def _flatten_batches(
        self,
        is_regression,
        batch_observations,
        batch_actions,
        targets,
        observation_shape,
    ):
        train_slice = slice(0, self.num_train_environments)
        flattened_observations = batch_observations[
            :, : self.num_train_environments
        ].reshape((-1,) + observation_shape)
        flattened_actions = batch_actions[:, train_slice].reshape(-1)

        if is_regression:
            flattened_targets = targets[:, train_slice].reshape(-1)
        else:
            flattened_targets = targets[:, train_slice].reshape(
                -1, self.number_quantiles
            )
        return flattened_observations, flattened_actions, flattened_targets

    def _update_network(
        self,
        is_regression,
        optimizer,
        scaler,
        flattened_observations,
        flattened_actions,
        flattened_targets,
    ):
        self._network.train()
        for _ in range(self.epochs):
            indices = torch.randperm(self.batch_size, device=self.device)

            for range_start in range(0, self.batch_size, self.minibatch_size):
                range_end = range_start + self.minibatch_size
                mini_batch_idx = indices[range_start:range_end]
                mini_batch_observations = flattened_observations[mini_batch_idx]
                mini_batch_actions = flattened_actions[mini_batch_idx]
                mini_batch_targets = flattened_targets[mini_batch_idx]

                if is_regression:
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

                    with torch.autocast(
                        device_type=self.device.type, dtype=torch.float16
                    ):
                        mini_batch_observations_float = mini_batch_observations.float()
                        features = self._network.phi(mini_batch_observations_float)
                        tau, tau_hat, q_probs, entropy = (
                            self._network.fraction_proposal(features.detach())
                        )
                        quantiles = self._network.quantile_value(features, tau_hat)

                        action_idx = (
                            mini_batch_actions.unsqueeze(1)
                            .unsqueeze(2)
                            .expand(-1, self.number_quantiles, -1)
                        )
                        current_quantiles = quantiles.gather(2, action_idx).squeeze(-1)

                        u = mini_batch_targets.unsqueeze(
                            1
                        ) - current_quantiles.unsqueeze(2)
                        huber_loss = torch.nn.functional.huber_loss(
                            u, torch.zeros_like(u), reduction="none", delta=1.0
                        )

                        tau_hat_expanded = tau_hat.unsqueeze(2).expand(
                            -1, -1, self.number_quantiles
                        )
                        asym_weights = torch.abs(tau_hat_expanded - (u < 0).float())
                        quantile_loss = (
                            (asym_weights * huber_loss).sum(dim=1).mean(dim=1).mean()
                        )

                        with torch.no_grad():
                            quantiles_tau = self._network.quantile_value(
                                features.detach(), tau[:, 1:-1]
                            )
                            action_idx_tau = (
                                mini_batch_actions.unsqueeze(1)
                                .unsqueeze(2)
                                .expand(-1, self.number_quantiles - 1, -1)
                            )
                            Z_tau = quantiles_tau.gather(2, action_idx_tau).squeeze(-1)

                        Z_tau_hat = current_quantiles.detach()
                        gradients_tau = 2 * Z_tau - Z_tau_hat[:, :-1] - Z_tau_hat[:, 1:]

                        entropy_coeff = getattr(self, "entropy_coef", 0.0)
                        fraction_loss = (tau[:, 1:-1] * gradients_tau.detach()).sum(
                            dim=1
                        ).mean() - entropy_coeff * entropy.mean()

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

    def _log_progress(self, update, frame_count):
        test_score = (
            0.0
            if len(self.results.rewards.test) < 10
            else numpy.mean(self.results.rewards.test[-10:])
        )

        if self.verbose and update % self.log_interval == 0:
            flush(f"Update {update} | Frames: {frame_count}")
            flush(
                f"Test Score: {test_score:.4f}",
            )

    def _train_loop(self, environment: str, seed: int):
        frame_count = 0
        is_regression = self.network in ["regression", "duelling"]

        (
            train_environment,
            test_environment,
            action_dimension,
            observation_shape,
            optimizer,
            scaler,
            observation,
            episode_returns,
        ) = self._initialize_training(environment, seed)

        (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
            batch_quantiles,
        ) = self._allocate_buffers(observation_shape, action_dimension, is_regression)

        training_start_time = time.time()

        for update in range(1, self.total_updates + 1):
            self._network.eval()

            observation, frame_count = self._collect_trajectories(
                is_regression,
                frame_count,
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
            )

            targets = self._compute_targets(
                is_regression,
                observation,
                batch_q,
                batch_rewards,
                batch_terminations,
                batch_quantiles,
            )

            (
                flattened_observations,
                flattened_actions,
                flattened_targets,
            ) = self._flatten_batches(
                is_regression,
                batch_observations,
                batch_actions,
                targets,
                observation_shape,
            )

            self._update_network(
                is_regression,
                optimizer,
                scaler,
                flattened_observations,
                flattened_actions,
                flattened_targets,
            )

            self._log_progress(update, frame_count)

        train_environment.close()
        test_environment.close()
        self.results.duration = time.time() - training_start_time
