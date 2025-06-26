import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from imitation.data import types as imitation_types

class PPOWithBCLoss(PPO):
    """
    A custom PPO algorithm that incorporates a Behavioral Cloning (BC) loss
    term to leverage expert demonstrations.

    This is the core of our custom DAPG implementation.
    """

    def __init__(
        self,
        policy: Union[str, Type[th.nn.Module]],
        env: Union[GymEnv, str],
        expert_demonstrations: imitation_types.Transitions,
        bc_loss_weight: float = 0.1,
        bc_batch_size: int = 256,
        **kwargs: Any,
    ):
        """
        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...).
        :param env: The environment to learn from (if registered in Gym, can be str).
        :param expert_demonstrations: A Transitions object containing expert demonstrations.
        :param bc_loss_weight: The weight of the behavioral cloning loss.
        :param bc_batch_size: The batch size for the BC loss calculation.
        :param kwargs: Other arguments to pass to the PPO constructor.
        """
        super().__init__(policy=policy, env=env, **kwargs)
        
        self.expert_demonstrations = expert_demonstrations
        self.bc_loss_weight = bc_loss_weight
        self.bc_batch_size = bc_batch_size
        
        print("--- Custom PPO with BC Loss Initialized ---")
        print(f"BC Loss Weight: {self.bc_loss_weight}")
        print(f"BC Batch Size: {self.bc_batch_size}")

    def _sample_expert_batch(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        Samples a batch of observations and actions from the expert demonstrations.
        """
        n_samples = len(self.expert_demonstrations.obs)
        indices = np.random.randint(0, n_samples, self.bc_batch_size)
        
        obs = th.as_tensor(self.expert_demonstrations.obs[indices]).to(self.device)
        acts = th.as_tensor(self.expert_demonstrations.acts[indices]).to(self.device)
        
        return obs, acts

    def train(self) -> None:
        """
        The standard PPO training method, modified to include a BC loss term.
        This method is largely a copy of the original Stable-Baselines3 `PPO.train()`
        method, with the BC loss calculation integrated.
        """
        self.policy.set_training_mode(True)
        # The learn method of the PPO class updates the schedules
        # so we don't need to do it here.

        # --- 1. PPO Loss Calculation ---
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                # Unpack the schedule to get the current float value
                current_clip_range = self.clip_range(self._current_progress_remaining)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is None:
                    # Unclipped value loss
                    values_pred = values
                else:
                    # Clipped value loss
                    current_clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -current_clip_range_vf, current_clip_range_vf
                    )
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # --- 2. Behavioral Cloning Loss Calculation ---
                expert_obs, expert_acts = self._sample_expert_batch()
                _, expert_log_prob, _ = self.policy.evaluate_actions(expert_obs, expert_acts)
                bc_loss = -th.mean(expert_log_prob)

                # --- 3. Combine all losses ---
                total_loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.bc_loss_weight * bc_loss
                )

                # Optimization step
                self.policy.optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        
        # --- Logging ---
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/bc_loss", bc_loss.item())
        if len(approx_kl_divs) > 0:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_range", self.clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self.clip_range_vf)
