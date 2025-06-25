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
        Override the standard PPO training method to include the BC loss.
        
        The method performs a standard PPO update, and then a separate BC
        update on a batch of expert data.
        """
        # Standard PPO training step
        super().train()
        
        # --- Behavioral Cloning Loss Step ---
        
        # Sample a batch of expert data
        expert_obs, expert_acts = self._sample_expert_batch()
        
        # Get the policy's action distribution for the expert observations
        # and calculate the log probability of the expert actions.
        _, log_prob, _ = self.policy.evaluate_actions(expert_obs, expert_acts)
        
        # The BC loss is the negative log likelihood of the expert actions.
        bc_loss = -th.mean(log_prob)
        
        # Combine with the weight
        total_bc_loss = self.bc_loss_weight * bc_loss
        
        # Log the BC loss
        self.logger.record("train/bc_loss", bc_loss.item())
        
        # Perform a gradient step to minimize the BC loss
        self.policy.optimizer.zero_grad()
        total_bc_loss.backward()
        self.policy.optimizer.step()
