import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict

from imitation.algorithms import bc
from imitation.data import types
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

"""
This script creates a minimal, self-contained reproducible example to diagnose a
paradoxical ValueError in the imitation library.

Problem:
The `train_bc.py` script fails with `ValueError: expected one more observations
than actions: 3 != 8 + 1`, even though the expert data is verified to be correct
(len(obs) == len(acts) + 1).

Hypothesis:
The error is not in our data but in how the `imitation` library handles
`gym.spaces.Dict` observation spaces. The strange error message (reporting obs
length as 3) suggests it might be misinterpreting the dictionary structure.

Goal:
Trigger the same error in a minimal environment to confirm if this is a library
bug or a subtle interaction with our specific environment's implementation.
"""

# 1. Define a minimal environment with a Dict observation space
class MinimalDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = Dict({
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "observation": Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        })
        self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Return a dummy observation
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

# 2. Create a tiny, but CORRECTLY FORMATTED, expert dataset
# We will create a single trajectory.
# Rule: len(obs) == len(acts) + 1
dummy_obs_1 = {
    "achieved_goal": np.array([1, 1, 1], dtype=np.float32),
    "desired_goal": np.array([2, 2, 2], dtype=np.float32),
    "observation": np.random.rand(10).astype(np.float32),
}
dummy_obs_2 = {
    "achieved_goal": np.array([1.1, 1.1, 1.1], dtype=np.float32),
    "desired_goal": np.array([2, 2, 2], dtype=np.float32),
    "observation": np.random.rand(10).astype(np.float32),
}
dummy_acts = [np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)]
dummy_rews = [0.0]  # One reward for each action
dummy_infos = [{}]  # infos can be empty

# The trajectory must have one more observation than action.
# Here we have 2 observations and 1 action. This is correct.
trajectory = types.TrajectoryWithRew(
    obs=np.array([dummy_obs_1, dummy_obs_2]),
    acts=np.array(dummy_acts),
    rews=np.array(dummy_rews, dtype=np.float32),
    infos=np.array(dummy_infos),
    terminal=True,
)

print("--- Data Verification ---")
print(f"Observations length: {len(trajectory.obs)}")
print(f"Actions length: {len(trajectory.acts)}")
print(f"Is data valid? (len(obs) == len(acts) + 1): {len(trajectory.obs) == len(trajectory.acts) + 1}")
print("-------------------------")

# 3. Instantiate the environment and the BC trainer
rng = np.random.default_rng()
env = MinimalDictEnv()

# The policy MUST be MultiInputActorCriticPolicy for Dict spaces
policy = MultiInputActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: 0.001,
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=[trajectory],
    policy=policy,
    rng=rng,
    batch_size=1,  # Set batch size to 1 to match our tiny dataset
)

# 4. Create a DataLoader
# This is the most robust way to feed data to the trainer.
class MinimalDataset(torch.utils.data.Dataset):
    def __init__(self, trajectory):
        self.obs = []
        self.acts = []
        # Create (obs, act) pairs from the trajectory
        for i in range(len(trajectory.acts)):
            self.obs.append(trajectory.obs[i])
            self.acts.append(trajectory.acts[i])

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, i):
        return self.obs[i], self.acts[i]

def minimal_collate_fn(batch):
    obs_list, acts_list = zip(*batch)
    obs_keys = obs_list[0].keys()
    obs_dict_of_arrays = {
        key: np.array([d[key] for d in obs_list]) for key in obs_keys
    }
    return {"obs": obs_dict_of_arrays, "acts": np.array(acts_list)}

dataset = MinimalDataset(trajectory)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=minimal_collate_fn
)

# Re-initialize the trainer to use the DataLoader
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=data_loader,
    policy=policy,
    rng=rng,
)

# 5. Run the training and see if it fails
print("\nAttempting to train BC agent with DataLoader...")
try:
    bc_trainer.train(n_epochs=1)
    print("\nSUCCESS: BC training completed without error.")
except ValueError as e:
    print(f"\nFAILURE: Caught expected ValueError.")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
    print("\nThis confirms the issue is within the `imitation` library's handling of Dict spaces.")
except Exception as e:
    print(f"\nFAILURE: Caught unexpected error.")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
