import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

class FlattenDictObsWrapper(gym.ObservationWrapper):
    """
    A wrapper to flatten the dictionary observation space of our dVRK environments.

    This wrapper concatenates the 'observation', 'achieved_goal', and
    'desired_goal' keys into a single flat numpy array. This is necessary
    to match the data format used for training the BC agent and to work
    with standard SB3 policies that expect a Box observation space.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Define the new, flattened observation space.
        # We concatenate the shapes of the original Dict space's components.
        original_space = env.observation_space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                original_space["observation"].shape[0]
                + original_space["achieved_goal"].shape[0]
                + original_space["desired_goal"].shape[0],
            ),
            dtype=np.float32,
        )

    def observation(self, obs: dict) -> np.ndarray:
        """
        Flattens the observation dictionary into a single numpy array.
        """
        return np.concatenate(
            [
                obs["observation"],
                obs["achieved_goal"],
                obs["desired_goal"],
            ]
        ).astype(np.float32)
