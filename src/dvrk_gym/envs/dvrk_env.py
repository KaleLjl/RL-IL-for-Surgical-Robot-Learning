# Author(s): Jiaqi Xu, Lele Chen (adaptation)
# Created on: 2020-11
# Last modified: 2025-06-24

"""
Base environment class for dVRK tasks, adapted from SurRoL.
This class handles the boilerplate for setting up a PyBullet simulation,
interfacing with the robot, and adhering to the Gymnasium API.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from ..robots import Psm
from ..utils.pybullet_utils import step

class DVRKEnv(gym.Env):
    """
    A Gymnasium Env wrapper for dVRK tasks.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: str = None):
        self.render_mode = render_mode
        
        # Connect to PyBullet
        if self.render_mode == 'human':
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        # PyBullet setup
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        
        # Robot and scene setup
        self.plane_id = p.loadURDF("plane.urdf", (0, 0, -0.001))
        self.psm1 = None
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        
        self._env_setup()
        
        # Gymnasium spaces
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self._duration = 0.25  # simulation seconds per step

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        step(self._duration)
        
        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_success(obs)
        truncated = False # Placeholder
        info = {'is_success': terminated}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        
        # Temporarily disable rendering to load scene faster
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        
        self.plane_id = p.loadURDF("plane.urdf", (0, 0, -0.001))
        self._env_setup()
        
        # Re-enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        obs = self._get_obs()
        return obs, {}

    def close(self):
        if self.cid >= 0:
            p.disconnect(self.cid)
            self.cid = -1

    def render(self):
        if self.render_mode == 'rgb_array':
            # Implement rendering to an RGB array
            # This is a placeholder
            return np.zeros((480, 640, 3), dtype=np.uint8)
        elif self.render_mode == 'human':
            # PyBullet GUI handles rendering, so nothing to do here
            pass

    # Methods to be implemented by subclasses
    # --------------------------------------------

    def _env_setup(self):
        """
        Sets up the environment.
        This method should be used to load robots, objects, etc.
        """
        raise NotImplementedError

    def _get_action_space(self) -> spaces.Space:
        """
        Returns the action space.
        """
        raise NotImplementedError

    def _get_observation_space(self) -> spaces.Space:
        """
        Returns the observation space.
        """
        raise NotImplementedError

    def _get_obs(self) -> dict:
        """
        Returns the current observation.
        """
        raise NotImplementedError

    def _set_action(self, action: np.ndarray):
        """
        Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, obs: dict) -> bool:
        """
        Indicates whether or not the task is successfully completed.
        """
        raise NotImplementedError

    def _get_reward(self, obs: dict) -> float:
        """
        Computes the reward for the current step.
        """
        raise NotImplementedError
