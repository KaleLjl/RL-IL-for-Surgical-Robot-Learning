# Author(s): Jiaqi Xu, Lele Chen (adaptation)
# Created on: 2020-11
# Last modified: 2025-06-24

"""
Base environment class for dVRK tasks, adapted from SurRoL.
This class handles the boilerplate for setting up a PyBullet simulation,
interfacing with the robot, and adhering to the Gymnasium API.
"""
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from ..robots import Psm
from ..utils.pybullet_utils import step, render_image

RENDER_HEIGHT = 480
RENDER_WIDTH = 640
ASSET_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets')

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
        
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # Create a temporary scene to correctly initialize observation and action spaces
        # This scene will be rebuilt in the first call to reset()
        self._pre_setup()

        # Setup spaces
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self._duration = 0.2  # simulation seconds per step, aligned with SurRoL

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        step(self._duration)
        
        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_success(obs)
        
        # Check for early termination (if implemented by subclass)
        truncated = False
        if hasattr(self, '_check_early_termination'):
            truncated = self._check_early_termination()
        
        info = {'is_success': terminated}
        if truncated:
            info['early_exit'] = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Re-create the scene on every reset, matching SurRoL's behavior
        self._pre_setup()
        
        # Sample a new goal
        self.goal = self._sample_goal()
        self._sample_goal_callback()

        # Re-enable rendering and set camera
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._setup_camera()
        
        obs = self._get_obs()
        return obs, {}

    def close(self):
        if self.cid >= 0:
            p.disconnect(self.cid)
            self.cid = -1

    def render(self):
        if self.render_mode == 'rgb_array':
            # Re-setup camera before rendering to get latest view matrix
            self._setup_camera()
            # Render an RGB image from the simulation
            rgb_array, _ = render_image(
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
                view_matrix=self._view_matrix,
                proj_matrix=self._proj_matrix
            )
            return rgb_array
        elif self.render_mode == 'human':
            # PyBullet GUI handles rendering, so nothing to do here
            pass

    def _setup_camera(self):
        """
        Sets up the camera view and projection matrices.
        Called in reset() and render() to ensure the camera is correctly positioned.
        """
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=0.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=45,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=20.0
        )
        if self.render_mode == 'human':
            p.resetDebugVisualizerCamera(
                cameraDistance=0.81 * self.SCALING,
                cameraYaw=90,
                cameraPitch=-30,
                cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING)
            )

    def _pre_setup(self):
        """
        A helper function to set up the scene for both __init__ and reset.
        This is necessary to ensure spaces are defined correctly at initialization time.
        """
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Disable rendering during setup

        # Load table
        # Note: self.SCALING must be defined in the subclass BEFORE super().__init__
        p.loadURDF(os.path.join(ASSET_DIR_PATH, 'table/table.urdf'),
                   (0.5, 0, 0.0), p.getQuaternionFromEuler((0, 0, 0)),
                   globalScaling=getattr(self, 'SCALING', 1.0))
        
        # Setup subclass-specific environment (creates robot, loads objects)
        self._env_setup()

        # Load goal visualization sphere
        self.goal_vis_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                      globalScaling=getattr(self, 'SCALING', 1.0))
        self.obj_ids['fixed'].append(self.goal_vis_id)


    # Methods to be implemented by subclasses
    # --------------------------------------------

    def _sample_goal(self) -> np.ndarray:
        """
        Samples a new goal and returns it.
        """
        raise NotImplementedError

    def _sample_goal_callback(self):
        """
        A custom callback that is called after sampling a new goal.
        Can be used to implement custom visualizations, like moving a goal marker.
        """
        pass

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
