# Author(s): Lele Chen
# Created on: 2025-06-24
# Last modified: 2025-06-24

"""
Needle Reach Task Environment for the dVRK.
This task requires the robot to move its end-effector to a target needle.
"""
import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from .dvrk_env import DVRKEnv
from ..robots import Psm
from ..utils.pybullet_utils import get_body_pose, get_link_pose

# Define the asset path relative to this file
ASSET_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets')

class NeedleReachEnv(DVRKEnv):
    """
    Gymnasium environment for the dVRK Needle Reach task.
    """
    def __init__(self, render_mode: str = None):
        super().__init__(render_mode=render_mode)
        
        self.success_threshold = 0.05  # 5 cm

    def _env_setup(self):
        """
        Loads the robot, table, and needle into the simulation.
        """
        # Load robot
        self.psm1 = Psm(pos=[0.2, 0, 0.1524], scaling=0.7)

        # Load table
        table_path = os.path.join(ASSET_DIR_PATH, 'table/table.urdf')
        p.loadURDF(table_path, [0.0, 0.0, 0.0], useFixedBase=True)

        # Load needle
        needle_path = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        self.needle_id = p.loadURDF(needle_path, useFixedBase=True)
        self.obj_ids['fixed'].append(self.needle_id)
        
        # Reset needle position
        self._reset_needle()

    def _get_action_space(self) -> spaces.Space:
        # Action is a 6-DoF delta pose (dx, dy, dz, d_roll, d_pitch, d_yaw)
        return spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def _get_observation_space(self) -> spaces.Space:
        # Observation is a dictionary containing robot and task states
        obs_shape = len(self._get_obs_robot_state())
        return spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(obs_shape,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        robot_state = self._get_obs_robot_state()
        eef_pos, _ = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)
        needle_pos, _ = get_body_pose(self.needle_id)

        return {
            'observation': robot_state,
            'achieved_goal': np.array(eef_pos, dtype=np.float32),
            'desired_goal': np.array(needle_pos, dtype=np.float32),
        }

    def _set_action(self, action: np.ndarray):
        # Convert delta pose to absolute pose
        current_pose_matrix = self.psm1.get_current_position()
        
        # Apply delta position
        delta_pos = action[:3] * 0.05  # Scale down the movement
        current_pose_matrix[:3, 3] += delta_pos
        
        # For this task, we ignore orientation changes for simplicity
        # delta_orn = action[3:]
        
        self.psm1.move(current_pose_matrix)

    def _is_success(self, obs: dict) -> bool:
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return dist < self.success_threshold

    def _get_reward(self, obs: dict) -> float:
        # Sparse reward
        return 1.0 if self._is_success(obs) else 0.0

    def _get_obs_robot_state(self):
        """
        Returns the current state of the robot.
        (joint positions, end-effector position)
        """
        joint_positions = self.psm1.get_current_joint_position()
        eef_pos, _ = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)
        return np.concatenate([
            np.array(joint_positions, dtype=np.float32),
            np.array(eef_pos, dtype=np.float32)
        ])

    def _reset_needle(self):
        """
        Resets the needle to a random position on the table.
        """
        # Define the workspace area on the table
        x_range = [-0.1, 0.1]
        y_range = [-0.1, 0.1]
        z_pos = 0.01 # Slightly above the table
        
        random_x = self.np_random.uniform(x_range[0], x_range[1])
        random_y = self.np_random.uniform(y_range[0], y_range[1])
        
        p.resetBasePositionAndOrientation(self.needle_id, [random_x, random_y, z_pos], [0, 0, 0, 1])
