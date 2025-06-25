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
    Aligned with the original SurRoL implementation parameters.
    """
    # Constants from original SurRoL implementation
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.
    
    # Constants from the parent PsmEnv in SurRoL
    POSE_PSM1 = ((0.05, 0.24, 0.8524), (0, 0, -(90 + 20) / 180 * np.pi))
    DISTANCE_THRESHOLD = 0.005

    def __init__(self, render_mode: str = None):
        # Correctly initialize workspace limits with offset before scaling
        workspace_limits = np.asarray(self.WORKSPACE_LIMITS) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))
        self.workspace_limits = workspace_limits * self.SCALING
        
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING
        
        super().__init__(render_mode=render_mode)
        
        self.success_threshold = self.distance_threshold

    def _env_setup(self):
        """
        Loads the robot, tray, and needle into the simulation, aligned with SurRoL parameters.
        """
        # Load robot - Pass unscaled position to constructor
        psm_pos = np.array(self.POSE_PSM1[0])
        psm_orn_eul = self.POSE_PSM1[1]
        psm_orn_quat = p.getQuaternionFromEuler(psm_orn_eul)
        self.psm1 = Psm(pos=psm_pos, orn=psm_orn_quat, scaling=self.SCALING)
        
        # Reset robot to a start pose within the workspace, using original orientation
        pos = (self.workspace_limits[0][0], self.workspace_limits[1][1], self.workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5) # Use original orientation
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        # Load tray pad
        tray_path = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_pos = np.array(self.POSE_TRAY[0]) * self.SCALING
        tray_orn_quat = p.getQuaternionFromEuler(self.POSE_TRAY[1])
        tray_id = p.loadURDF(tray_path, tray_pos, tray_orn_quat,
                             globalScaling=self.SCALING, useFixedBase=False)
        p.changeVisualShape(tray_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(tray_id)

        # Load needle
        needle_path = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        self.needle_id = p.loadURDF(needle_path, useFixedBase=False, globalScaling=self.SCALING)
        p.changeVisualShape(self.needle_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(self.needle_id)
        
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
        # Aligned with SurRoL's PsmEnv _set_action
        action = action.copy()
        delta_pos = action[:3] * 0.01 * self.SCALING # Scale down the movement
        
        current_pose_matrix = self.psm1.get_current_position()
        current_pose_matrix[:3, 3] += delta_pos
        
        # Clip to workspace
        current_pose_matrix[:3, 3] = np.clip(
            current_pose_matrix[:3, 3],
            self.workspace_limits[:, 0],
            self.workspace_limits[:, 1]
        )
        
        # For this task, we ignore orientation changes for simplicity
        self.psm1.move(current_pose_matrix)

    def _is_success(self, obs: dict) -> bool:
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return dist < self.success_threshold

    def _get_reward(self, obs: dict) -> float:
        # Sparse reward: -1 for failure, 0 for success
        return -1.0 if not self._is_success(obs) else 0.0

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
        Resets the needle to a random position within the workspace, aligned with SurRoL logic.
        """
        # Use the unscaled workspace limits for random generation
        unscaled_ws = self.WORKSPACE_LIMITS
        random_x = self.np_random.uniform(unscaled_ws[0][0], unscaled_ws[0][1])
        random_y = self.np_random.uniform(unscaled_ws[1][0], unscaled_ws[1][1])
        # Add offset before scaling
        z_pos = unscaled_ws[2][0] + 0.01
        
        # Scale the final position
        final_pos = np.array([random_x, random_y, z_pos]) * self.SCALING
        
        random_yaw = (self.np_random.uniform() - 0.5) * np.pi
        random_orn = p.getQuaternionFromEuler([0, 0, random_yaw])
        
        p.resetBasePositionAndOrientation(self.needle_id, final_pos, random_orn)
