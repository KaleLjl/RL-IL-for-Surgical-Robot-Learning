# Author(s): Lele Chen
# Created on: 2025-06-24
# Last modified: 2025-06-24

"""
Needle Reach Task Environment for the dVRK.
This task requires the robot to move its end-effector to a target needle.
"""
import os
from collections import deque
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
    POSE_PSM1 = ((0.05, 0.24, 0.8024), (0, 0, -(90 + 20) / 180 * np.pi))
    DISTANCE_THRESHOLD = 0.005

    def __init__(self, render_mode: str = None, use_dense_reward: bool = False):
        # Store render_mode and reward setting
        self.render_mode = render_mode
        self.use_dense_reward = use_dense_reward
        
        # Correctly initialize workspace limits with offset before scaling
        workspace_limits = np.asarray(self.WORKSPACE_LIMITS) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))
        self.workspace_limits = workspace_limits * self.SCALING
        
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING
        
        super().__init__(render_mode=self.render_mode)
        
        self.success_threshold = self.distance_threshold
        
        # Success criterion: require K of last M steps inside threshold (with hysteresis)
        # Defaults: K=3, M=5
        self.success_window_size = 5
        self.success_required = 3
        self._success_window = deque(maxlen=self.success_window_size)
        self.success_enter_threshold = self.distance_threshold
        self.success_exit_threshold = float(self.distance_threshold) * 1.5

    def _env_setup(self):
        """
        Loads the robot, tray, and needle into the simulation, aligned with SurRoL parameters.
        """
        # Load robot - Pass unscaled position to constructor
        psm_pos = np.array(self.POSE_PSM1[0])
        psm_orn_eul = self.POSE_PSM1[1]
        psm_orn_quat = p.getQuaternionFromEuler(psm_orn_eul)
        self.psm1 = Psm(pos=psm_pos, orn=psm_orn_quat, scaling=self.SCALING)
        
        # The robot is initialized to its default pose in the Psm constructor.
        # No explicit reset is needed here, aligning with a more stable setup.
        self.block_gripper = True

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

    def reset(self, seed=None, options=None):
        # Reset success window at the beginning of each episode
        self._success_window.clear()
        return super().reset(seed=seed, options=options)

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
        
        return {
            'observation': robot_state,
            'achieved_goal': np.array(eef_pos, dtype=np.float32),
            'desired_goal': self.goal.copy().astype(np.float32),
        }

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it, aligned with SurRoL.
        """
        # The goal is the position of the needle's center.
        # SurRoL adds a small z-offset.
        pos, _ = get_body_pose(self.needle_id)
        goal = np.array([pos[0], pos[1], pos[2] + 0.005 * self.SCALING], dtype=np.float32)
        return goal.copy()

    def _sample_goal_callback(self):
        """ Moves the goal visualization sphere to the new goal position.
        """
        p.resetBasePositionAndOrientation(self.goal_vis_id, self.goal, (0, 0, 0, 1))

    def get_oracle_action(self, obs: dict) -> np.ndarray:
        """
        Define a human expert strategy (P-controller).
        Aligned with the original SurRoL implementation.
        """
        # The P-controller logic from SurRoL is simplified here.
        # The original implementation had a division by 0.01, which is likely
        # a remnant of a different scaling system. We directly use the delta.
        delta_pos = obs['desired_goal'] - obs['achieved_goal']

        # Stop if close enough - adjusted threshold for direct distance
        # A large multiplier is used to generate a strong signal towards the goal.
        # This is then clipped to a max of 1.0 to ensure controlled movement.
        delta_pos *= 10  # Amplification factor
        
        # Clip the action to be within the standard [-1, 1] range
        delta_pos = np.clip(delta_pos, -1.0, 1.0)

        # Construct the final action array (dx, dy, dz, d_roll, d_pitch, d_yaw)
        # Set rotational components to zero as this is a reaching task.
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0., 0.])
        return action

    def _set_action(self, action: np.ndarray):
        """
        Applies the given action to the simulation.
        The action is a delta pose in the world frame.
        This method is aligned with the logic in SurRoL's psm_env.py.
        """
        action = action.copy()
        delta_pos = action[:3] * 0.01 * self.SCALING  # Scale down the movement

        # Get current end-effector pose in the world frame
        current_eef_pos, current_eef_orn = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)
        
        # Calculate the new target position in the world frame
        new_pos = current_eef_pos + delta_pos
        
        # Clip the new position to the workspace limits
        new_pos = np.clip(
            new_pos,
            self.workspace_limits[:, 0],
            self.workspace_limits[:, 1]
        )
        
        # The orientation is kept constant for this reaching task
        target_orn = current_eef_orn
        
        # Convert the world-frame target pose to the RCM frame for the robot controller
        target_pose_rcm = self.psm1.pose_world2rcm((new_pos, target_orn), option='matrix')

        # Move the robot
        self.psm1.move(target_pose_rcm)

    def _is_success(self, obs: dict) -> bool:
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        # Hysteresis logic for inside/outside decision
        prev = self._success_window[-1] if len(self._success_window) > 0 else False
        if dist <= self.success_enter_threshold:
            inside = True
        elif dist >= self.success_exit_threshold:
            inside = False
        else:
            inside = prev
        # Update window and evaluate K-of-M
        self._success_window.append(inside)
        return sum(1 for v in self._success_window if v) >= self.success_required

    def _get_reward(self, obs: dict) -> float:
        """
        Calculates the reward based on the current observation.
        This method routes to the appropriate reward function based on the
        `use_dense_reward` flag set during initialization.
        """
        if self.use_dense_reward:
            return self._get_dense_reward(obs)
        else:
            return self._get_sparse_reward(obs)

    def _get_sparse_reward(self, obs: dict) -> float:
        """
        Sparse reward: -1 for failure, 0 for success.
        Ideal for DAPG and other imitation learning setups.
        """
        return -1.0 if not self._is_success(obs) else 0.0

    def _get_dense_reward(self, obs: dict) -> float:
        """
        Dense reward: Negative distance to the goal.
        This encourages the agent to move closer to the goal at every step,
        which is crucial for pure RL algorithms like PPO.
        """
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        # The reward is the negative distance. Closer is better (less negative).
        return -dist

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
