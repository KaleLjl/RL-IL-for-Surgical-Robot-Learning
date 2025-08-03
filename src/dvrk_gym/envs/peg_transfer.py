# Author(s): Lele Chen
# Created on: 2025-07-07
# Last modified: 2025-07-07

"""
Peg Transfer Task Environment for the dVRK.
This task requires the robot to pick up blocks and transfer them to designated pegs.
Based on the original SurRoL implementation.
"""
import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from .dvrk_env import DVRKEnv
from ..robots import Psm
from ..utils.pybullet_utils import get_body_pose, get_link_pose, wrap_angle

# Define the asset path relative to this file
ASSET_DIR_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets')

class PegTransferEnv(DVRKEnv):
    """
    Gymnasium environment for the dVRK Peg Transfer task.
    Aligned with the original SurRoL implementation parameters.
    """
    # Constants from original SurRoL implementation
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.
    
    # Constants from the parent PsmEnv in SurRoL
    POSE_PSM1 = ((0.05, 0.24, 0.8524), (0, 0, -(90 + 20) / 180 * np.pi))
    DISTANCE_THRESHOLD = 0.005

    def __init__(self, render_mode: str = None, use_dense_reward: bool = False, curriculum_level: int = 4):
        # Store render_mode and reward setting
        self.render_mode = render_mode
        self.use_dense_reward = use_dense_reward
        self.curriculum_level = curriculum_level  # 1-4 for different curriculum stages
        
        # Initialize workspace limits (matching SurRoL PsmEnv parent class)
        workspace_limits = np.asarray(self.WORKSPACE_LIMITS) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))  # tip-eef offset with collision margin
        self.workspace_limits = workspace_limits * self.SCALING
        
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING
        
        super().__init__(render_mode=self.render_mode)
        
        self.success_threshold = self.distance_threshold
        
        # Task-specific attributes
        self.has_object = True
        self._activated = -1
        self._contact_constraint = None
        # Use distance-based activation for Level 3 (grasping level)
        self._contact_approx = True if curriculum_level == 3 else False
        self._waypoint_goal = True   # PegTransfer has waypoint goals, so should be True
        self._waypoints = None
        
        # Reward tracking to prevent exploitation
        self._approach_achieved = False
        self._position_achieved = False
        self._gripper_attempt_achieved = False
        self._contact_achieved = False
        self._grasp_achieved = False
        
        
        # Curriculum tracking
        self._grasp_stable_steps = 0     # For level 2

    def _env_setup(self):
        """
        Loads the robot, peg board, and blocks into the simulation, aligned with SurRoL parameters.
        """
        # Load robot - Pass unscaled position to constructor
        psm_pos = np.array(self.POSE_PSM1[0])
        psm_orn_eul = self.POSE_PSM1[1]
        psm_orn_quat = p.getQuaternionFromEuler(psm_orn_eul)
        self.psm1 = Psm(pos=psm_pos, orn=psm_orn_quat, scaling=self.SCALING)
        
        # Load peg board
        peg_board_path = os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf')
        peg_board_pos = np.array(self.POSE_BOARD[0]) * self.SCALING
        peg_board_orn = p.getQuaternionFromEuler(self.POSE_BOARD[1])
        self.peg_board_id = p.loadURDF(peg_board_path, peg_board_pos, peg_board_orn,
                                       globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(self.peg_board_id)
        
        # Setup peg indices - original SurRoL implementation
        self._pegs = np.arange(12)
        np.random.shuffle(self._pegs[:6])
        np.random.shuffle(self._pegs[6:12])
        
        # Load blocks
        self._load_blocks()
        
        # Initialize robot position (same as SurROL)
        workspace_limits = self.workspace_limits
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

    def _load_blocks(self):
        """
        Load blocks on the peg board, following the original SurRoL implementation.
        """
        num_blocks = 4
        self._blocks = []
        
        # Load blocks on the second set of pegs (6-11)
        for i in self._pegs[6:6 + num_blocks]:
            pos, orn = get_link_pose(self.peg_board_id, i)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            
            block_path = os.path.join(ASSET_DIR_PATH, 'block/block.urdf')
            # Add 0.03 above the peg position (same as SurRoL - no scaling for this offset)
            block_pos = np.array(pos) + np.array([0, 0, 0.03])
            block_orn = p.getQuaternionFromEuler((0, 0, yaw))
            
            block_id = p.loadURDF(block_path, block_pos, block_orn,
                                  useFixedBase=False, globalScaling=self.SCALING)
            self.obj_ids['rigid'].append(block_id)
            self._blocks.append(block_id)
        
        self._blocks = np.array(self._blocks)
        np.random.shuffle(self._blocks)
        
        # Change color of the first block to red (target block)
        for obj_id in self._blocks[:1]:
            p.changeVisualShape(obj_id, -1, rgbaColor=(255/255, 69/255, 58/255, 1))
        
        # Set the target object (following SurRoL implementation)
        self.obj_id = self._blocks[0]
        self.obj_link1 = 1  # This is used for waypoint calculations

    def _get_action_space(self) -> spaces.Space:
        # Action is a 5-DoF delta pose (dx, dy, dz, dyaw, gripper)
        return spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def _get_observation_space(self) -> spaces.Space:
        # Observation is a dictionary containing robot and task states
        obs_shape = len(self._get_obs_robot_state())
        return spaces.Dict({
            'observation': spaces.Box(-np.inf, np.inf, shape=(obs_shape,), dtype=np.float32),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        robot_state = self._get_obs_robot_state().astype(np.float32)
        
        # For Level 1, 2, and 3: Use TIP position as achieved_goal (positioning/grasping tasks)
        # For higher levels (4+): Use object position as achieved_goal (manipulation tasks)
        if self.curriculum_level <= 3:
            tip_pos, _ = get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)
            achieved_goal = np.array(tip_pos, dtype=np.float32)
        else:
            # Get object position (the red block) - this is what we want to move to the goal
            obj_pos, _ = get_body_pose(self.obj_id)
            achieved_goal = np.array(obj_pos, dtype=np.float32)
        
        return {
            'observation': robot_state,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy().astype(np.float32),
        }

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it, aligned with SurRoL waypoints.
        """
        obj_pos, _ = get_link_pose(self.obj_id, self.obj_link1)
        
        if self.curriculum_level == 1:
            # Level 1 = Waypoint 0: Safe approach above object
            above_height = obj_pos[2] + 0.045 * self.SCALING  # 4.5cm above object (waypoint 0)
            goal = np.array([obj_pos[0], obj_pos[1], above_height], dtype=np.float32)
            
        elif self.curriculum_level == 2:
            # Level 2 = Waypoint 1: Move to grasp position (approach from above)
            object_surface_height = obj_pos[2] + (0.003 + 0.0102) * self.SCALING
            target_tip_height = object_surface_height - 0.022 * self.SCALING
            grasp_height = target_tip_height + 0.051
            goal = np.array([obj_pos[0], obj_pos[1], grasp_height], dtype=np.float32)
            
        elif self.curriculum_level == 3:
            # Level 3 = Waypoint 2: Same as Level 2 but close gripper to grasp
            object_surface_height = obj_pos[2] + (0.003 + 0.0102) * self.SCALING
            target_tip_height = object_surface_height - 0.022 * self.SCALING
            grasp_height = target_tip_height + 0.051
            goal = np.array([obj_pos[0], obj_pos[1], grasp_height], dtype=np.float32)
            
        elif self.curriculum_level == 4:
            # Level 4 = Waypoint 3: Lift to above object position
            above_height = obj_pos[2] + 0.045 * self.SCALING  # 4.5cm above object
            goal = np.array([obj_pos[0], obj_pos[1], above_height], dtype=np.float32)
            
        elif self.curriculum_level == 5:
            # Level 5 = Waypoint 4: Transport to above goal position
            goal_pos, _ = get_link_pose(self.peg_board_id, self._pegs[0])
            pos_peg = get_link_pose(self.peg_board_id, self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]
            pos_place = [goal_pos[0] + obj_pos[0] - pos_peg[0],
                        goal_pos[1] + obj_pos[1] - pos_peg[1]]
            above_height = obj_pos[2] + 0.045 * self.SCALING  # Same height as waypoint 0/3
            goal = np.array([pos_place[0], pos_place[1], above_height], dtype=np.float32)
            
        elif self.curriculum_level == 6:
            # Level 6 = Waypoint 5: Lower to release height above goal
            goal_pos, _ = get_link_pose(self.peg_board_id, self._pegs[0])
            pos_peg = get_link_pose(self.peg_board_id, self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]
            pos_place = [goal_pos[0] + obj_pos[0] - pos_peg[0],
                        goal_pos[1] + obj_pos[1] - pos_peg[1]]
            release_height = goal_pos[2] + 0.021 * self.SCALING  # 2.1cm above goal
            goal = np.array([pos_place[0], pos_place[1], release_height], dtype=np.float32)
            
        elif self.curriculum_level == 7:
            # Level 7 = Waypoint 6: Same position as Level 6, but release object
            goal_pos, _ = get_link_pose(self.peg_board_id, self._pegs[0])
            pos_peg = get_link_pose(self.peg_board_id, self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]
            pos_place = [goal_pos[0] + obj_pos[0] - pos_peg[0],
                        goal_pos[1] + obj_pos[1] - pos_peg[1]]
            release_height = goal_pos[2] + 0.021 * self.SCALING  # 2.1cm above goal
            goal = np.array([pos_place[0], pos_place[1], release_height], dtype=np.float32)
            
        else:
            # Fallback: Full task - destination peg position
            goal_pos, _ = get_link_pose(self.peg_board_id, self._pegs[0])
            goal = np.array(goal_pos, dtype=np.float32)
            
        return goal.copy()
    
    def reset(self, seed=None, options=None):
        """Reset environment and clear reward tracking flags."""
        # Reset reward tracking flags
        self._approach_achieved = False
        self._position_achieved = False
        self._gripper_attempt_achieved = False
        self._contact_achieved = False
        self._grasp_achieved = False
        
        # Reset curriculum tracking
        self._grasp_stable_steps = 0
        
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Store initial object height for level tracking if needed
        if self.curriculum_level >= 2:
            obj_pos, _ = get_body_pose(self.obj_id)
            self._initial_obj_height = obj_pos[2]
        
        return obs, info

    def _sample_goal_callback(self):
        """ Moves the goal visualization sphere to the new goal position.
        """
        # Move goal visualization
        p.resetBasePositionAndOrientation(self.goal_vis_id, self.goal, (0, 0, 0, 1))
        
        # Define waypoints after goal is sampled (but don't execute them yet)
        # Waypoints will be defined when first oracle action is called
        self._waypoints = None

    def _define_waypoints(self):
        """ Define waypoints for the oracle policy, following SurRoL implementation.
        """
        self._waypoints = [None, None, None, None, None, None, None]  # seven waypoints
        
        # Get object position and orientation
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        
        # Get end-effector orientation
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        
        # Calculate optimal yaw to minimize rotation
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
              else wrap_angle(orn[2] + np.pi)
        
        # SurROL logic with TIP-EEF offset compensation for physical contact
        above_height = pos_obj[2] + 0.045 * self.SCALING  # High above object
        
        # Calculate grasp height to ensure TIP reaches near object surface for physical contact
        # Target: TIP should reach object surface (or slightly below) for physical contact detection
        object_surface_height = pos_obj[2] + (0.003 + 0.0102) * self.SCALING  # Object top surface
        target_tip_height = object_surface_height - 0.022 * self.SCALING  # 2.2cm below surface for contact
        grasp_height = target_tip_height + 0.051  # Add TIP-EEF offset to get EEF target
        
        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1], above_height, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1], grasp_height, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1], grasp_height, yaw, -0.5])  # grasp (close gripper)
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1], above_height, yaw, -0.5])  # lift up
        
        # Calculate place position considering offset (following SurRoL)
        pos_peg = get_link_pose(self.peg_board_id, self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]
        
        # Calculate lower release height for more accurate placement
        release_height = self.goal[2] + 0.021 * self.SCALING  # 2.1cm above goal position
        
        self._waypoints[4] = np.array([pos_place[0], pos_place[1], above_height, yaw, -0.5])  # above goal, still grasping
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], release_height, yaw, -0.5])  # lower to release height, still grasping
        
        # 6th waypoint for actual release
        self._waypoints[6] = np.array([pos_place[0], pos_place[1], release_height, yaw, 0.5])  # release at lower height

    def get_oracle_action(self, obs: dict) -> np.ndarray:
        """
        Define a human expert strategy following SurRoL implementation.
        """
        # Ensure waypoints are defined
        if self._waypoints is None:
            self._define_waypoints()
        
        # seven waypoints executed in sequential order
        action = np.zeros(5)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            
            # Get current EEF position and yaw from observation
            # obs['observation'] contains: [eef_pos(3), eef_euler(3), jaw_angle(1), obj_pos(3), obj_rel_pos(3)]
            current_pos = obs['observation'][:3]  # EEF position
            current_yaw = obs['observation'][5]   # EEF yaw angle
            
            delta_pos = (waypoint[:3] - current_pos) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - current_yaw).clip(-0.4, 0.4)
            
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            
            scale_factor = 0.7
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            
            # Use same thresholds as original SurRoL (delta_pos is already scaled)
            pos_threshold = 2e-3  # Same as original SurRoL
            yaw_threshold = np.deg2rad(2.)  # Same as original SurRoL
            
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < pos_threshold and np.abs(delta_yaw) < yaw_threshold:
                self._waypoints[i] = None
            break
        
        return action

    def _set_action(self, action: np.ndarray):
        """
        Applies the given action to the simulation.
        The action is a delta pose in the world frame with gripper control.
        """
        action = action.copy()
        action[:3] *= 0.01 * self.SCALING  # position scaling
        
        # Get current end-effector pose
        pose_world = self.psm1.pose_rcm2world(self.psm1.get_current_position())
        
        # Apply position delta with workspace limits (more generous height limits)
        pose_world[:3, 3] = np.clip(pose_world[:3, 3] + action[:3],
                                    self.workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                    self.workspace_limits[:, 1] + [0.02, 0.02, 0.15])
        
        # Apply yaw rotation
        from ..utils.robotics import get_euler_from_matrix, get_matrix_from_euler
        rot = get_euler_from_matrix(pose_world[:3, :3])
        psm1_eul = np.array([np.deg2rad(-90), 0., rot[2]])  # Fixed roll and pitch
        
        action[3] *= np.deg2rad(30)  # yaw scaling
        new_yaw = wrap_angle(rot[2] + action[3])
        rot = (psm1_eul[0], psm1_eul[1], new_yaw)
        
        pose_world[:3, :3] = get_matrix_from_euler(rot)
        action_rcm = self.psm1.pose_world2rcm(pose_world)
        
        # Move the robot
        self.psm1.move(action_rcm)
        
        
        # Handle gripper (no blocking - let agent control freely)
        if action[4] < 0:
            self.psm1.close_jaw()
            # Use physical contact detection (original SurROL approach)
            self._activate(0)  # Only activates if physical contact condition is met
        else:
            self.psm1.move_jaw(np.deg2rad(40))
            self._release(0)
        
        # Process contact constraints after action (crucial step!)
        self._step_callback()

    def _activate(self, idx: int):
        """
        Activation logic for grasping following SurRoL implementation.
        """
        
        if self._activated < 0:
            # Only activate one psm
            psm = self.psm1
            if self._contact_approx:
                # activate if the distance between the object and the tip below a threshold
                pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
                if not self._waypoint_goal:
                    link_id = -1
                else:
                    link_id = self.obj_link1 if idx == 0 else self.obj_link2  # TODO: check
                pos_obj, _ = get_link_pose(self.obj_id, link_id)
                if np.linalg.norm(np.array(pos_tip) - np.array(pos_obj)) < 2e-3 * self.SCALING:
                    self._activated = idx
                    # disable collision
                    p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                             linkIndexA=6, linkIndexB=-1, enableCollision=0)
                    p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                             linkIndexA=7, linkIndexB=-1, enableCollision=0)
            else:
                # Improved physical contact detection - more forgiving than strict dual-finger contact
                points_1 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
                points_2 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
                points_1 = [point for point in points_1 if point[2] == self.obj_id]
                points_2 = [point for point in points_2 if point[2] == self.obj_id]
                
                has_contact_1 = len(points_1) > 0
                has_contact_2 = len(points_2) > 0
                
                # Calculate distances from finger tips to object
                pos_tip_1, _ = get_link_pose(psm.body, 6)  # Left finger
                pos_tip_2, _ = get_link_pose(psm.body, 7)  # Right finger
                pos_obj, _ = get_link_pose(self.obj_id, self.obj_link1 if idx == 0 else self.obj_link2)
                
                dist_1 = np.linalg.norm(np.array(pos_tip_1) - np.array(pos_obj))
                dist_2 = np.linalg.norm(np.array(pos_tip_2) - np.array(pos_obj))
                
                # More realistic activation conditions:
                # 1. At least one finger has contact AND the other is close enough
                # 2. Or both fingers are very close to the object
                close_threshold = 3e-3 * self.SCALING  # 1.5cm threshold
                very_close_threshold = 1.5e-3 * self.SCALING  # 0.75cm threshold
                
                if ((has_contact_1 and dist_2 < close_threshold) or 
                    (has_contact_2 and dist_1 < close_threshold) or
                    (dist_1 < very_close_threshold and dist_2 < very_close_threshold)):
                    self._activated = idx

    def _create_contact_constraint(self):
        """
        Create a contact constraint to make grasping stable.
        """
        if self._contact_constraint is not None:
            return
        
        psm = self.psm1
        body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
        obj_pose = p.getBasePositionAndOrientation(self.obj_id)
        
        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1],
                                           obj_pose[0], obj_pose[1])
        
        self._contact_constraint = p.createConstraint(
            parentBodyUniqueId=psm.body,
            parentLinkIndex=psm.EEF_LINK_INDEX,
            childBodyUniqueId=self.obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=obj_to_body[0],
            parentFrameOrientation=obj_to_body[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0))
        
        p.changeConstraint(self._contact_constraint, maxForce=50)

    def _release(self, idx: int):
        """
        Release the grasped object.
        """
        if self._activated != idx:
            return
        
        self._activated = -1
        
        if self._contact_constraint is not None:
            try:
                p.removeConstraint(self._contact_constraint)
                self._contact_constraint = None
                
                # Re-enable collision detection
                psm = self.psm1
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                         linkIndexA=6, linkIndexB=-1, enableCollision=1)
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                         linkIndexA=7, linkIndexB=-1, enableCollision=1)
            except:
                pass

    def _step_callback(self):
        """
        Process contact constraints after each action step.
        This is crucial for proper grasping behavior.
        """
        if not self.has_object or self._activated < 0:
            return
        elif self._contact_constraint is None:
            # The gripper is activated; check if we can create a constraint
            psm = self.psm1
            if self._meet_contact_constraint_requirement():
                body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
                obj_pose = p.getBasePositionAndOrientation(self.obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1],
                                                   obj_pose[0], obj_pose[1])

                self._contact_constraint = p.createConstraint(
                    parentBodyUniqueId=psm.body,
                    parentLinkIndex=psm.EEF_LINK_INDEX,
                    childBodyUniqueId=self.obj_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))
                
                p.changeConstraint(self._contact_constraint, maxForce=50)
                
                # Disable collision to prevent interference
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                         linkIndexA=6, linkIndexB=-1, enableCollision=0)
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                         linkIndexA=7, linkIndexB=-1, enableCollision=0)
        else:
            # Constraint exists; check if contact is maintained
            psm = self.psm1
            points = p.getContactPoints(bodyA=psm.body, linkIndexA=6) + \
                     p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            points = [point for point in points if point[2] == self.obj_id]
            remain_contact = len(points) > 0

            if not remain_contact and not self._contact_approx:
                # Release the object if no contact
                self._release(self._activated)

    def _meet_contact_constraint_requirement(self) -> bool:
        """
        Check if the contact constraint requirement is met.
        Relaxed constraint creation: create constraint when object is lifted above goal height + 0.5cm.
        This makes grasping more forgiving while still preventing grab-drop cycles.
        """
        if not self.has_object:
            return False
        
        # Relaxed constraint creation: only need 0.5cm lift instead of 1cm
        # This makes BC training more successful while maintaining physical realism
        from ..utils.pybullet_utils import get_body_pose
        obj_pose = get_body_pose(self.obj_id)
        return obj_pose[0][2] > self.goal[2] + 0.005 * self.SCALING

    def _is_success(self, obs: dict) -> bool:
        """
        Check if the task is successfully completed based on curriculum level.
        """
        if self.curriculum_level == 1:
            return self._is_level_1_success(obs)
        elif self.curriculum_level == 2:
            return self._is_level_2_success(obs)
        elif self.curriculum_level == 3:
            return self._is_level_3_success(obs)
        elif self.curriculum_level == 4:
            return self._is_level_4_success(obs)
        elif self.curriculum_level == 5:
            return self._is_level_5_success(obs)
        elif self.curriculum_level == 6:
            return self._is_level_6_success(obs)
        elif self.curriculum_level == 7:
            return self._is_level_7_success(obs)
        else:  # Fallback: full task
            return self._is_level_4_success(obs)
    
    def _is_level_1_success(self, obs: dict) -> bool:
        """Level 1: Gripper body must attach precisely to target with open gripper."""
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        is_gripper_open = jaw_angle > 0.3
        
        # Success requires BOTH conditions:
        # 1. Precise attachment (normal threshold)
        # 2. Gripper must be open
        return distance < self.success_threshold and is_gripper_open
    
    def _is_level_2_success(self, obs: dict) -> bool:
        """Level 2 = Waypoint 1: Reach grasp position with open gripper."""
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        is_gripper_open = jaw_angle > 0.3
        
        # Success requires BOTH conditions:
        # 1. Precise positioning at grasp height
        # 2. Gripper must be open
        return distance < self.success_threshold and is_gripper_open
    
    def _is_level_3_success(self, obs: dict) -> bool:
        """Level 3 = Waypoint 2: Close gripper and grasp object."""
        # Simplified success: just need to be in position and close gripper
        # This allows the agent to learn gripper control first
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        # Use more generous threshold for Level 3 to help learning
        level_3_threshold = self.success_threshold * 2.5  # 0.0625 instead of 0.025
        return distance < level_3_threshold and jaw_angle <= 0.0
    
    def _is_level_4_success(self, obs: dict) -> bool:
        """Level 4 = Waypoint 3: Lift grasped object to above position."""
        # Must be grasped and at lift position
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        if not is_grasped:
            return False
        
        # Check if EEF reached the lift goal
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return distance < self.success_threshold
    
    def _is_level_5_success(self, obs: dict) -> bool:
        """Level 5 = Waypoint 4: Transport to above goal position while grasped."""
        # Must be grasped and at transport position
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        if not is_grasped:
            return False
        
        # Check if EEF reached the transport goal
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return distance < self.success_threshold
    
    def _is_level_6_success(self, obs: dict) -> bool:
        """Level 6 = Waypoint 5: Lower to release height while grasped."""
        # Must be grasped and at release height
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        if not is_grasped:
            return False
        
        # Check if EEF reached the release height
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        return distance < self.success_threshold
    
    def _is_level_7_success(self, obs: dict) -> bool:
        """Level 7 = Waypoint 6: Release object at correct position."""
        # Success when object is placed correctly (original task completion)
        goal_distance_2d = np.linalg.norm(obs['achieved_goal'][:2] - obs['desired_goal'][:2])
        height_diff = np.abs(obs['achieved_goal'][2] - obs['desired_goal'][2])
        
        return (goal_distance_2d < 5e-3 * self.SCALING and
                height_diff < 4e-3 * self.SCALING)

    def _get_reward(self, obs: dict) -> float:
        """
        Calculates the reward based on the current observation.
        Level 1 uses dense rewards for better learning, others use sparse rewards.
        """
        if self.curriculum_level == 1:
            # Level 1 always uses dense rewards for better learning
            return self._get_level_1_dense_reward(obs)
        elif self.curriculum_level == 2:
            # Level 2 uses dense rewards to encourage grasping
            return self._get_level_2_dense_reward(obs)
        elif self.curriculum_level == 3:
            # Level 3 uses dense rewards to encourage grasping
            return self._get_level_3_dense_reward(obs)
        elif self.curriculum_level == 4:
            # Level 4 uses dense rewards to encourage lifting
            return self._get_level_4_dense_reward(obs)
        elif self.curriculum_level == 5:
            # Level 5 uses dense rewards to encourage transport
            return self._get_level_5_dense_reward(obs)
        elif self.curriculum_level == 6:
            # Level 6 uses dense rewards to encourage lowering
            return self._get_level_6_dense_reward(obs)
        elif self.curriculum_level == 7:
            # Level 7 uses dense rewards to encourage release
            return self._get_level_7_dense_reward(obs)
        elif not self.use_dense_reward:
            # Fallback: sparse rewards
            return self._get_sparse_reward(obs)
        else:
            # Level 4 can use dense rewards if enabled
            return self._get_dense_reward(obs)

    def _get_sparse_reward(self, obs: dict) -> float:
        """
        Sparse reward: -1 for failure, 0 for success.
        """
        return -1.0 if not self._is_success(obs) else 0.0

    def _get_dense_reward(self, obs: dict) -> float:
        """
        One-time intermediate states reward system.
        Each sub-gesture reward is given only once to prevent exploitation.
        """
        if self._is_success(obs):
            return 20.0
        
        reward = 0.0
        
        # Get relevant states
        eef_pos = obs['observation'][:3]
        obj_pos = obs['achieved_goal']
        jaw_angle = obs['observation'][6]
        dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
        
        # Sub-gesture 1: Approach object (one-time 1.0)
        if dist_to_obj < 0.05 * self.SCALING and not self._approach_achieved:
            reward += 1.0
            self._approach_achieved = True
        
        # Sub-gesture 2: Close positioning (one-time 1.0)
        if dist_to_obj < 0.02 * self.SCALING and not self._position_achieved:
            reward += 1.0
            self._position_achieved = True
        
        # Sub-gesture 3: Gripper closing attempt when close (one-time 2.0)
        if dist_to_obj < 0.02 * self.SCALING and jaw_angle < -0.5 and not self._gripper_attempt_achieved:
            reward += 2.0
            self._gripper_attempt_achieved = True
        
        # Sub-gesture 4: Contact detected/activated (one-time 3.0)
        if self._activated >= 0 and not self._contact_achieved:
            reward += 3.0
            self._contact_achieved = True
        
        # Sub-gesture 5: Full grasp achieved (one-time 5.0)
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        if is_grasped and not self._grasp_achieved:
            reward += 5.0
            self._grasp_achieved = True
            
        # Sub-gesture 6: Transport progress (continuous but only if grasped)
        if is_grasped:
            initial_dist = 0.2 * self.SCALING
            current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            progress = max(0, (initial_dist - current_dist) / initial_dist)
            reward += 5.0 * progress * 0.1  # Scaled down to 0.5 max per step
        
        return reward

    def _get_level_1_dense_reward(self, obs: dict) -> float:
        """
        NeedleReach-style reward for Level 1: Negative distance + gripper state penalty.
        Simple and clean like NeedleReach but with gripper considerations.
        """
        # Base reward: negative distance to goal (same as NeedleReach)
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        reward = -distance
        
        # Gripper state penalty (only addition to NeedleReach style)
        jaw_angle = obs['observation'][6]
        is_gripper_open = jaw_angle > 0.3  # Open if > ~17 degrees
        
        if not is_gripper_open:
            reward -= 1.0  # Penalty for closed gripper (keep it simple)
        
        return reward

    def _get_level_2_dense_reward(self, obs: dict) -> float:
        """
        Level 2 = Waypoint 1: Approach to grasp position with open gripper.
        Simple reach-style reward: negative distance to goal.
        """
        # Simple negative distance reward (like reach environment)
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        reward = -distance
        
        # Small penalty for closing gripper (should stay open)
        jaw_angle = obs['observation'][6]
        if jaw_angle < 0.3:  # Gripper closing
            reward -= 1.0
        
        return reward

    def _get_level_3_dense_reward(self, obs: dict) -> float:
        """
        Level 3 = Waypoint 2: Close gripper to grasp object.
        Simplified reward to prevent reward hacking.
        """
        # Check if Level 3 success achieved
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        level_3_threshold = self.success_threshold * 2.5  # Match success condition
        
        if distance < level_3_threshold and jaw_angle <= 0.0:
            return 10.0  # Success reward
        
        # Not grasped yet - position and gripper control
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        
        # Base reward: negative distance (reach position first)
        reward = -distance
        
        # At grasp position? Encourage gripper closing
        if distance < level_3_threshold:
            if jaw_angle > 0.3:  # Gripper still open
                reward -= 1.0  # Penalty for not closing
            elif jaw_angle < -0.3:  # Gripper closing/closed
                reward += 2.0  # Bonus for closing action
        
        return reward

    def _get_level_4_dense_reward(self, obs: dict) -> float:
        """Level 4 = Waypoint 3: Lift grasped object to above position."""
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        
        if not is_grasped:
            return -10.0  # Heavy penalty for dropping object
        
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        
        if distance < self.success_threshold:
            return 10.0  # Success! Lifted to target height
        
        # Encourage lifting while maintaining grasp
        reward = -distance
        if jaw_angle < -0.3:
            reward += 0.2  # Maintain closed gripper
        else:
            reward -= 0.5  # Penalty for opening gripper
        
        return reward

    def _get_level_5_dense_reward(self, obs: dict) -> float:
        """Level 5 = Waypoint 4: Transport to above goal position."""
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        
        if not is_grasped:
            return -10.0  # Heavy penalty for dropping object
        
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        
        if distance < self.success_threshold:
            return 10.0  # Success! Transported to goal area
        
        # Encourage horizontal transport while maintaining grasp
        reward = -distance
        if jaw_angle < -0.3:
            reward += 0.2  # Maintain closed gripper
        else:
            reward -= 0.5  # Penalty for opening gripper
        
        return reward

    def _get_level_6_dense_reward(self, obs: dict) -> float:
        """Level 6 = Waypoint 5: Lower to release height."""
        is_grasped = self._activated >= 0 and self._contact_constraint is not None
        
        if not is_grasped:
            return -10.0  # Heavy penalty for dropping object
        
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        
        if distance < self.success_threshold:
            return 10.0  # Success! At release height
        
        # Encourage lowering while maintaining grasp
        reward = -distance
        if jaw_angle < -0.3:
            reward += 0.2  # Maintain closed gripper
        else:
            reward -= 0.5  # Penalty for opening gripper prematurely
        
        return reward

    def _get_level_7_dense_reward(self, obs: dict) -> float:
        """Level 7 = Waypoint 6: Release object at correct position."""
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        jaw_angle = obs['observation'][6]
        
        # Check if object is placed correctly (task completion)
        goal_distance_2d = np.linalg.norm(obs['achieved_goal'][:2] - obs['desired_goal'][:2])
        height_diff = np.abs(obs['achieved_goal'][2] - obs['desired_goal'][2])
        
        if (goal_distance_2d < 5e-3 * self.SCALING and height_diff < 4e-3 * self.SCALING):
            return 10.0  # Success! Task completed
        
        # At release position, encourage opening gripper
        reward = -1.0  # Base time penalty
        
        if distance < self.success_threshold:  # At correct EEF position
            if jaw_angle > 0.3:  # Gripper open
                reward += 2.0  # Good! Released at correct position
            elif jaw_angle < -0.3:  # Still grasping
                reward -= 0.5  # Should release now
        else:
            reward -= distance  # Get to release position first
        
        return reward

    def _get_obs_robot_state(self):
        """
        Returns the current state of the robot.
        SurROL uses EEF position (from pose_rcm2world) as the first 3 elements of robot state.
        """
        # Get robot pose in world frame (same as SurROL)
        pose_world = self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')
        jaw_angle = self.psm1.get_current_jaw_position()
        
        # Get object state
        obj_pos, _ = get_body_pose(self.obj_id)
        eef_pos = np.array(pose_world[0])  # EEF position (same as SurROL)
        object_rel_pos = np.array(obj_pos) - eef_pos
        
        return np.concatenate([
            eef_pos,  # 3 elements - EEF position (same as SurROL)
            np.array(p.getEulerFromQuaternion(pose_world[1])),  # 3 elements
            np.array(jaw_angle).ravel(),  # 1 element
            np.array(obj_pos),  # 3 elements
            object_rel_pos,  # 3 elements
        ]).astype(np.float32)  # Total: 13 elements