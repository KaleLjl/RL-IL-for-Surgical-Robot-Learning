import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_surrol_pose():
    """Debug what pose_rcm2world actually returns in our implementation"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== SurROL POSE ANALYSIS ===")
    
    psm = env.unwrapped.psm1
    
    # Test different ways to get position
    current_position_rcm = psm.get_current_position()  # 4x4 matrix
    pose_world_tuple = psm.pose_rcm2world(current_position_rcm, 'tuple')
    
    # Direct link positions
    tip_pos, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
    eef_pos, _ = get_link_pose(psm.body, psm.EEF_LINK_INDEX)
    
    print(f"pose_rcm2world result: {pose_world_tuple[0]}")
    print(f"Direct TIP position: {tip_pos}")
    print(f"Direct EEF position: {eef_pos}")
    print()
    
    # Compare distances
    rcm_to_tip = np.linalg.norm(np.array(pose_world_tuple[0]) - np.array(tip_pos))
    rcm_to_eef = np.linalg.norm(np.array(pose_world_tuple[0]) - np.array(eef_pos))
    
    print(f"pose_rcm2world to TIP distance: {rcm_to_tip:.6f}")
    print(f"pose_rcm2world to EEF distance: {rcm_to_eef:.6f}")
    print()
    
    # Check the robot's DH parameters and tool_T_tip
    print(f"tool_T_tip matrix:")
    print(psm.tool_T_tip)
    print()
    
    # Check if pose_rcm2world actually returns TIP position by design
    print("=== CHECKING RCM TO WORLD TRANSFORMATION ===")
    print(f"EEF link index: {psm.EEF_LINK_INDEX}")
    print(f"TIP link index: {psm.TIP_LINK_INDEX}")
    
    # Check what get_current_position returns
    print(f"get_current_position() matrix:")
    print(current_position_rcm)
    print()
    
    env.close()

if __name__ == "__main__":
    debug_surrol_pose()