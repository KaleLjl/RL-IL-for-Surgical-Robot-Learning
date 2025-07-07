import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_pose_details():
    """Debug pose calculation details"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== POSE CALCULATION DEBUG ===")
    
    psm = env.unwrapped.psm1
    
    # Get current pose from robot
    current_position_rcm = psm.get_current_position()  # 4x4 matrix in RCM frame
    pose_world_tuple = psm.pose_rcm2world(current_position_rcm, 'tuple')
    pose_world_matrix = psm.pose_rcm2world(current_position_rcm, 'matrix')
    
    print(f"Current position (RCM frame):")
    print(current_position_rcm)
    print()
    
    print(f"Pose world (tuple): {pose_world_tuple}")
    print(f"Pose world (matrix):")
    print(pose_world_matrix)
    print()
    
    # Direct link poses
    pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
    pos_eef, _ = get_link_pose(psm.body, psm.EEF_LINK_INDEX)
    
    print(f"Direct TIP position: {pos_tip}")
    print(f"Direct EEF position: {pos_eef}")
    print()
    
    # From observation
    obs_pos = obs['observation'][:3]
    print(f"Observation position: {obs_pos}")
    print()
    
    # Compare
    print("=== COMPARISONS ===")
    print(f"pose_rcm2world[0:3] vs TIP: {np.linalg.norm(np.array(pose_world_tuple[0]) - np.array(pos_tip)):.6f}")
    print(f"pose_rcm2world[0:3] vs EEF: {np.linalg.norm(np.array(pose_world_tuple[0]) - np.array(pos_eef)):.6f}")
    print(f"observation vs TIP: {np.linalg.norm(obs_pos - np.array(pos_tip)):.6f}")
    print(f"observation vs EEF: {np.linalg.norm(obs_pos - np.array(pos_eef)):.6f}")
    print(f"observation vs pose_rcm2world: {np.linalg.norm(obs_pos - np.array(pose_world_tuple[0])):.6f}")
    
    env.close()

if __name__ == "__main__":
    debug_pose_details()