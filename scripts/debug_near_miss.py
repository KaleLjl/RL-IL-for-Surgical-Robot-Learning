import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_near_miss():
    """Debug the exact distance when gripper is closest to object"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== NEAR MISS ANALYSIS ===")
    
    min_distance = float('inf')
    closest_step = -1
    closest_positions = {}
    
    for step in range(80):
        action = env.unwrapped.get_oracle_action(obs)
        
        # Get current positions
        tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
        obj_pos, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
        eef_pos = obs['observation'][:3]
        
        # Calculate distance
        tip_to_obj = np.linalg.norm(np.array(tip_pos) - np.array(obj_pos))
        
        # Track minimum distance
        if tip_to_obj < min_distance:
            min_distance = tip_to_obj
            closest_step = step
            closest_positions = {
                'tip': tip_pos,
                'obj': obj_pos,
                'eef': eef_pos,
                'action': action.copy(),
                'gripper': action[4]
            }
        
        # Print when grasping starts
        if action[4] < 0 and step % 5 == 0:
            print(f"Step {step:2d}: TIP-obj={tip_to_obj:.4f}, gripper={action[4]:.2f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"\n=== CLOSEST APPROACH ===")
    print(f"Closest at step {closest_step}: {min_distance:.6f} meters ({min_distance*100:.2f} cm)")
    print(f"TIP position: {closest_positions['tip']}")
    print(f"Object position: {closest_positions['obj']}")
    print(f"EEF position: {closest_positions['eef']}")
    print(f"Gripper action: {closest_positions['gripper']:.2f}")
    
    # Analyze the difference vector
    diff = np.array(closest_positions['obj']) - np.array(closest_positions['tip'])
    print(f"\nDifference vector (obj - tip): {diff}")
    print(f"X offset: {diff[0]:.4f} ({diff[0]*100:.1f} cm)")
    print(f"Y offset: {diff[1]:.4f} ({diff[1]*100:.1f} cm)")  
    print(f"Z offset: {diff[2]:.4f} ({diff[2]*100:.1f} cm)")
    
    # Check what would happen if we moved slightly
    needed_action = diff / (0.01 * env.unwrapped.SCALING)  # Convert to action units
    print(f"\nAction needed to reach object: {needed_action[:3]}")
    print(f"Current action was: {closest_positions['action'][:3]}")
    
    # Check activation threshold
    threshold = 2e-3 * env.unwrapped.SCALING
    print(f"\nActivation threshold: {threshold:.6f} ({threshold*100:.2f} cm)")
    print(f"Distance ratio: {min_distance/threshold:.2f}x threshold")
    
    env.close()

if __name__ == "__main__":
    debug_near_miss()