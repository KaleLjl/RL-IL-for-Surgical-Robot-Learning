import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_precision():
    """Debug why there's still precision gap"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== PRECISION ANALYSIS ===")
    
    # Define waypoints manually
    env.unwrapped._define_waypoints()
    waypoint_2 = env.unwrapped._waypoints[2]  # Grasp waypoint
    
    # Get object and TIP positions
    obj_pos, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
    tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
    eef_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.EEF_LINK_INDEX)
    
    print(f"Object position (link1): {obj_pos}")
    print(f"Initial TIP position: {tip_pos}")
    print(f"Initial EEF position: {eef_pos}")
    print(f"Waypoint 2 (grasp): {waypoint_2}")
    print()
    
    # Calculate height differences
    waypoint_height = waypoint_2[2]
    obj_height = obj_pos[2]
    height_diff = waypoint_height - obj_height
    
    print(f"Waypoint grasp height: {waypoint_height:.4f}")
    print(f"Object height: {obj_height:.4f}")
    print(f"Height difference: {height_diff:.4f} ({height_diff*100:.1f}cm)")
    print()
    
    # Check waypoint calculation
    scaling = env.unwrapped.SCALING
    calculated_grasp_height = obj_pos[2] + (0.003 + 0.0102) * scaling
    print(f"Calculated grasp height: {calculated_grasp_height:.4f}")
    print(f"Actual waypoint height: {waypoint_height:.4f}")
    print(f"Calculation error: {abs(waypoint_height - calculated_grasp_height):.6f}")
    print()
    
    # Check TIP vs EEF offset
    tip_eef_offset = np.array(tip_pos) - np.array(eef_pos)
    print(f"TIP-EEF offset: {tip_eef_offset}")
    print(f"TIP-EEF distance: {np.linalg.norm(tip_eef_offset):.6f}")
    print()
    
    # Simulate reaching waypoint 2
    print("=== SIMULATING WAYPOINT REACH ===")
    for step in range(50):
        action = env.unwrapped.get_oracle_action(obs)
        
        if step > 10 and action[4] < 0:  # When grasping
            current_eef = obs['observation'][:3]
            current_tip, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
            current_obj, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
            
            eef_to_waypoint = np.linalg.norm(current_eef - waypoint_2[:3])
            tip_to_obj = np.linalg.norm(np.array(current_tip) - np.array(current_obj))
            
            print(f"Step {step}:")
            print(f"  EEF to waypoint: {eef_to_waypoint:.6f}")
            print(f"  TIP to object: {tip_to_obj:.6f}")
            print(f"  Gripper action: {action[4]:.2f}")
            
            # Check if we're at waypoint 2
            active_waypoints = [i for i, wp in enumerate(env.unwrapped._waypoints) if wp is not None]
            if 2 not in active_waypoints:
                print(f"  >>> Reached waypoint 2! TIP-Object distance: {tip_to_obj:.6f}")
                
                # Check why there's still distance
                print(f"  Object position: {current_obj}")
                print(f"  TIP position: {current_tip}")
                print(f"  Height diff (TIP-OBJ): {current_tip[2] - current_obj[2]:.6f}")
                break
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    debug_precision()