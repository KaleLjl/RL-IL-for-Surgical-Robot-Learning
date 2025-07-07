import gymnasium as gym
import numpy as np
import dvrk_gym
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_workspace():
    """Debug workspace limits and their effect on motion"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== WORKSPACE LIMITS DEBUG ===")
    
    # Check workspace limits
    workspace_limits = env.unwrapped.workspace_limits
    print(f"Workspace limits:")
    print(f"  X: [{workspace_limits[0, 0]:.4f}, {workspace_limits[0, 1]:.4f}]")
    print(f"  Y: [{workspace_limits[1, 0]:.4f}, {workspace_limits[1, 1]:.4f}]")
    print(f"  Z: [{workspace_limits[2, 0]:.4f}, {workspace_limits[2, 1]:.4f}]")
    print()
    
    # Check object and robot positions relative to workspace
    obj_pos, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
    eef_pos = obs['observation'][:3]
    
    print(f"Object position: {obj_pos}")
    print(f"EEF position: {eef_pos}")
    print()
    
    # Check if object is within workspace
    obj_in_workspace = all([
        workspace_limits[i, 0] <= obj_pos[i] <= workspace_limits[i, 1] 
        for i in range(3)
    ])
    print(f"Object within workspace: {obj_in_workspace}")
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if not (workspace_limits[i, 0] <= obj_pos[i] <= workspace_limits[i, 1]):
            print(f"  {axis}-axis violation: obj={obj_pos[i]:.4f}, limits=[{workspace_limits[i, 0]:.4f}, {workspace_limits[i, 1]:.4f}]")
    print()
    
    # Check action space clipping in _set_action
    action_limits_min = workspace_limits[:, 0] - np.array([0.02, 0.02, 0.0])
    action_limits_max = workspace_limits[:, 1] + np.array([0.02, 0.02, 0.15])
    
    print(f"Action clipping limits:")
    print(f"  X: [{action_limits_min[0]:.4f}, {action_limits_max[0]:.4f}]")
    print(f"  Y: [{action_limits_min[1]:.4f}, {action_limits_max[1]:.4f}]") 
    print(f"  Z: [{action_limits_min[2]:.4f}, {action_limits_max[2]:.4f}]")
    print()
    
    # Check if EEF can theoretically reach object height
    can_reach_obj_height = action_limits_min[2] <= obj_pos[2] <= action_limits_max[2]
    print(f"EEF can reach object height: {can_reach_obj_height}")
    if not can_reach_obj_height:
        print(f"  Object Z={obj_pos[2]:.4f} vs EEF Z limits=[{action_limits_min[2]:.4f}, {action_limits_max[2]:.4f}]")
    print()
    
    # Test manual downward movement to see where it gets clipped
    print("=== TESTING DOWNWARD MOVEMENT ===")
    for step in range(20):
        current_eef = obs['observation'][:3]
        
        # Try to move down aggressively
        manual_action = np.array([0.0, 0.0, -1.0, 0.0, 0.0])  # Maximum down
        
        print(f"Step {step:2d}: EEF Z={current_eef[2]:.4f}, action Z={manual_action[2]:.2f}")
        
        obs, reward, terminated, truncated, info = env.step(manual_action)
        
        new_eef = obs['observation'][:3]
        z_change = new_eef[2] - current_eef[2]
        
        print(f"         New EEF Z={new_eef[2]:.4f}, change={z_change:.4f}")
        
        if abs(z_change) < 0.001:  # No movement
            print(f"  >>> MOVEMENT STOPPED! Likely hit workspace limit.")
            print(f"      Current Z={new_eef[2]:.4f} vs min limit={action_limits_min[2]:.4f}")
            break
    
    env.close()

if __name__ == "__main__":
    debug_workspace()