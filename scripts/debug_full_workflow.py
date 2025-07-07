import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_full_workflow():
    """Debug the complete PegTransfer workflow to see where it fails"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== FULL WORKFLOW DEBUG ===")
    
    step_count = 0
    max_steps = 150
    
    while step_count < max_steps:
        action = env.unwrapped.get_oracle_action(obs)
        
        # Monitor key metrics
        if step_count % 10 == 0:
            tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
            obj_pos, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
            eef_pos = obs['observation'][:3]
            
            tip_to_obj = np.linalg.norm(np.array(tip_pos) - np.array(obj_pos))
            activated = env.unwrapped._activated
            contact_constraint = env.unwrapped._contact_constraint is not None
            gripper_action = action[4]
            
            # Check which waypoints are still active
            active_waypoints = []
            if env.unwrapped._waypoints is not None:
                for i, wp in enumerate(env.unwrapped._waypoints):
                    if wp is not None:
                        active_waypoints.append(i)
            
            print(f"Step {step_count:3d}: TIP-obj={tip_to_obj:.4f}, activated={activated}, constraint={contact_constraint}, gripper={gripper_action:.2f}, waypoints={active_waypoints}")
            
            # Check specific activation threshold
            if gripper_action < 0 and tip_to_obj < 0.015:  # Within 1.5cm (close to threshold)
                print(f"  >>> CLOSE TO ACTIVATION: TIP-object distance {tip_to_obj:.6f} vs threshold {2e-3 * env.unwrapped.SCALING:.6f}")
                print(f"  >>> TIP position: {tip_pos}")
                print(f"  >>> Object position: {obj_pos}")
                print(f"  >>> EEF position: {eef_pos}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if terminated or truncated:
            print(f"Episode ended at step {step_count}: terminated={terminated}, truncated={truncated}")
            if terminated:
                print("SUCCESS!")
            break
    
    print("=== FINAL STATE ===")
    tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
    obj_pos, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
    goal_pos = env.unwrapped.goal
    
    print(f"Final TIP-object distance: {np.linalg.norm(np.array(tip_pos) - np.array(obj_pos)):.6f}")
    print(f"Final object-goal distance: {np.linalg.norm(np.array(obj_pos) - goal_pos):.6f}")
    print(f"Final activated state: {env.unwrapped._activated}")
    print(f"Final constraint state: {env.unwrapped._contact_constraint is not None}")
    
    env.close()

if __name__ == "__main__":
    debug_full_workflow()