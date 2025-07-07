import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_activation():
    """Debug activation logic specifically"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== ACTIVATION DEBUG ===")
    
    # Check object links
    obj_id = env.unwrapped.obj_id
    obj_link1 = env.unwrapped.obj_link1
    
    print(f"Object ID: {obj_id}")
    print(f"Object link1: {obj_link1}")
    print(f"_waypoint_goal: {env.unwrapped._waypoint_goal}")
    print(f"_contact_approx: {env.unwrapped._contact_approx}")
    print(f"Activation threshold: {2e-3 * env.unwrapped.SCALING:.6f}")
    print()
    
    # Get positions using different links
    pos_obj_base, _ = get_link_pose(obj_id, -1)  # Base link
    pos_obj_link1, _ = get_link_pose(obj_id, obj_link1)  # Link 1
    tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
    
    print(f"Object base position (link -1): {pos_obj_base}")
    print(f"Object link1 position (link {obj_link1}): {pos_obj_link1}")
    print(f"TIP position: {tip_pos}")
    print()
    
    # Calculate distances
    dist_tip_to_base = np.linalg.norm(np.array(tip_pos) - np.array(pos_obj_base))
    dist_tip_to_link1 = np.linalg.norm(np.array(tip_pos) - np.array(pos_obj_link1))
    
    print(f"TIP to object base: {dist_tip_to_base:.6f}")
    print(f"TIP to object link1: {dist_tip_to_link1:.6f}")
    print(f"Difference: {abs(dist_tip_to_base - dist_tip_to_link1):.6f}")
    print()
    
    # Try to get closer by manual action
    print("=== MANUAL APPROACH TEST ===")
    for step in range(30):
        # Move down manually
        manual_action = np.array([0.0, 0.0, -0.8, 0.0, -0.5])  # Down and close gripper
        
        obs, reward, terminated, truncated, info = env.step(manual_action)
        
        # Check activation status every few steps
        if step % 5 == 0:
            tip_pos, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
            pos_obj_link1, _ = get_link_pose(obj_id, obj_link1)
            
            dist = np.linalg.norm(np.array(tip_pos) - np.array(pos_obj_link1))
            activated = env.unwrapped._activated
            threshold = 2e-3 * env.unwrapped.SCALING
            
            print(f"Step {step:2d}: TIP-obj={dist:.6f}, threshold={threshold:.6f}, activated={activated}")
            
            if dist < threshold:
                print(f"  >>> WITHIN THRESHOLD! Should activate.")
                break
    
    print("=== FINAL ACTIVATION CHECK ===")
    print(f"Final activated state: {env.unwrapped._activated}")
    print(f"Final constraint state: {env.unwrapped._contact_constraint is not None}")
    
    env.close()

if __name__ == "__main__":
    debug_activation()