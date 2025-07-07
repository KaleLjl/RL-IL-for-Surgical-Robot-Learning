import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_gripper_movement():
    """Debug why gripper doesn't reach the object"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== GRIPPER MOVEMENT DEBUG ===")
    
    psm = env.unwrapped.psm1
    obj_id = env.unwrapped.obj_id
    
    # Initial positions
    pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
    pos_obj, _ = get_link_pose(obj_id, env.unwrapped.obj_link1)
    
    print(f"Initial TIP position: {pos_tip}")
    print(f"Initial OBJ position: {pos_obj}")
    print(f"Initial distance: {np.linalg.norm(np.array(pos_tip) - np.array(pos_obj)):.6f}")
    print()
    
    print("=== WAYPOINT ANALYSIS ===")
    if env.unwrapped._waypoints is None:
        env.unwrapped._define_waypoints()
    
    for i, wp in enumerate(env.unwrapped._waypoints):
        if wp is not None:
            print(f"Waypoint {i}: [{wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}, {wp[3]:.4f}, {wp[4]:.2f}]")
    print()
    
    print("=== STEP-BY-STEP MOVEMENT ===")
    
    for step in range(100):
        action = env.unwrapped.get_oracle_action(obs)
        
        # Get current positions before action
        pos_tip_before, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
        pos_obj_current, _ = get_link_pose(obj_id, env.unwrapped.obj_link1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get positions after action
        pos_tip_after, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
        
        if step < 20 or step % 10 == 0:  # Print first 20 steps and every 10th
            distance_before = np.linalg.norm(np.array(pos_tip_before) - np.array(pos_obj_current))
            distance_after = np.linalg.norm(np.array(pos_tip_after) - np.array(pos_obj_current))
            movement = np.linalg.norm(np.array(pos_tip_after) - np.array(pos_tip_before))
            
            print(f"Step {step:2d}:")
            print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.2f}]")
            print(f"  TIP before: [{pos_tip_before[0]:.4f}, {pos_tip_before[1]:.4f}, {pos_tip_before[2]:.4f}]")
            print(f"  TIP after:  [{pos_tip_after[0]:.4f}, {pos_tip_after[1]:.4f}, {pos_tip_after[2]:.4f}]")
            print(f"  OBJ pos:    [{pos_obj_current[0]:.4f}, {pos_obj_current[1]:.4f}, {pos_obj_current[2]:.4f}]")
            print(f"  Distance before: {distance_before:.6f}")
            print(f"  Distance after:  {distance_after:.6f}")
            print(f"  TIP movement: {movement:.6f}")
            
            # Check active waypoint
            active_waypoints = [i for i, wp in enumerate(env.unwrapped._waypoints) if wp is not None]
            if active_waypoints:
                current_wp = min(active_waypoints)
                print(f"  Active waypoint: {current_wp}")
                wp = env.unwrapped._waypoints[current_wp]
                wp_distance = np.linalg.norm(np.array(pos_tip_after) - np.array(wp[:3]))
                print(f"  Distance to waypoint: {wp_distance:.6f}")
            
            print()
        
        # Check if we're getting close enough for contact
        final_distance = np.linalg.norm(np.array(pos_tip_after) - np.array(pos_obj_current))
        if final_distance < 0.05:  # Within 5cm
            print(f"*** CLOSE APPROACH at step {step}! Distance: {final_distance:.6f} ***")
            
            # Test contact detection at close range
            contacts_6 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
            contacts_7 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            obj_contacts_6 = [c for c in contacts_6 if c[2] == obj_id]
            obj_contacts_7 = [c for c in contacts_7 if c[2] == obj_id]
            
            print(f"  Contacts at close range - Link6: {len(obj_contacts_6)}, Link7: {len(obj_contacts_7)}")
            
            if obj_contacts_6 or obj_contacts_7:
                print("  >>> CONTACT DETECTED! <<<")
                break
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Final distance: {final_distance:.6f}")
            break
    
    env.close()

if __name__ == "__main__":
    debug_gripper_movement()