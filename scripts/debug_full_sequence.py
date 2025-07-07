import gymnasium as gym
import numpy as np
import dvrk_gym
import time

def debug_full_sequence():
    """Debug the complete peg transfer sequence"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== COMPLETE SEQUENCE DEBUGGING ===")
    print()
    
    initial_obj_pos = obs['achieved_goal'].copy()
    goal_pos = obs['desired_goal'].copy()
    
    print(f"Initial object position: {initial_obj_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Initial distance to goal: {np.linalg.norm(initial_obj_pos[:2] - goal_pos[:2]):.6f}")
    print()
    
    waypoint_stage = -1
    object_positions = []
    
    for step in range(150):
        action = env.unwrapped.get_oracle_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_obj_pos = obs['achieved_goal']
        object_positions.append(current_obj_pos.copy())
        
        # Track waypoint progression
        if env.unwrapped._waypoints is not None:
            active_waypoints = [i for i, wp in enumerate(env.unwrapped._waypoints) if wp is not None]
            if active_waypoints:
                current_stage = min(active_waypoints)
                if current_stage != waypoint_stage:
                    waypoint_stage = current_stage
                    print(f"Step {step}: Advanced to waypoint {waypoint_stage}")
                    print(f"  Object pos: {current_obj_pos}")
                    print(f"  Distance to goal: {np.linalg.norm(current_obj_pos[:2] - goal_pos[:2]):.6f}")
                    
                    if waypoint_stage == 2:  # Grasping
                        print(f"  Activation status: {env.unwrapped._activated}")
                        print(f"  Constraint exists: {env.unwrapped._contact_constraint is not None}")
                    
                    if waypoint_stage == 5:  # Release
                        print(f"  Final object position: {current_obj_pos}")
                        print(f"  Final distance to goal: {np.linalg.norm(current_obj_pos[:2] - goal_pos[:2]):.6f}")
                        print(f"  Success: {info.get('is_success', False)}")
        
        # Check if object moved significantly
        if step > 0:
            movement = np.linalg.norm(current_obj_pos - object_positions[-2])
            if movement > 0.01:  # Significant movement
                if step % 20 == 0:  # Print every 20 steps during movement
                    print(f"Step {step}: Object moving, pos={current_obj_pos}, dist_to_goal={np.linalg.norm(current_obj_pos[:2] - goal_pos[:2]):.6f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Final object position: {current_obj_pos}")
            print(f"Final distance to goal: {np.linalg.norm(current_obj_pos[:2] - goal_pos[:2]):.6f}")
            print(f"Height difference: {abs(current_obj_pos[2] - goal_pos[2]):.6f}")
            print(f"Final success: {info.get('is_success', False)}")
            
            # Check success thresholds
            success_2d = np.linalg.norm(current_obj_pos[:2] - goal_pos[:2]) < 5e-3 * env.unwrapped.SCALING
            success_height = abs(current_obj_pos[2] - goal_pos[2]) < 4e-3 * env.unwrapped.SCALING
            print(f"2D success: {success_2d} (threshold: {5e-3 * env.unwrapped.SCALING:.6f})")
            print(f"Height success: {success_height} (threshold: {4e-3 * env.unwrapped.SCALING:.6f})")
            break
    
    env.close()

if __name__ == "__main__":
    debug_full_sequence()