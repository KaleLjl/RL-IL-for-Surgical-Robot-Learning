import gymnasium as gym
import numpy as np
import dvrk_gym
import time

def debug_peg_transfer():
    """
    Debug script for PegTransfer oracle policy behavior
    """
    print("Starting PegTransfer debug session...")
    env = gym.make('PegTransfer-v0', render_mode='human')
    
    obs, info = env.reset()
    print("Environment reset complete")
    
    # Let environment settle
    for _ in range(10):
        env.step(np.zeros(5))
        time.sleep(0.1)
    
    print("\n=== Initial State ===")
    print(f"Object position: {obs['achieved_goal']}")
    print(f"Goal position: {obs['desired_goal']}")
    print(f"EEF position: {obs['observation'][:3]}")
    print(f"EEF orientation: {obs['observation'][3:6]}")
    print(f"Jaw angle: {obs['observation'][6]}")
    
    step_count = 0
    max_steps = 200
    
    while step_count < max_steps:
        # Get oracle action
        action = env.unwrapped.get_oracle_action(obs)
        
        print(f"\n=== Step {step_count} ===")
        print(f"Oracle action: {action}")
        print(f"Position delta: {action[:3]}")
        print(f"Yaw delta: {action[3]}")
        print(f"Gripper: {action[4]} ({'CLOSE' if action[4] < 0 else 'OPEN'})")
        
        # Check waypoints status
        if hasattr(env.unwrapped, '_waypoints') and env.unwrapped._waypoints is not None:
            active_waypoints = [i for i, wp in enumerate(env.unwrapped._waypoints) if wp is not None]
            if active_waypoints:
                current_waypoint = active_waypoints[0]
                waypoint = env.unwrapped._waypoints[current_waypoint]
                print(f"Active waypoint {current_waypoint}: pos={waypoint[:3]}, yaw={waypoint[3]}, gripper={waypoint[4]}")
                
                # Calculate distance to waypoint
                current_pos = obs['observation'][:3]
                current_yaw = obs['observation'][5]
                pos_dist = np.linalg.norm(waypoint[:3] - current_pos)
                yaw_dist = abs(waypoint[3] - current_yaw)
                print(f"Distance to waypoint: pos={pos_dist:.4f}, yaw={yaw_dist:.4f}")
        
        # Check activation status
        if hasattr(env.unwrapped, '_activated'):
            print(f"Activation status: {env.unwrapped._activated}")
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"New EEF position: {obs['observation'][:3]}")
        print(f"New object position: {obs['achieved_goal']}")
        print(f"Reward: {reward}")
        print(f"Success: {info.get('is_success', False)}")
        
        if terminated or truncated:
            print(f"\nEpisode ended after {step_count} steps")
            print(f"Final success: {info.get('is_success', False)}")
            break
        
        step_count += 1
        time.sleep(0.2)  # Slow down for observation
        
        # Emergency stop if waypoints aren't progressing
        if step_count > 50 and hasattr(env.unwrapped, '_waypoints'):
            if env.unwrapped._waypoints is not None:
                remaining = sum(1 for wp in env.unwrapped._waypoints if wp is not None)
                if remaining == len([wp for wp in env.unwrapped._waypoints if wp is not None]):
                    print("WARNING: No waypoint progress detected!")
                    break
    
    env.close()
    print("Debug session complete")

if __name__ == "__main__":
    debug_peg_transfer()