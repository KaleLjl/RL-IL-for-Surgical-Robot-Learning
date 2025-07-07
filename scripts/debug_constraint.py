import gymnasium as gym
import numpy as np
import dvrk_gym
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_constraint():
    """Debug constraint creation during grasping"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== CONSTRAINT CREATION DEBUG ===")
    
    goal_height = env.unwrapped.goal[2]
    constraint_threshold = goal_height + 0.01 * env.unwrapped.SCALING
    
    print(f"Goal height: {goal_height:.4f}")
    print(f"Constraint threshold: {constraint_threshold:.4f}")
    print()
    
    for step in range(60):
        action = env.unwrapped.get_oracle_action(obs)
        
        # Monitor during grasping phase
        if action[4] < 0:  # When gripper is closing
            obj_pose = get_link_pose(env.unwrapped.obj_id, -1)
            obj_height = obj_pose[0][2]
            activated = env.unwrapped._activated
            constraint_exists = env.unwrapped._contact_constraint is not None
            meets_requirement = env.unwrapped._meet_contact_constraint_requirement()
            
            if activated >= 0:
                print(f"Step {step:2d}: activated={activated}, constraint={constraint_exists}")
                print(f"         obj_height={obj_height:.4f}, threshold={constraint_threshold:.4f}")
                print(f"         meets_requirement={meets_requirement}, height_diff={obj_height - constraint_threshold:.4f}")
                
                if constraint_exists:
                    print(f"         >>> CONSTRAINT CREATED!")
                elif meets_requirement:
                    print(f"         >>> Should create constraint but didn't!")
                else:
                    print(f"         >>> Object too low for constraint")
                print()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print("=== FINAL STATE ===")
    print(f"Final activated: {env.unwrapped._activated}")
    print(f"Final constraint: {env.unwrapped._contact_constraint is not None}")
    
    env.close()

if __name__ == "__main__":
    debug_constraint()