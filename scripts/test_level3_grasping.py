#!/usr/bin/env python3
"""
Test script to verify Level 3 grasping detection mechanism.
This script tests whether the environment can properly detect successful grasping.
"""
import os
import sys
import numpy as np
import gymnasium as gym
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def test_grasping_detection():
    """Test grasping detection with oracle policy and manual actions."""
    
    print("="*60)
    print("Testing Level 3 Grasping Detection")
    print("="*60)
    
    # Create Level 3 environment
    env = gym.make('PegTransfer-v0', 
                   curriculum_level=3, 
                   use_dense_reward=True)  # No rendering in Docker
    
    # Don't wrap with FlattenDictObsWrapper so we can access the original environment
    print(f"Environment created: Level {env.unwrapped.curriculum_level}")
    print(f"Contact approximation: {env.unwrapped._contact_approx}")
    print(f"Waypoint goal: {env.unwrapped._waypoint_goal}")
    print(f"Success threshold: {env.unwrapped.success_threshold}")
    print()
    
    # Test 1: Oracle policy test
    print("TEST 1: Oracle Policy Grasping Test")
    print("-" * 40)
    
    obs, info = env.reset()
    print(f"Initial activated state: {env.unwrapped._activated}")
    print(f"Initial constraint state: {env.unwrapped._contact_constraint}")
    print(f"Initial is_success: {env.unwrapped._is_success(obs)}")
    print()
    
    max_steps = 200
    step_count = 0
    activation_step = None
    constraint_step = None
    success_step = None
    
    for step in range(max_steps):
        # Get oracle action
        action = env.unwrapped.get_oracle_action(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # Monitor key states
        activated = env.unwrapped._activated >= 0
        has_constraint = env.unwrapped._contact_constraint is not None
        is_success = env.unwrapped._is_success(obs)
        
        # Record when states change
        if activated and activation_step is None:
            activation_step = step
            print(f"✓ ACTIVATION detected at step {step}")
            print(f"  - activated value: {env.unwrapped._activated}")
            
            # Get detailed distance info
            if hasattr(env.unwrapped.psm1, 'TIP_LINK_INDEX'):
                from dvrk_gym.utils.pybullet_utils import get_link_pose
                pos_tip, _ = get_link_pose(env.unwrapped.psm1.body, env.unwrapped.psm1.TIP_LINK_INDEX)
                pos_obj, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
                tip_obj_distance = np.linalg.norm(np.array(pos_tip) - np.array(pos_obj))
                print(f"  - tip-object distance: {tip_obj_distance:.6f}")
                print(f"  - activation threshold: {2e-3 * env.unwrapped.SCALING:.6f}")
        
        if has_constraint and constraint_step is None:
            constraint_step = step
            print(f"✓ CONSTRAINT created at step {step}")
            print(f"  - constraint ID: {env.unwrapped._contact_constraint}")
        
        if is_success and success_step is None:
            success_step = step
            print(f"✓ SUCCESS detected at step {step}")
            print(f"  - reward: {reward:.3f}")
            break
        
        # Print periodic status
        if step % 50 == 0:
            jaw_angle = obs['observation'][6] if isinstance(obs, dict) else obs[6]
            print(f"Step {step}: activated={activated}, constraint={has_constraint}, "
                  f"success={is_success}, reward={reward:.3f}, jaw={jaw_angle:.3f}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break
        
        # Add small delay for visualization
        time.sleep(0.01)
    
    print()
    print("ORACLE TEST RESULTS:")
    print(f"- Activation achieved: {'YES at step ' + str(activation_step) if activation_step else 'NO'}")
    print(f"- Constraint created: {'YES at step ' + str(constraint_step) if constraint_step else 'NO'}")
    print(f"- Success achieved: {'YES at step ' + str(success_step) if success_step else 'NO'}")
    print(f"- Final activated state: {env.unwrapped._activated}")
    print(f"- Final constraint state: {env.unwrapped._contact_constraint is not None}")
    print()
    
    # Test 2: Manual action test to isolate issues
    print("TEST 2: Manual Action Test")
    print("-" * 40)
    
    obs, info = env.reset()
    
    # Get object position for manual targeting
    from dvrk_gym.utils.pybullet_utils import get_link_pose
    pos_obj, _ = get_link_pose(env.unwrapped.obj_id, env.unwrapped.obj_link1)
    print(f"Object position: {pos_obj}")
    
    # Manual approach sequence
    print("Manually approaching object...")
    
    # Step 1: Move above object
    target_above = [pos_obj[0], pos_obj[1], pos_obj[2] + 0.05]
    current_pos = obs['observation'][:3] if isinstance(obs, dict) else obs[:3]
    
    for i in range(20):
        delta = np.array(target_above) - current_pos
        delta = np.clip(delta / 0.01 / env.unwrapped.SCALING, -1, 1)
        action = np.array([delta[0], delta[1], delta[2], 0, 0.5])  # Open gripper
        
        obs, reward, terminated, truncated, info = env.step(action)
        current_pos = obs['observation'][:3] if isinstance(obs, dict) else obs[:3]
        
        if np.linalg.norm(current_pos - target_above) < 0.01:
            print(f"✓ Reached above position at step {i}")
            break
    
    # Step 2: Lower to grasp position
    print("Lowering to grasp position...")
    target_grasp = [pos_obj[0], pos_obj[1], pos_obj[2] + 0.005]  # Just above object
    
    for i in range(20):
        delta = np.array(target_grasp) - current_pos
        delta = np.clip(delta / 0.01 / env.unwrapped.SCALING, -1, 1)
        action = np.array([delta[0], delta[1], delta[2], 0, 0.5])  # Keep gripper open
        
        obs, reward, terminated, truncated, info = env.step(action)
        current_pos = obs['observation'][:3] if isinstance(obs, dict) else obs[:3]
        
        # Check for activation
        if env.unwrapped._activated >= 0:
            print(f"✓ ACTIVATION during approach at step {i}")
            break
        
        if np.linalg.norm(current_pos - target_grasp) < 0.005:
            print(f"✓ Reached grasp position at step {i}")
            break
    
    # Step 3: Close gripper
    print("Closing gripper...")
    for i in range(10):
        action = np.array([0, 0, 0, 0, -0.5])  # Close gripper
        obs, reward, terminated, truncated, info = env.step(action)
        
        activated = env.unwrapped._activated >= 0
        has_constraint = env.unwrapped._contact_constraint is not None
        is_success = env.unwrapped._is_success(obs)
        jaw_angle = obs['observation'][6] if isinstance(obs, dict) else obs[6]
        
        print(f"Gripper step {i}: activated={activated}, constraint={has_constraint}, success={is_success}, jaw={jaw_angle:.3f}")
        
        if is_success:
            print("✓ SUCCESS achieved!")
            break
    
    print()
    print("MANUAL TEST RESULTS:")
    print(f"- Final activated state: {env.unwrapped._activated}")
    print(f"- Final constraint state: {env.unwrapped._contact_constraint is not None}")
    print(f"- Final success state: {env.unwrapped._is_success(obs)}")
    
    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_grasping_detection()