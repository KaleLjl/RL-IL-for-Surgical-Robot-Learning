#!/usr/bin/env python3
"""
Test script to demonstrate the new Level 1 dense reward system.
Shows how rewards change as the robot approaches and stabilizes near the object.
"""
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium as gym
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def test_level1_dense_rewards():
    """Test the Level 1 dense reward system by simulating different distances."""
    
    print("Testing Level 1 Dense Reward System")
    print("=" * 50)
    
    # Create environment for Level 1 
    env = gym.make(
        'PegTransfer-v0',
        curriculum_level=1,
        render_mode=None
    )
    env = FlattenDictObsWrapper(env)
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"Object position: {obs['achieved_goal']}")
    print(f"Initial EEF position: {obs['observation'][:3]}")
    print(f"Initial distance: {np.linalg.norm(obs['observation'][:3] - obs['achieved_goal']):.4f}")
    print(f"Scaling factor: {env.unwrapped.SCALING}")
    print()
    
    # Test different distances by manually setting EEF position
    test_distances = [0.5, 0.3, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005]  # In scaled units
    
    print("Distance (scaled) | Distance (real) | Reward   | Explanation")
    print("-" * 70)
    
    for test_dist in test_distances:
        # Simulate EEF at different distances from object
        # Create a mock observation with the robot at specified distance
        mock_obs = obs.copy()
        
        # Set EEF position to be exactly test_dist away from object
        direction = np.array([1, 0, 0])  # Move in x direction
        mock_eef_pos = obs['achieved_goal'] + direction * test_dist
        mock_obs['observation'][:3] = mock_eef_pos
        
        # Reset stability counter for clean test
        env.unwrapped._approach_stable_steps = 0
        
        # Calculate reward manually by accessing the environment's reward function
        reward = env.unwrapped._get_level_1_dense_reward(mock_obs)
        
        # Convert to real-world distance
        real_dist = test_dist / env.unwrapped.SCALING
        
        # Determine explanation
        approach_threshold = 0.01 * env.unwrapped.SCALING
        if test_dist < approach_threshold:
            explanation = "Within success zone + precision bonus"
        elif test_dist < 0.1:
            explanation = "Close approach - good distance reward"
        else:
            explanation = "Far away - low distance reward"
        
        print(f"{test_dist:13.3f} | {real_dist:11.4f} | {reward:8.3f} | {explanation}")
    
    print()
    print("Stability Test (within success zone):")
    print("-" * 50)
    
    # Test stability bonus by simulating consecutive steps in success zone
    mock_obs = obs.copy()
    close_pos = obs['achieved_goal'] + np.array([0.005, 0, 0])  # Very close position
    mock_obs['observation'][:3] = close_pos
    
    # Simulate multiple steps at the same close position
    for step in range(8):
        # Manually set the stability counter (normally handled by _is_level_1_success)
        env.unwrapped._approach_stable_steps = step
        reward = env.unwrapped._get_level_1_dense_reward(mock_obs)
        
        if step < 5:
            status = f"Stable for {step} steps"
        elif step == 5:
            status = "SUCCESS! (5 stable steps reached)"
        else:
            status = f"Still successful ({step} stable steps)"
        
        print(f"Step {step}: Reward = {reward:6.3f} | {status}")
    
    print()
    print("Comparison: Level 1 vs Other Levels")
    print("-" * 40)
    
    # Test different curriculum levels to show the difference
    distance = 0.05  # Moderate distance
    mock_obs = obs.copy()
    mock_eef_pos = obs['achieved_goal'] + np.array([distance, 0, 0])
    mock_obs['observation'][:3] = mock_eef_pos
    
    for level in [1, 2, 3, 4]:
        env_test = gym.make('PegTransfer-v0', curriculum_level=level, render_mode=None)
        env_test = FlattenDictObsWrapper(env_test)
        obs_test, _ = env_test.reset()
        
        # Use the same mock position
        mock_obs_test = obs_test.copy()
        mock_obs_test['observation'][:3] = obs_test['achieved_goal'] + np.array([distance, 0, 0])
        
        reward = env_test.unwrapped._get_reward(mock_obs_test)
        
        if level == 1:
            reward_type = "Dense (continuous guidance)"
        else:
            reward_type = "Sparse (-1 for not success)"
        
        print(f"Level {level}: Reward = {reward:6.3f} | {reward_type}")
        env_test.close()
    
    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_level1_dense_rewards()