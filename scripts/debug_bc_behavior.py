#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import pickle
from stable_baselines3 import PPO
import dvrk_gym

def analyze_bc_vs_expert_behavior(model_path, expert_data_path, env_name="PegTransfer-v0"):
    """Compare BC model behavior with expert behavior"""
    
    print("=== BC vs Expert Behavior Analysis ===")
    
    # Load expert data
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)
    
    # Load BC model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make(env_name)
    
    print(f"Loaded {len(trajectories)} expert trajectories")
    print(f"Loaded BC model from {model_path}")
    
    # Test 1: Compare action statistics
    print("\n=== Action Statistics Comparison ===")
    
    # Expert actions
    all_expert_actions = []
    for traj in trajectories:
        all_expert_actions.extend(traj['acts'])
    expert_actions = np.array(all_expert_actions)
    
    print(f"Expert actions shape: {expert_actions.shape}")
    print(f"Expert action mean: {np.mean(expert_actions, axis=0)}")
    print(f"Expert action std: {np.std(expert_actions, axis=0)}")
    print(f"Expert action range: [{np.min(expert_actions, axis=0)}, {np.max(expert_actions, axis=0)}]")
    
    # BC model actions (sample from environment)
    bc_actions = []
    num_steps = min(1000, len(all_expert_actions))  # Sample 1000 steps
    
    obs, info = env.reset()
    for _ in range(num_steps):
        # Flatten observation for BC model
        flat_obs = np.concatenate([
            obs['observation'],
            obs['achieved_goal'], 
            obs['desired_goal']
        ])
        
        action, _ = model.predict(flat_obs, deterministic=True)
        bc_actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    bc_actions = np.array(bc_actions)
    
    print(f"\nBC actions shape: {bc_actions.shape}")
    print(f"BC action mean: {np.mean(bc_actions, axis=0)}")
    print(f"BC action std: {np.std(bc_actions, axis=0)}")
    print(f"BC action range: [{np.min(bc_actions, axis=0)}, {np.max(bc_actions, axis=0)}]")
    
    # Test 2: Run a single episode with detailed logging
    print("\n=== Single Episode Analysis ===")
    
    obs, info = env.reset()
    episode_actions = []
    episode_rewards = []
    
    for step in range(50):  # Run for 50 steps
        # Flatten observation
        flat_obs = np.concatenate([
            obs['observation'],
            obs['achieved_goal'], 
            obs['desired_goal']
        ])
        
        action, _ = model.predict(flat_obs, deterministic=True)
        episode_actions.append(action.copy())
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        
        if step < 10:  # Print first 10 steps
            print(f"Step {step}: Action={action}, Reward={reward:.3f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}, Total reward: {sum(episode_rewards):.3f}")
            break
    
    # Test 3: Compare with expert episode
    print("\n=== Expert Episode Comparison ===")
    expert_traj = trajectories[0]  # Use first expert trajectory
    expert_obs = expert_traj['obs']
    expert_acts = expert_traj['acts']
    
    print(f"Expert episode length: {len(expert_acts)}")
    print(f"Expert first 5 actions:")
    for i in range(min(5, len(expert_acts))):
        print(f"  Step {i}: {expert_acts[i]}")
    
    # Test 4: Action magnitude analysis
    print("\n=== Action Magnitude Analysis ===")
    expert_magnitudes = np.linalg.norm(expert_actions, axis=1)
    bc_magnitudes = np.linalg.norm(bc_actions, axis=1)
    
    print(f"Expert action magnitude - Mean: {np.mean(expert_magnitudes):.4f}, Std: {np.std(expert_magnitudes):.4f}")
    print(f"BC action magnitude - Mean: {np.mean(bc_magnitudes):.4f}, Std: {np.std(bc_magnitudes):.4f}")
    
    # Test 5: Check if BC model is outputting reasonable values
    print("\n=== BC Model Output Analysis ===")
    
    # Test with expert observation
    if len(expert_obs['observation']) > 0:
        expert_flat_obs = np.concatenate([
            expert_obs['observation'][0],
            expert_obs['achieved_goal'][0],
            expert_obs['desired_goal'][0]
        ])
        
        bc_action_for_expert_obs, _ = model.predict(expert_flat_obs, deterministic=True)
        print(f"Expert obs[0]: {expert_flat_obs[:5]}...")  # First 5 elements
        print(f"Expert action[0]: {expert_acts[0]}")
        print(f"BC action for expert obs[0]: {bc_action_for_expert_obs}")
        print(f"Action difference: {expert_acts[0] - bc_action_for_expert_obs}")
    
    env.close()
    
    # Summary
    print("\n=== Analysis Summary ===")
    action_diff_mean = np.abs(np.mean(expert_actions, axis=0) - np.mean(bc_actions, axis=0))
    print(f"Mean action difference: {action_diff_mean}")
    print(f"Max action difference: {np.max(action_diff_mean)}")
    
    if np.max(action_diff_mean) > 0.1:
        print("⚠️  Large action differences detected - BC may not be learning expert behavior")
    else:
        print("✅ Action distributions are similar")

if __name__ == "__main__":
    model_path = "models/bc_peg_transfer_1752144939.zip"
    expert_data_path = "data/expert_data_peg_transfer.pkl"
    
    analyze_bc_vs_expert_behavior(model_path, expert_data_path)