#!/usr/bin/env python3
"""
Test script to compare different reward schemes for PegTransfer to fix reward hacking.
"""
import gymnasium as gym
import numpy as np
import dvrk_gym


def test_reward_scheme(reward_func, num_episodes=10):
    """Test a reward function and collect statistics."""
    env = gym.make('PegTransfer-v0', render_mode=None, use_dense_reward=True)
    
    # Monkey-patch the reward function - access through unwrapped environment
    original_reward = env.unwrapped._get_dense_reward
    env.unwrapped._get_dense_reward = lambda obs: reward_func(env.unwrapped, obs)
    
    rewards_collected = []
    grasp_attempts = []
    successful_grasps = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_grasp_attempts = 0
        ep_successful_grasps = 0
        
        while not done:
            # Simple policy: approach, grasp, lift
            action = env.action_space.sample()
            
            # Track grasp attempts
            if action[4] < 0:  # Gripper closing
                ep_grasp_attempts += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
            # Track successful grasps
            if hasattr(env.unwrapped, '_activated') and env.unwrapped._activated >= 0:
                if hasattr(env.unwrapped, '_contact_constraint') and env.unwrapped._contact_constraint is not None:
                    ep_successful_grasps = 1
        
        rewards_collected.append(ep_reward)
        grasp_attempts.append(ep_grasp_attempts)
        successful_grasps.append(ep_successful_grasps)
    
    env.close()
    
    return {
        'mean_reward': np.mean(rewards_collected),
        'std_reward': np.std(rewards_collected),
        'mean_grasp_attempts': np.mean(grasp_attempts),
        'grasp_success_rate': np.mean(successful_grasps)
    }


def original_dense_reward(env, obs):
    """Current reward system that leads to reward hacking."""
    reward = 0.0
    
    if env._is_success(obs):
        return 10.0
    
    eef_pos = obs['observation'][:3]
    obj_pos = obs['achieved_goal']
    goal_pos = obs['desired_goal']
    jaw_angle = obs['observation'][6]
    
    # Sub-goal 1: Approach object
    dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
    if dist_to_obj < 0.02 * env.SCALING:
        reward += 1.0
    
    # Sub-goal 2: Grasp attempt
    if jaw_angle < 0 and dist_to_obj < 0.02 * env.SCALING:
        reward += 2.0
    
    # Sub-goal 3: Successful grasp
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    if is_grasped:
        reward += 3.0
        
        # Sub-goal 4: Transport
        initial_dist = 0.2 * env.SCALING
        current_dist = np.linalg.norm(obj_pos - goal_pos)
        progress = max(0, (initial_dist - current_dist) / initial_dist)
        reward += progress * 3.0
    
    return reward


def penalty_based_reward(env, obs):
    """Reward system with penalties for failed grasps."""
    if env._is_success(obs):
        return 10.0
    
    reward = 0.0``
    
    # Check actual grasp status using environment state
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    
    if is_grasped:
        # Strong reward for successful grasp + transport progress
        initial_dist = 0.2 * env.SCALING
        current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        progress = max(0, (initial_dist - current_dist) / initial_dist)
        reward = 2.0 + (5.0 * progress)  # 2 for grasp, up to 5 for transport
    else:
        # Pre-grasp phase
        eef_pos = obs['observation'][:3]
        obj_pos = obs['achieved_goal']
        jaw_angle = obs['observation'][6]
        dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
        
        if env._activated >= 0 and jaw_angle < 0:
            # Gripper activated but no constraint yet
            if dist_to_obj < 0.01 * env.SCALING:  # Very close
                reward = 0.5  # Small reward for good positioning
            else:
                reward = -0.3  # Penalty for premature activation
        else:
            # Approach phase - small distance-based reward
            if dist_to_obj < 0.02 * env.SCALING:
                reward = 0.3
            else:
                reward = max(0, 1 - dist_to_obj / (0.1 * env.SCALING)) * 0.1
    
    return reward


def staged_reward(env, obs):
    """Staged reward that only rewards complete sub-tasks."""
    if env._is_success(obs):
        return 10.0
    
    # Check grasp status using proper environment state
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    
    if not is_grasped:
        # Very small distance bonus to guide initial exploration
        dist_to_obj = np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'])
        if dist_to_obj < 0.02 * env.SCALING:
            return 0.1  # Tiny reward when close
        return 0.0
    
    # Significant reward jump for successful grasp + transport progress
    initial_dist = 0.2 * env.SCALING
    current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    progress = max(0, (initial_dist - current_dist) / initial_dist)
    
    return 3.0 + (4.0 * progress)  # 3 for grasp, up to 4 for transport


def balanced_reward(env, obs):
    """Balanced reward with progressive sub-goals but no reward hacking."""
    if env._is_success(obs):
        return 10.0
    
    reward = 0.0
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    
    if is_grasped:
        # Major reward for grasp + transport
        initial_dist = 0.2 * env.SCALING
        current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        progress = max(0, (initial_dist - current_dist) / initial_dist)
        reward = 4.0 + (3.0 * progress)
    else:
        # Pre-grasp: progressive but controlled rewards
        eef_pos = obs['observation'][:3]
        obj_pos = obs['achieved_goal']
        jaw_angle = obs['observation'][6]
        dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
        
        # Distance-based approach reward (max 0.5)
        if dist_to_obj < 0.1 * env.SCALING:
            approach_reward = 0.5 * (1 - dist_to_obj / (0.1 * env.SCALING))
            reward += approach_reward
        
        # Positioning bonus when very close (max 0.5)
        if dist_to_obj < 0.01 * env.SCALING:
            reward += 0.5
        
        # Only penalize bad grasp attempts
        if env._activated >= 0 and jaw_angle < 0 and dist_to_obj > 0.02 * env.SCALING:
            reward -= 0.5
    
    return reward


def main():
    print("Testing different reward schemes for PegTransfer...\n")
    
    reward_schemes = {
        "Original (Hackable)": original_dense_reward,
        "Penalty-Based": penalty_based_reward,
        "Staged": staged_reward,
        "Balanced": balanced_reward
    }
    
    for name, reward_func in reward_schemes.items():
        print(f"\nTesting {name} reward scheme...")
        stats = test_reward_scheme(reward_func, num_episodes=5)
        
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Grasp Attempts/Episode: {stats['mean_grasp_attempts']:.1f}")
        print(f"  Grasp Success Rate: {stats['grasp_success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("- Original: Vulnerable to reward hacking (3 pts for approach+attempt)")
    print("- Penalty-Based: Penalizes premature grasps (-0.3), rewards actual success")
    print("- Staged: Big reward jump for grasp (3.0), minimal pre-grasp guidance")
    print("- Balanced: Max 1.0 pre-grasp, 4.0+ for grasp, prevents exploitation")
    print("\nKEY INSIGHT: All use _activated & _contact_constraint for proper grasp detection")


if __name__ == "__main__":
    main()