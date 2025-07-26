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
    
    # Monkey-patch the reward function
    original_reward = env._get_dense_reward
    env._get_dense_reward = lambda obs: reward_func(env, obs)
    
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
            if hasattr(env, '_activated') and env._activated >= 0:
                if hasattr(env, '_contact_constraint') and env._contact_constraint is not None:
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
    
    reward = 0.0
    
    # Only reward actual grasping
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    
    if is_grasped:
        # Reward based on transport progress only
        initial_dist = 0.2 * env.SCALING
        current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        progress = max(0, (initial_dist - current_dist) / initial_dist)
        reward = 5.0 * progress
    else:
        # Small reward for being close, penalty for failed grasp attempts
        dist_to_obj = np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'])
        jaw_angle = obs['observation'][6]
        
        if jaw_angle < 0 and dist_to_obj < 0.02 * env.SCALING:
            # Grasp attempt without success = penalty
            reward = -0.5
        else:
            # Small distance-based reward to guide exploration
            reward = max(0, 1 - dist_to_obj / (0.1 * env.SCALING)) * 0.1
    
    return reward


def staged_reward(env, obs):
    """Staged reward that only rewards complete sub-tasks."""
    if env._is_success(obs):
        return 10.0
    
    # Check grasp status
    is_grasped = env._activated >= 0 and env._contact_constraint is not None
    
    if not is_grasped:
        # No reward until successful grasp
        return 0.0
    
    # Only reward transport progress after successful grasp
    initial_dist = 0.2 * env.SCALING
    current_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    progress = max(0, (initial_dist - current_dist) / initial_dist)
    
    return 5.0 * progress


def main():
    print("Testing different reward schemes for PegTransfer...\n")
    
    reward_schemes = {
        "Original (Hackable)": original_dense_reward,
        "Penalty-Based": penalty_based_reward,
        "Staged": staged_reward
    }
    
    for name, reward_func in reward_schemes.items():
        print(f"\nTesting {name} reward scheme...")
        stats = test_reward_scheme(reward_func, num_episodes=5)
        
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Grasp Attempts/Episode: {stats['mean_grasp_attempts']:.1f}")
        print(f"  Grasp Success Rate: {stats['grasp_success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("- Original: High rewards from repeated failed grasps (reward hacking)")
    print("- Penalty-Based: Discourages failed attempts, guides exploration")
    print("- Staged: Forces learning complete skill chain, harder but cleaner")


if __name__ == "__main__":
    main()