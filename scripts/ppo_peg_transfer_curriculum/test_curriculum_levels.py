#!/usr/bin/env python3
"""
Test Script for PPO Curriculum Learning Levels

This script allows testing each curriculum level independently to verify
the implementation and success criteria.
"""
import os
import sys
import argparse
import time
import numpy as np
import gymnasium as gym

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dvrk_gym
from curriculum_config import get_level_config, CURRICULUM_LEVELS


def test_level(env_name: str, level: int, n_episodes: int = 10, render: bool = False):
    """Test a specific curriculum level."""
    
    level_config = get_level_config(level)
    print(f"\n{'='*60}")
    print(f"Testing Curriculum Level {level}: {level_config['name']}")
    print(f"Description: {level_config['description']}")
    print(f"Max episode steps: {level_config['max_episode_steps']}")
    print(f"Success criteria: {level_config['success_criteria']}")
    print(f"{'='*60}\n")
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(
        env_name,
        render_mode=render_mode,
        use_dense_reward=False,  # Always sparse for curriculum
        early_exit_enabled=True,
        curriculum_level=level
    )
    
    # Test statistics
    successes = 0
    total_rewards = []
    episode_lengths = []
    early_exits = 0
    
    # Detailed tracking for analysis
    approach_successes = 0
    grasp_successes = 0
    transport_successes = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # Episode tracking
        achieved_approach = False
        achieved_grasp = False
        achieved_transport = False
        min_distance = float('inf')
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not (done or truncated) and step_count < level_config['max_episode_steps']:
            # Use oracle action for testing
            action = env.unwrapped.get_oracle_action(obs)
            
            # Add some noise for more realistic testing
            if np.random.random() < 0.1:  # 10% noise
                action += np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action, -1, 1)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Track progress
            eef_pos = obs['observation'][:3]
            obj_pos = obs['achieved_goal']
            distance = np.linalg.norm(eef_pos - obj_pos)
            min_distance = min(min_distance, distance)
            
            # Check approach
            if distance < 0.01 * env.unwrapped.SCALING:
                achieved_approach = True
            
            # Check grasp
            if (hasattr(env.unwrapped, '_activated') and 
                env.unwrapped._activated >= 0 and 
                env.unwrapped._contact_constraint is not None):
                achieved_grasp = True
            
            # Check transport progress
            if achieved_grasp:
                goal_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
                if goal_distance < 0.02 * env.unwrapped.SCALING:
                    achieved_transport = True
            
            # Print progress for debugging
            if step_count % 10 == 0 or done or truncated:
                print(f"  Step {step_count:3d}: Distance={distance:.3f}, "
                      f"Reward={reward:+.1f}, Total={episode_reward:+.1f}")
        
        # Record results
        success = info.get('is_success', False)
        early_exit = info.get('early_exit', False)
        
        if success:
            successes += 1
        if early_exit:
            early_exits += 1
        if achieved_approach:
            approach_successes += 1
        if achieved_grasp:
            grasp_successes += 1
        if achieved_transport:
            transport_successes += 1
            
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Episode summary
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'} "
              f"({'early exit' if early_exit else 'full episode'})")
        print(f"  Progress: Approach={'✓' if achieved_approach else '✗'}, "
              f"Grasp={'✓' if achieved_grasp else '✗'}, "
              f"Transport={'✓' if achieved_transport else '✗'}")
        print(f"  Min distance: {min_distance:.3f}")
    
    env.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Level {level} Test Summary ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success rate: {successes/n_episodes:.1%} ({successes}/{n_episodes})")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Early exits: {early_exits/n_episodes:.1%} ({early_exits}/{n_episodes})")
    print()
    print(f"Sub-task success rates:")
    print(f"  Approach: {approach_successes/n_episodes:.1%}")
    print(f"  Grasp: {grasp_successes/n_episodes:.1%}")
    print(f"  Transport: {transport_successes/n_episodes:.1%}")
    
    # Check advancement criteria
    criteria = level_config['advancement']['success_rate_threshold']
    print(f"\nAdvancement threshold: {criteria:.0%}")
    if successes/n_episodes >= criteria:
        print("✓ Meets advancement criteria!")
    else:
        print("✗ Does not meet advancement criteria")
    
    return successes/n_episodes


def test_all_levels(env_name: str, n_episodes: int = 10):
    """Test all curriculum levels sequentially."""
    
    print("\n" + "="*80)
    print("TESTING ALL CURRICULUM LEVELS")
    print("="*80)
    
    results = {}
    
    for level in range(1, 5):
        success_rate = test_level(env_name, level, n_episodes, render=False)
        results[level] = success_rate
        time.sleep(1)  # Brief pause between levels
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL CURRICULUM TEST SUMMARY")
    print("="*80)
    
    for level, success_rate in results.items():
        config = get_level_config(level)
        print(f"Level {level} ({config['name']}): {success_rate:.1%} success rate")
    
    # Test progression logic
    print("\nProgression Analysis:")
    can_complete = True
    for level in range(1, 5):
        config = get_level_config(level)
        threshold = config['advancement']['success_rate_threshold']
        
        if results[level] >= threshold:
            print(f"  Level {level}: ✓ Can advance (achieved {results[level]:.1%} >= {threshold:.0%})")
        else:
            print(f"  Level {level}: ✗ Cannot advance (achieved {results[level]:.1%} < {threshold:.0%})")
            can_complete = False
            break
    
    if can_complete:
        print("\n✓ Oracle policy can complete full curriculum!")
    else:
        print(f"\n✗ Oracle policy stuck at Level {level}")


def test_specific_scenarios(env_name: str, level: int):
    """Test specific challenging scenarios for each level."""
    
    print(f"\n{'='*60}")
    print(f"Testing Specific Scenarios for Level {level}")
    print(f"{'='*60}\n")
    
    env = gym.make(
        env_name,
        render_mode=None,
        use_dense_reward=False,
        early_exit_enabled=True,
        curriculum_level=level
    )
    
    if level == 1:
        print("Scenario: Testing approach stability requirement")
        obs, _ = env.reset()
        
        # Move close to object
        for _ in range(20):
            action = env.unwrapped.get_oracle_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print(f"  Success after maintaining stable position!")
                break
        
        # Test unstable approach (oscillating)
        print("\nScenario: Testing unstable approach (should fail)")
        obs, _ = env.reset()
        stable_steps = 0
        
        for step in range(50):
            if step % 10 < 5:
                action = np.array([0.5, 0, 0, 0, 0])  # Move away
            else:
                action = np.array([-0.5, 0, 0, 0, 0])  # Move close
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                print(f"  Terminated at step {step} (should not succeed)")
                break
    
    elif level == 2:
        print("Scenario: Testing grasp stability requirement")
        # Test achieving grasp but not maintaining it
        # (Implementation would depend on specific physics)
        pass
    
    elif level == 3:
        print("Scenario: Testing transport without drops")
        # Test moving too fast or carelessly during transport
        pass
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test PPO Curriculum Learning Levels"
    )
    
    parser.add_argument(
        "--env",
        default="PegTransfer-v0",
        help="Environment name"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        help="Test specific level (if not specified, tests all levels)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per level"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during testing"
    )
    
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Test specific challenging scenarios"
    )
    
    args = parser.parse_args()
    
    if args.scenarios and args.level:
        test_specific_scenarios(args.env, args.level)
    elif args.level:
        test_level(args.env, args.level, args.episodes, args.render)
    else:
        test_all_levels(args.env, args.episodes)


if __name__ == "__main__":
    main()