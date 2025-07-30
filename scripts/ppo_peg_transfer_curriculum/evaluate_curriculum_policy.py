#!/usr/bin/env python3
"""
Simple Model Evaluation Script for PPO Curriculum Learning
"""
import os
import sys
import argparse
import json
import re
import numpy as np
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
from curriculum_config import get_level_config


def detect_level_from_path(model_path: str) -> int:
    """Automatically detect curriculum level from model filename."""
    
    # Try to match patterns like "level_1", "level_2", etc.
    patterns = [
        r"level[_\s]*(\d)",      # level_1, level 1
        r"l(\d)_",               # l1_
        r"_l(\d)_",              # _l1_
        r"curriculum_(\d)",      # curriculum_1
    ]
    
    filename = os.path.basename(model_path).lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 7:
                return level
    
    # If no level found in filename, ask user
    return None


def evaluate_model(model_path: str, level: int = None, n_episodes: int = 50, render: bool = False):
    """Evaluate a trained model."""
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Auto-detect level if not provided
    if level is None:
        level = detect_level_from_path(model_path)
        if level is None:
            print("Could not detect level from filename.")
            print("Please specify level with --level (1-7)")
            return
        print(f"Auto-detected Level {level} from filename")
    
    config = get_level_config(level)
    print(f"\nEvaluating: {os.path.basename(model_path)}")
    print(f"Level {level}: {config['name']}")
    print("-" * 60)
    
    # Create environment
    env = gym.make(
        "PegTransfer-v0",
        render_mode="human" if render else None,
        use_dense_reward=True,  # Match training environment
        curriculum_level=level
    )
    env = FlattenDictObsWrapper(env)
    
    # Load model
    model = PPO.load(model_path)
    
    # Track metrics
    successes = 0
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        # Check if successful
        if info.get('is_success', False):
            successes += 1
        
        episode_rewards.append(episode_reward)
        
        # Progress update
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode + 1}/{n_episodes} episodes")
    
    env.close()
    
    # Results
    success_rate = successes / n_episodes
    mean_reward = np.mean(episode_rewards)
    
    print(f"\nRESULTS:")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Average Reward: {mean_reward:.2f}")
    print(f"  Total Episodes: {n_episodes}")
    
    # Save results
    results = {
        "model_path": model_path,
        "level": level,
        "level_name": config['name'],
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "n_episodes": n_episodes,
        "evaluation_date": datetime.now().isoformat()
    }
    
    # Save to same directory as model
    save_dir = os.path.dirname(model_path)
    save_filename = f"eval_{os.path.basename(model_path).replace('.zip', '')}.json"
    save_path = os.path.join(save_dir, save_filename)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")








def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO Curriculum Model"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model file (e.g., model_level_1_final.zip)"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Curriculum level (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of test episodes (default: 50)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show visualization"
    )
    
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate_model(
        model_path=args.model_path,
        level=args.level,
        n_episodes=args.episodes,
        render=args.render
    )


if __name__ == "__main__":
    main()