#!/usr/bin/env python3
"""
Evaluation script with optional action noise to test if model relies on exploration.
"""
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def evaluate_with_noise(model_path, curriculum_level=2, n_episodes=20, noise_std=0.1):
    """
    Evaluate model with different action noise levels.
    """
    print(f"Evaluating: {model_path}")
    print(f"Curriculum Level: {curriculum_level}")
    print("="*60)
    
    # Create environment
    env = gym.make('PegTransfer-v0', 
                   curriculum_level=curriculum_level,
                   use_dense_reward=True,
                   render_mode=None)
    env = FlattenDictObsWrapper(env)
    
    # Load model
    model = PPO.load(model_path)
    
    # Test with different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    for noise_std in noise_levels:
        successes = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                # Get deterministic action
                action, _ = model.predict(obs, deterministic=True)
                
                # Add Gaussian noise (similar to training exploration)
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, size=action.shape)
                    action = action + noise
                    # Clip to action space bounds
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, done, truncated, info = env.step(action)
            
            if info.get('is_success', False):
                successes += 1
        
        success_rate = successes / n_episodes
        print(f"Noise std={noise_std:.2f}: Success rate = {success_rate:.1%}")
    
    env.close()
    
    print("\nAnalysis:")
    print("If success rate INCREASES with noise, the model relies on exploration.")
    print("If success rate DECREASES with noise, the model has learned a robust policy.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("--level", type=int, default=2, help="Curriculum level")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per noise level")
    
    args = parser.parse_args()
    
    evaluate_with_noise(args.model_path, args.level, args.episodes)

if __name__ == "__main__":
    main()