#!/usr/bin/env python3
"""
Evaluation script that pauses when the model reaches its target instead of terminating.
Allows visual inspection of where the model stops.
"""
import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def evaluate_with_pause(model_path, env_name="PegTransfer-v0", curriculum_level=2, n_episodes=5):
    """
    Evaluate a trained model and pause when it reaches the target.
    
    Args:
        model_path: Path to the trained model
        env_name: Environment name
        curriculum_level: Curriculum level (1-4)
        n_episodes: Number of episodes to run
    """
    
    print(f"Loading model from: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Curriculum level: {curriculum_level}")
    print("="*60)
    
    # Create environment with rendering
    env = gym.make(env_name, 
                   curriculum_level=curriculum_level,
                   use_dense_reward=True,  # Use dense reward for PPO models
                   render_mode="human")
    
    # Wrap environment to flatten observations (required for PPO models)
    env = FlattenDictObsWrapper(env)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        obs, info = env.reset()
        step_count = 0
        episode_reward = 0
        
        print("Model is acting... Press Ctrl+C to stop early if needed.")
        
        while True:
            # Get action from the trained model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            episode_reward += reward
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                if hasattr(env.unwrapped, '_get_achieved_goal'):
                    current_pos = env.unwrapped._get_achieved_goal()
                    desired_goal = env.unwrapped._get_desired_goal()
                    distance = np.linalg.norm(current_pos - desired_goal)
                    print(f"Step {step_count}: distance to goal = {distance:.4f}")
                else:
                    print(f"Step {step_count}: episode reward = {episode_reward:.2f}")
            
            # Check if episode should end
            if terminated or truncated:
                print(f"\nEpisode ended at step {step_count}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Episode reward: {episode_reward:.2f}")
                
                # Get final state information
                if hasattr(env.unwrapped, '_get_achieved_goal'):
                    final_pos = env.unwrapped._get_achieved_goal()
                    desired_goal = env.unwrapped._get_desired_goal()
                    final_distance = np.linalg.norm(final_pos - desired_goal)
                    print(f"Final distance to goal: {final_distance:.4f}")
                
                is_success = info.get('is_success', False)
                print(f"Success: {is_success}")
                
                print(f"\nModel stopped here. Observe the final position.")
                print(f"Press any key to continue to next episode...")
                input()
                break
            
            # Add small delay to make it observable
            time.sleep(0.02)
    
    print(f"\nEvaluation completed for {n_episodes} episodes.")
    print("Closing environment...")
    env.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model with pause at target")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--env", type=str, default="PegTransfer-v0",
                       choices=["PegTransfer-v0", "NeedleReach-v0"],
                       help="Environment name")
    parser.add_argument("--curriculum-level", type=int, default=2,
                       choices=[1, 2, 3, 4],
                       help="Curriculum level")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    evaluate_with_pause(
        model_path=args.model_path,
        env_name=args.env,
        curriculum_level=args.curriculum_level,
        n_episodes=args.episodes
    )

if __name__ == "__main__":
    main()