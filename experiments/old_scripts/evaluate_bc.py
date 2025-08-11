import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import pathlib
import argparse

import dvrk_gym

def flatten_obs(obs):
    """
    Flattens a dictionary observation into a single numpy array.
    This is necessary because the policy was trained on flattened observations.
    """
    return np.concatenate([
        obs['observation'],
        obs['achieved_goal'],
        obs['desired_goal']
    ])

def main():
    """
    Evaluates a trained Behavioral Cloning model.
    """
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Evaluate a trained BC model")
    parser.add_argument("--model-path", required=True,
                       help="Path to the trained BC model (.zip file)")
    parser.add_argument("--env", default="NeedleReach-v0",
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to evaluate on")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for faster evaluation")
    
    args = parser.parse_args()
    
    print(f"Loading environment: {args.env}")
    render_mode = None if args.no_render else "human"
    env = gym.make(args.env, render_mode=render_mode)

    print(f"Loading model from: {args.model_path}")
    # The policy was saved as a standard SB3 ActorCriticPolicy,
    # so we should load it using the same class.
    # The `imitation.reconstruct_policy` has different expectations.
    policy = ActorCriticPolicy.load(args.model_path)

    successes = 0
    total_rewards = 0

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            # Flatten the observation to match the policy's input format
            flat_obs = flatten_obs(obs)
            
            action, _ = policy.predict(flat_obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not args.no_render:
                env.render()
            episode_reward += reward

        if info.get('is_success', False):
            successes += 1
        
        total_rewards += episode_reward
        print(f"Episode {episode + 1}/{args.episodes} - Reward: {episode_reward:.2f} - Success: {info.get('is_success', False)}")

    env.close()

    success_rate = (successes / args.episodes) * 100
    avg_reward = total_rewards / args.episodes

    print("\n--- Evaluation Summary ---")
    print(f"Total Episodes: {args.episodes}")
    print(f"Successes: {successes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
