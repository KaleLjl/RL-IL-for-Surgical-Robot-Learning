import gymnasium as gym
import numpy as np
import os
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

def evaluate_with_action_noise(policy, env, episodes=50, noise_std=0.1):
    """
    Evaluate BC policy with Gaussian noise added to actions.
    This tests robustness to distribution shift, which predicts PPO+IL performance.
    
    Args:
        policy: Trained BC policy
        env: Environment to evaluate in
        episodes: Number of episodes to evaluate
        noise_std: Standard deviation of Gaussian noise to add to actions
        
    Returns:
        Tuple of (success_rate, average_reward, episode_rewards)
    """
    successes = 0
    total_rewards = 0
    episode_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            # Get action from policy
            flat_obs = flatten_obs(obs)
            action, _ = policy.predict(flat_obs, deterministic=True)
            
            # Add Gaussian noise to action
            action_noise = np.random.normal(0, noise_std, action.shape)
            noisy_action = action + action_noise
            
            # Clip action to valid range (assuming [-1, 1])
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = env.step(noisy_action)
            episode_reward += reward
        
        if info.get('is_success', False):
            successes += 1
        
        total_rewards += episode_reward
        episode_rewards.append(episode_reward)
    
    success_rate = (successes / episodes) * 100
    avg_reward = total_rewards / episodes
    
    return success_rate, avg_reward, episode_rewards

def main():
    """
    Evaluates a trained Behavioral Cloning model.
    """
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Evaluate a trained BC model")
    parser.add_argument("--model", required=True,
                       help="Path to the trained BC model (.zip file)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for evaluation results (if not specified, only console output)")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for faster evaluation")
    parser.add_argument("--action-noise-test", action="store_true",
                       help="Enable action noise robustness testing")
    parser.add_argument("--noise-std", type=float, default=0.1,
                       help="Standard deviation of action noise for robustness testing")
    
    args = parser.parse_args()
    
    # Auto-detect environment from model path
    if "needle_reach" in args.model.lower():
        env_name = "NeedleReach-v0"
    elif "peg_transfer" in args.model.lower():
        env_name = "PegTransfer-v0"
    else:
        # Default to NeedleReach if can't detect
        env_name = "NeedleReach-v0"
        print(f"Warning: Could not detect environment from model path, using {env_name}")
    
    print(f"Loading environment: {env_name}")
    render_mode = None if args.no_render else "human"
    env = gym.make(env_name, render_mode=render_mode)

    print(f"Loading model from: {args.model}")
    # The policy was saved as a standard SB3 ActorCriticPolicy,
    # so we should load it using the same class.
    # The `imitation.reconstruct_policy` has different expectations.
    policy = ActorCriticPolicy.load(args.model)

    # Standard evaluation
    successes = 0
    total_rewards = 0
    episode_rewards = []

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
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{args.episodes} - Reward: {episode_reward:.2f} - Success: {info.get('is_success', False)}")

    success_rate = (successes / args.episodes) * 100
    avg_reward = total_rewards / args.episodes
    
    # Action noise robustness test (if enabled)
    action_noise_success_rate = None
    action_noise_avg_reward = None
    action_noise_episode_rewards = []
    composite_score = success_rate  # Default to standard score
    
    if args.action_noise_test:
        print(f"\n--- Running Action Noise Robustness Test (noise_std={args.noise_std}) ---")
        action_noise_success_rate, action_noise_avg_reward, action_noise_episode_rewards = evaluate_with_action_noise(
            policy, env, args.episodes, args.noise_std
        )
        
        # Compute composite score: 70% standard + 30% noisy
        composite_score = 0.7 * success_rate + 0.3 * action_noise_success_rate
        
        print(f"Action Noise Success Rate: {action_noise_success_rate:.2f}%")
        print(f"Action Noise Average Reward: {action_noise_avg_reward:.2f}")
        print(f"Composite Score: {composite_score:.2f}%")

    env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Total Episodes: {args.episodes}")
    print(f"Standard Evaluation:")
    print(f"  Successes: {successes}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Average Reward: {avg_reward:.2f}")
    if args.action_noise_test:
        print(f"Action Noise Evaluation:")
        print(f"  Success Rate: {action_noise_success_rate:.2f}%")
        print(f"  Average Reward: {action_noise_avg_reward:.2f}")
        print(f"Composite Score: {composite_score:.2f}%")
    print(f"Algorithm: BC (Behavioral Cloning)")
    print(f"Environment: {env_name}")
    print("--------------------------\n")
    
    # Save results if output directory is specified
    if args.output_dir:
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Try to extract hyperparameters from the model
        try:
            # Get model info - BC models are saved as ActorCriticPolicy
            model_info = {}
            if hasattr(policy, 'net_arch'):
                model_info['net_arch'] = policy.net_arch
            if hasattr(policy, 'lr_schedule'):
                try:
                    model_info['learning_rate'] = float(policy.lr_schedule(1.0))
                except:
                    model_info['learning_rate'] = 'unknown'
        except:
            model_info = {}
        
        # Environment-specific hyperparameters used in training
        if env_name == "PegTransfer-v0":
            training_hyperparams = {
                "learning_rate": 5e-5,
                "net_arch": [128, 128],
                "n_epochs": 25,
                "weight_decay": 1e-3,
                "batch_size": 64
            }
        elif env_name == "NeedleReach-v0":
            training_hyperparams = {
                "learning_rate": 1e-4,
                "net_arch": [256, 256],
                "n_epochs": 200,
                "weight_decay": 1e-5,
                "batch_size": 64
            }
        else:
            training_hyperparams = {
                "learning_rate": 5e-5,
                "net_arch": [128, 128],
                "n_epochs": 50,
                "weight_decay": 5e-4,
                "batch_size": 64
            }
        
        results = {
            "model_path": args.model,
            "environment": env_name,
            "algorithm": "BC",
            "total_episodes": args.episodes,
            "successes": successes,
            "success_rate": success_rate,
            "average_reward": avg_reward,
            "episode_rewards": episode_rewards,
            "composite_score": composite_score,
            "hyperparameters": {
                "training_hyperparams": training_hyperparams,
                "model_info": model_info,
                "reward_type": "sparse",
                "observation_space": "flattened_dict"
            }
        }
        
        # Add action noise test results if performed
        if args.action_noise_test:
            results["action_noise_test"] = {
                "enabled": True,
                "noise_std": args.noise_std,
                "success_rate": action_noise_success_rate,
                "average_reward": action_noise_avg_reward,
                "episode_rewards": action_noise_episode_rewards
            }
        else:
            results["action_noise_test"] = {
                "enabled": False
            }
        
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        print(f"Hyperparameters included: {list(training_hyperparams.keys())}")

if __name__ == "__main__":
    main()
