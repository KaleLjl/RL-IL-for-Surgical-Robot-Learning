import argparse
import sys
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Import dvrk_gym to register the environment
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def evaluate_with_action_noise(model, env, episodes=50, noise_std=0.1):
    """
    Evaluate PPO policy with Gaussian noise added to actions.
    This tests robustness to distribution shift.
    
    Args:
        model: Trained PPO model
        env: Environment to evaluate in
        episodes: Number of episodes to evaluate
        noise_std: Standard deviation of Gaussian noise to add to actions
        
    Returns:
        Tuple of (success_rate, average_reward, episode_rewards)
    """
    episode_rewards = []
    episode_successes = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not done and not truncated:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Add Gaussian noise to action
            action_noise = np.random.normal(0, noise_std, action.shape)
            noisy_action = action + action_noise
            
            # Clip action to valid range
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
            
            obs, reward, done, truncated, info = env.step(noisy_action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        if 'is_success' in info:
            episode_successes.append(float(info['is_success']))
        else:
            episode_successes.append(0.0)
    
    success_rate = np.mean(episode_successes) * 100
    avg_reward = np.mean(episode_rewards)
    
    return success_rate, avg_reward, episode_rewards

def evaluate_ppo_agent(env_name, model_path, n_episodes, output_dir=None, action_noise_test=False, noise_std=0.1):
    """
    Evaluates a trained PPO agent in the specified environment.

    Args:
        env_name (str): The name of the Gymnasium environment.
        model_path (str): Path to the trained model (.zip file).
        n_episodes (int): The number of episodes to run for evaluation.
        output_dir (str): Output directory for results (optional).
    """
    print(f"--- Starting PPO Evaluation ---")
    print(f"Environment: {env_name}")
    print(f"Model Path: {model_path}")
    print(f"Number of Episodes: {n_episodes}")

    # --- 1. Create Environment ---
    # PPO models use dense rewards and flattened observations
    render_mode = None if '--no-render' in sys.argv else 'human'
    env_kwargs = {'render_mode': render_mode, 'use_dense_reward': True}
    
    env = gym.make(env_name, **env_kwargs)

    # Apply flattened wrapper (PPO models were trained with this)
    print("Applying FlattenDictObsWrapper to the environment.")
    env = FlattenDictObsWrapper(env)
    
    print("Environment created.")

    # --- 2. Load Model ---
    try:
        model = PPO.load(model_path)
        print("PPO model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # --- 3. Evaluation Loop ---
    episode_rewards = []
    episode_successes = []

    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        print(f"\n--- Episode {i + 1}/{n_episodes} ---")
        
        while not done and not truncated:
            # Use deterministic=True for evaluation to get the best action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Render the environment if render mode is enabled
            if render_mode:
                env.render()

        print(f"Episode finished.")
        print(f"Total Reward: {total_reward}")
        
        # Check for success info at the end of the episode
        if 'is_success' in info:
            success = info['is_success']
            print(f"Success: {success}")
            episode_successes.append(float(success))
        else:
            print("Success status not available for this episode.")

        episode_rewards.append(total_reward)

    # Action noise robustness test (if enabled)
    action_noise_success_rate = None
    action_noise_avg_reward = None
    action_noise_episode_rewards = []
    composite_score = np.mean(episode_successes) * 100 if episode_successes else 0.0  # Default to standard score
    
    if action_noise_test:
        print(f"\n--- Running Action Noise Robustness Test (noise_std={noise_std}) ---")
        action_noise_success_rate, action_noise_avg_reward, action_noise_episode_rewards = evaluate_with_action_noise(
            model, env, n_episodes, noise_std
        )
        
        # Compute composite score: 70% standard + 30% noisy
        standard_success_rate = np.mean(episode_successes) * 100 if episode_successes else 0.0
        composite_score = 0.7 * standard_success_rate + 0.3 * action_noise_success_rate
        
        print(f"Action Noise Success Rate: {action_noise_success_rate:.2f}%")
        print(f"Action Noise Average Reward: {action_noise_avg_reward:.2f}")
        print(f"Composite Score: {composite_score:.2f}%")

    # --- 4. Print Final Statistics ---
    print("\n" + "="*50)
    print("           PPO EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print("-"*50)
    
    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Reward Range: [{min_reward:.2f}, {max_reward:.2f}]")
    
    if episode_successes:
        success_rate = np.mean(episode_successes) * 100
        num_successes = int(np.sum(episode_successes))
        print(f"Standard Success Rate: {success_rate:.2f}% ({num_successes}/{n_episodes})")
    else:
        print("Success Rate: N/A (no success info available)")
        success_rate = 0.0
        num_successes = 0
    
    if action_noise_test:
        print(f"Action Noise Success Rate: {action_noise_success_rate:.2f}%")
        print(f"Action Noise Average Reward: {action_noise_avg_reward:.2f}")
        print(f"Composite Score: {composite_score:.2f}%")
    
    print("="*50)
    
    # Save results if output directory is specified
    if output_dir:
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to extract hyperparameters from the model
        try:
            model_hyperparams = {}
            if hasattr(model, 'learning_rate'):
                model_hyperparams['learning_rate'] = float(model.learning_rate)
            if hasattr(model, 'n_steps'):
                model_hyperparams['n_steps'] = int(model.n_steps)
            if hasattr(model, 'batch_size'):
                model_hyperparams['batch_size'] = int(model.batch_size)
            if hasattr(model, 'gamma'):
                model_hyperparams['gamma'] = float(model.gamma)
            if hasattr(model, 'gae_lambda'):
                model_hyperparams['gae_lambda'] = float(model.gae_lambda)
            if hasattr(model, 'clip_range'):
                model_hyperparams['clip_range'] = float(model.clip_range)
        except:
            model_hyperparams = {}
        
        results = {
            "model_path": model_path,
            "environment": env_name,
            "algorithm": "PPO",
            "total_episodes": n_episodes,
            "average_reward": float(mean_reward) if episode_rewards else 0.0,
            "std_reward": float(std_reward) if episode_rewards else 0.0,
            "min_reward": float(min_reward) if episode_rewards else 0.0,
            "max_reward": float(max_reward) if episode_rewards else 0.0,
            "success_rate": float(success_rate),
            "num_successes": int(num_successes) if 'num_successes' in locals() else 0,
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_successes": [float(s) for s in episode_successes],
            "composite_score": float(composite_score),
            "hyperparameters": {
                "model_hyperparams": model_hyperparams,
                "reward_type": "dense",
                "observation_space": "flattened_dict"
            }
        }
        
        # Add action noise test results if performed
        if action_noise_test:
            results["action_noise_test"] = {
                "enabled": True,
                "noise_std": noise_std,
                "success_rate": action_noise_success_rate,
                "average_reward": action_noise_avg_reward,
                "episode_rewards": action_noise_episode_rewards
            }
        else:
            results["action_noise_test"] = {
                "enabled": False
            }
        
        results_file = os.path.join(output_dir, "ppo_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    # --- 5. Close Environment ---
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained PPO model (.zip file)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (if not specified, only console output)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="The number of episodes to run for evaluation."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering for faster evaluation."
    )
    parser.add_argument(
        "--action-noise-test",
        action="store_true",
        help="Enable action noise robustness testing."
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Standard deviation of action noise for robustness testing (default: 0.1)."
    )

    args = parser.parse_args()
    
    # Auto-detect environment from model path
    if "needle_reach" in args.model.lower():
        env_name = "NeedleReach-v0"
    elif "peg_transfer" in args.model.lower():
        env_name = "PegTransfer-v0"
    else:
        env_name = "NeedleReach-v0"
        print(f"Warning: Could not detect environment from model path, using {env_name}")
    
    print(f"Auto-detected environment: {env_name}")
    print(f"Using dense rewards (appropriate for PPO)")
    print(f"Using flattened observations")

    evaluate_ppo_agent(
        env_name=env_name,
        model_path=args.model,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
        action_noise_test=args.action_noise_test,
        noise_std=args.noise_std
    )