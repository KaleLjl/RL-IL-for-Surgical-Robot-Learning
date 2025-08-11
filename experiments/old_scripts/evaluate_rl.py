import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Import dvrk_gym to register the environment
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

def evaluate_agent(env_name, model_path, n_episodes, is_flattened, use_dense_reward):
    """
    Evaluates a trained agent in the specified environment.

    Args:
        env_name (str): The name of the Gymnasium environment.
        model_path (str): Path to the trained model (.zip file).
        n_episodes (int): The number of episodes to run for evaluation.
        is_flattened (bool): Whether the model was trained on flattened observations.
        use_dense_reward (bool): Whether to use the dense reward for evaluation.
    """
    print(f"--- Starting Evaluation ---")
    print(f"Environment: {env_name}")
    print(f"Model Path: {model_path}")
    print(f"Number of Episodes: {n_episodes}")
    print(f"Using Flattened Observations: {is_flattened}")
    print(f"Using Dense Reward: {use_dense_reward}")

    # --- 1. Create Environment ---
    # Build keyword arguments for environment creation
    env_kwargs = {'render_mode': 'human'}
    if use_dense_reward:
        env_kwargs['use_dense_reward'] = True
    
    env = gym.make(env_name, **env_kwargs)

    # Use a wrapper if the policy was trained on flattened observations
    if is_flattened:
        print("Applying FlattenDictObsWrapper to the environment.")
        env = FlattenDictObsWrapper(env)
    
    print("Environment created.")

    # --- 2. Load Model ---
    # PPO.load is versatile and can load policies from BC and DAPG as well,
    # as long as they share the Actor-Critic structure.
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully.")
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

            # Render the environment
            # Note: The environment's render method is what actually
            # updates the PyBullet GUI.
            env.render()

        print(f"Episode finished.")
        print(f"Total Reward: {total_reward}")
        
        # Check for success info at the end of the episode
        if 'is_success' in info:
            success = info['is_success']
            print(f"Success: {success}")
            episode_successes.append(float(success))
        else:
            # If 'is_success' is not in info, we cannot calculate success rate.
            # This can happen if the episode is truncated before reaching a terminal state.
            print("Success status not available for this episode.")


        episode_rewards.append(total_reward)

    # --- 4. Print Final Statistics ---
    print("\n--- Evaluation Complete ---")
    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Average Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if episode_successes:
        success_rate = np.mean(episode_successes) * 100
        print(f"Success Rate: {success_rate:.2f}%")
    else:
        print("Could not calculate success rate as 'is_success' info was not available.")


    # --- 5. Close Environment ---
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)."
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="NeedleReach-v0",
        choices=["NeedleReach-v0", "PegTransfer-v0"],
        help="The Gymnasium environment name to run the evaluation on."
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="The number of episodes to run for evaluation."
    )
    # This flag is crucial because our BC/DAPG models were trained on flattened
    # observations, while the pure RL model was trained on Dict observations.
    # The environment wrapper needs to match the policy's expectation.
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Add this flag if the model was trained on flattened observations."
    )
    parser.add_argument(
        "--dense-reward",
        action="store_true",
        help="Add this flag to use the dense reward function during evaluation."
    )

    args = parser.parse_args()

    evaluate_agent(
        env_name=args.env_name,
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        is_flattened=args.flatten,
        use_dense_reward=args.dense_reward
    )
