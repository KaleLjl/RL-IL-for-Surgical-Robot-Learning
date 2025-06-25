import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import pathlib

import dvrk_gym

# --- Parameters ---
ENV_NAME = "NeedleReach-v0"
MODEL_PATH = "models/bc_needle_reach.zip"
NUM_EPISODES = 100

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
    print(f"Loading environment: {ENV_NAME}")
    env = gym.make(ENV_NAME, render_mode="human")

    print(f"Loading model from: {MODEL_PATH}")
    # The policy was saved as a standard SB3 ActorCriticPolicy,
    # so we should load it using the same class.
    # The `imitation.reconstruct_policy` has different expectations.
    policy = ActorCriticPolicy.load(MODEL_PATH)

    successes = 0
    total_rewards = 0

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            # Flatten the observation to match the policy's input format
            flat_obs = flatten_obs(obs)
            
            action, _ = policy.predict(flat_obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            episode_reward += reward

        if info.get('is_success', False):
            successes += 1
        
        total_rewards += episode_reward
        print(f"Episode {episode + 1}/{NUM_EPISODES} - Reward: {episode_reward:.2f} - Success: {info.get('is_success', False)}")

    env.close()

    success_rate = (successes / NUM_EPISODES) * 100
    avg_reward = total_rewards / NUM_EPISODES

    print("\n--- Evaluation Summary ---")
    print(f"Total Episodes: {NUM_EPISODES}")
    print(f"Successes: {successes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()
