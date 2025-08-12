import os
import pickle
import gymnasium as gym
import numpy as np
import argparse
import dvrk_gym  # Import to register the environment

def generate_expert_data(env_name, num_episodes=100, data_path="expert_data.pkl"):
    """
    Generates expert demonstration data and saves it as a list of trajectories.

    Args:
        env_name (str): The name of the Gymnasium environment.
        num_episodes (int): The number of expert trajectories to generate.
        data_path (str): The path to save the pickled expert data file.
    """
    print(f"Initializing environment: {env_name}")
    env = gym.make(env_name, render_mode='human')

    all_trajectories = []
    total_transitions = 0

    print(f"Starting data generation for {num_episodes} episodes...")

    for i in range(num_episodes):
        print(f"--- Episode {i + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        done = False
        
        episode_obs = [obs]
        episode_actions = []
        
        while not done:
            action = env.get_oracle_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_obs.append(obs)
            episode_actions.append(action)

            if done:
                print(f"Episode finished. Success: {info.get('is_success', False)}")
        
        # Convert lists to numpy arrays
        # Observations need to be converted from a list of dicts to a dict of lists, then to a dict of arrays
        obs_keys = episode_obs[0].keys()
        final_obs = {key: np.array([o[key] for o in episode_obs]) for key in obs_keys}

        # --- Verification Step ---
        # For each key in the observation dictionary, its length must be len(actions) + 1
        num_actions = len(episode_actions)
        for key, value in final_obs.items():
            assert len(value) == num_actions + 1, \
                f"Mismatch in trajectory lengths for key '{key}': obs_len={len(value)}, acts_len={num_actions}"

        # Create trajectory dictionary
        trajectory = {
            "obs": final_obs,
            "acts": np.array(episode_actions),
        }
        all_trajectories.append(trajectory)
        total_transitions += len(episode_actions)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Save the data using pickle
    with open(data_path, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print(f"\nExpert data saved successfully to {data_path}")
    print(f"Total trajectories collected: {len(all_trajectories)}")
    print(f"Total transitions collected: {total_transitions}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert data for NeedleReach task")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to generate (default: 50)")
    parser.add_argument("--data_path", type=str, default="data/expert_data_needle_reach.pkl",
                        help="Path to save the expert data (default: data/expert_data_needle_reach.pkl)")
    
    args = parser.parse_args()
    
    ENV_NAME = "NeedleReach-v0"
    generate_expert_data(ENV_NAME, num_episodes=args.num_episodes, data_path=args.data_path)
