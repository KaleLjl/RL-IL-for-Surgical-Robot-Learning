import os
import gymnasium as gym
import numpy as np
import dvrk_gym  # Import to register the environment

def generate_expert_data(env_name, num_episodes=100, data_path="expert_data.npz"):
    """
    Generates expert demonstration data using the environment's oracle.

    Args:
        env_name (str): The name of the Gymnasium environment.
        num_episodes (int): The number of expert trajectories to generate.
        data_path (str): The path to save the expert data file.
    """
    print(f"Initializing environment: {env_name}")
    # Use 'human' render_mode to visualize the data collection process
    env = gym.make(env_name, render_mode='human')

    trajectories = {
        'actions': [],
        'obs': [],
        'rewards': [],
        'episode_returns': [],
        'episode_starts': [],
    }

    print(f"Starting data generation for {num_episodes} episodes...")

    for i in range(num_episodes):
        print(f"--- Episode {i + 1}/{num_episodes} ---")
        episode_obs, episode_actions, episode_rewards = [], [], []
        
        obs, info = env.reset()
        done = False
        
        trajectories['episode_starts'].append(True)
        
        while not done:
            action = env.get_oracle_action(obs)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)

            if done:
                print(f"Episode finished. Success: {info.get('is_success', False)}")

        # Append episode data
        trajectories['obs'].extend(episode_obs)
        trajectories['actions'].extend(episode_actions)
        trajectories['rewards'].extend(episode_rewards)
        trajectories['episode_returns'].append(np.sum(episode_rewards))
        
        # Mark the end of the episode
        # We need to add one more 'episode_starts' marker for the next episode
        # The imitation library expects len(episode_starts) == len(obs)
        # Since we don't add the final observation, we mark the start of the *next* theoretical step
        # as False, except for the very last one.
        for _ in range(len(episode_obs) - 1):
             trajectories['episode_starts'].append(False)


    # Convert lists to numpy arrays
    # The imitation library expects observations to be a dictionary of arrays
    # and actions to be a single array.
    
    # Process observations
    obs_dict = {}
    if len(trajectories['obs']) > 0:
        keys = trajectories['obs'][0].keys()
        for key in keys:
            obs_dict[key] = np.array([o[key] for o in trajectories['obs']])

    # Finalize data structure for saving
    expert_data = {
        'actions': np.array(trajectories['actions']),
        'obs': obs_dict,
        'rewards': np.array(trajectories['rewards']),
        'episode_returns': np.array(trajectories['episode_returns']),
        'episode_starts': np.array(trajectories['episode_starts'], dtype=bool),
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Save the data
    np.savez_compressed(data_path, **expert_data)
    print(f"\nExpert data saved successfully to {data_path}")
    print(f"Total transitions collected: {len(expert_data['actions'])}")

    env.close()

if __name__ == "__main__":
    ENV_NAME = "NeedleReach-v0"
    DATA_SAVE_PATH = os.path.join("data", "expert_data_needle_reach.npz")
    generate_expert_data(ENV_NAME, num_episodes=50, data_path=DATA_SAVE_PATH)
