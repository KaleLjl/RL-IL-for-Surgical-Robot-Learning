import os
import pickle
import gymnasium as gym
import numpy as np
import dvrk_gym  # Import to register the environment

def generate_expert_data_peg_transfer(num_episodes=100, data_path="data/expert_data_peg_transfer.pkl"):
    """
    Generates expert demonstration data for PegTransfer task and saves it as a list of trajectories.

    Args:
        num_episodes (int): The number of expert trajectories to generate.
        data_path (str): The path to save the pickled expert data file.
    """
    print("Initializing PegTransfer environment...")
    env = gym.make('PegTransfer-v0', render_mode='human')

    all_trajectories = []
    total_transitions = 0

    print(f"Starting PegTransfer expert data generation for {num_episodes} episodes...")

    for i in range(num_episodes):
        print(f"--- Episode {i + 1}/{num_episodes} ---")
        
        obs, info = env.reset()
        done = False
        
        episode_obs = [obs]
        episode_actions = []
        
        step_count = 0
        max_steps = 150  # PegTransfer episode limit
        
        while not done and step_count < max_steps:
            action = env.unwrapped.get_oracle_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_obs.append(obs)
            episode_actions.append(action)
            step_count += 1

            if done:
                success = info.get('is_success', False)
                print(f"Episode finished after {step_count} steps. Success: {success}")
        
        # Only keep successful episodes for PegTransfer
        if info.get('is_success', False):
            # Convert lists to numpy arrays
            obs_keys = episode_obs[0].keys()
            final_obs = {key: np.array([o[key] for o in episode_obs]) for key in obs_keys}

            # Verification step
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
            print(f"Successfully added trajectory with {len(episode_actions)} transitions")
        else:
            print("Episode failed - not adding to dataset")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Save the data using pickle
    with open(data_path, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print(f"\nPegTransfer expert data saved successfully to {data_path}")
    print(f"Total successful trajectories collected: {len(all_trajectories)}")
    print(f"Total transitions collected: {total_transitions}")

    env.close()

if __name__ == "__main__":
    generate_expert_data_peg_transfer(num_episodes=1000, data_path="data/expert_data_peg_transfer.pkl")