import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import dvrk_gym
import os

def main():
    """
    Main function to train a PPO agent on the NeedleReach-v0 environment.
    """
    # --- Configuration ---
    env_name = "NeedleReach-v0"
    log_dir = "./logs/"
    model_save_path = os.path.join(log_dir, "ppo_needle_reach")
    
    os.makedirs(log_dir, exist_ok=True)

    # --- Environment Setup ---
    # Create the Gym environment
    # The 'render_mode' is set to 'human' to visualize the training process.
    # This requires a display connection (e.g., X11 forwarding).
    env = gym.make(env_name, render_mode='human')
    
    # --- Agent and Training Setup ---
    # Checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Initialize the PPO agent
    # Using MultiInputPolicy because the observation space is a dictionary
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir
    )

    # --- Start Training ---
    print(f"Starting training on {env_name}...")
    model.learn(
        total_timesteps=100000, 
        callback=checkpoint_callback
    )

    # --- Save Final Model ---
    print(f"Training complete. Saving final model to {model_save_path}.zip")
    model.save(model_save_path)

    # --- Close Environment ---
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
