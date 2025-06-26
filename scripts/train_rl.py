import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
import os
import time

def main():
    """
    Main function to train a PPO agent on the NeedleReach-v0 environment.
    """
    # --- Configuration ---
    env_name = "NeedleReach-v0"
    
    # --- Path Configuration ---
    # 1. Root directory for all experiments
    experiment_name = f"ppo_needle_reach_{int(time.time())}"
    experiment_dir = os.path.join("logs", experiment_name)

    # 2. Directory for this specific training run (run_1, run_2, etc.)
    run_id = 1
    while os.path.exists(os.path.join(experiment_dir, f"run_{run_id}")):
        run_id += 1
    run_dir = os.path.join(experiment_dir, f"run_{run_id}")

    # 3. Subdirectories for checkpoints and TensorBoard logs
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_log_dir = os.path.join(run_dir, "tensorboard_logs")
    
    # 4. Path for the final model in the top-level 'models' directory
    model_dir = "models"
    final_model_path = os.path.join(model_dir, f"{experiment_name}.zip")

    # Create all necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Environment Setup ---
    # Create the Gym environment
    # The 'render_mode' is set to 'human' to visualize the training process.
    # This requires a display connection (e.g., X11 forwarding).
    # For pure RL, we use the dense reward function.
    print("Creating environment with DENSE reward for pure RL training.")
    env = gym.make(env_name, render_mode='human', use_dense_reward=True)
    # Apply the wrapper to flatten the dictionary observation space
    print("Applying FlattenDictObsWrapper to the environment.")
    env = FlattenDictObsWrapper(env)
    
    # --- Agent and Training Setup ---
    # Checkpoint callback to save model periodically inside the run's checkpoint folder
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Initialize the PPO agent
    # Using MlpPolicy because the observation space is now a flattened Box
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=tensorboard_log_dir
    )

    # --- Start Training ---
    print(f"Starting training on {env_name}...")
    model.learn(
        total_timesteps=100000, 
        callback=checkpoint_callback
    )

    # --- Save Final Model ---
    # This saves the *last* model of the training run. The user should manually
    # select the best model from the checkpoints for the "hall of fame" in models/.
    print(f"Training complete. Saving final model to {final_model_path}")
    model.save(final_model_path)

    # --- Close Environment ---
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
