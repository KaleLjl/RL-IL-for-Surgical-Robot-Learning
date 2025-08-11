import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
from dvrk_gym.utils.callbacks import TrainingAnalysisCallback
import os
import time
import argparse

def main():
    """
    Main function to train a PPO agent on the specified environment.
    """
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train PPO agent on dVRK environments")
    parser.add_argument("--env", default="NeedleReach-v0", 
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--timesteps", type=int, 
                       help="Total training timesteps (auto-selected if not provided)")
    parser.add_argument("--checkpoint-freq", type=int,
                       help="Checkpoint save frequency (auto-selected if not provided)")
    parser.add_argument("--learning-rate", type=float,
                       help="Learning rate (auto-selected if not provided)")
    parser.add_argument("--n-steps", type=int,
                       help="Number of steps per rollout (auto-selected if not provided)")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size (auto-selected if not provided)")
    parser.add_argument("--log-interval", type=int, default=1000,
                       help="Steps between analysis logging (default: 1000)")
    
    args = parser.parse_args()
    
    # Environment-specific optimal parameters
    if args.env == "NeedleReach-v0":
        defaults = {
            "timesteps": 100000,
            "checkpoint_freq": 10000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
        }
    elif args.env == "PegTransfer-v0":
        defaults = {
            "timesteps": 300000,
            "checkpoint_freq": 20000,
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 256,
        }
    
    # Use provided arguments or fall back to environment defaults
    timesteps = args.timesteps or defaults["timesteps"]
    checkpoint_freq = args.checkpoint_freq or defaults["checkpoint_freq"]
    learning_rate = args.learning_rate or defaults["learning_rate"]
    n_steps = args.n_steps or defaults["n_steps"]
    batch_size = args.batch_size or defaults["batch_size"]
    
    print(f"Training {args.env} with optimized parameters:")
    print(f"  Timesteps: {timesteps}")
    print(f"  Checkpoint freq: {checkpoint_freq}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  N steps: {n_steps}")
    print(f"  Batch size: {batch_size}")
    
    # --- Configuration ---
    env_name = args.env
    
    # --- Path Configuration ---
    # 1. Root directory for all experiments
    env_suffix = env_name.lower().replace("-v0", "").replace("reach", "_reach").replace("transfer", "_transfer")
    experiment_name = f"ppo_{env_suffix}_{int(time.time())}"
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
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Training analysis callback for debugging and monitoring
    analysis_callback = TrainingAnalysisCallback(
        log_interval=args.log_interval,
        analysis_dir=os.path.join(run_dir, "analysis"),
        algorithm="ppo",
        verbose=1
    )
    
    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, analysis_callback])

    # Initialize the PPO agent with environment-specific parameters
    # Using MlpPolicy because the observation space is now a flattened Box
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1, 
        tensorboard_log=tensorboard_log_dir
    )

    # --- Start Training ---
    print(f"Starting training on {env_name}...")
    model.learn(
        total_timesteps=timesteps, 
        callback=callback_list
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
