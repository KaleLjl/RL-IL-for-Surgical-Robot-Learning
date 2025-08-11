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
    parser.add_argument("--env", required=True, 
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for logs and model (if not specified, no files are saved)")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering during training (default: disabled)")
    
    args = parser.parse_args()
    
    print(f"Training PPO on {args.env}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print("No output directory - running without saving files")
    
    # Environment-specific optimal parameters
    if args.env == "NeedleReach-v0":
        timesteps = 100000
        checkpoint_freq = 10000
        learning_rate = 3e-4
        n_steps = 2048
        batch_size = 64
    elif args.env == "PegTransfer-v0":
        timesteps = 300000
        checkpoint_freq = 20000
        learning_rate = 1e-4
        n_steps = 4096
        batch_size = 256
    
    print(f"Optimized parameters:")
    print(f"  Timesteps: {timesteps}")
    print(f"  Checkpoint freq: {checkpoint_freq}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  N steps: {n_steps}")
    print(f"  Batch size: {batch_size}")
    
    # --- Configuration ---
    env_name = args.env
    
    # --- Path Configuration ---
    if args.output_dir:
        env_suffix = env_name.lower().replace("-v0", "").replace("reach", "_reach").replace("transfer", "_transfer")
        experiment_name = f"ppo_{env_suffix}_{int(time.time())}"
        
        # Set up paths
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
        analysis_dir = os.path.join(args.output_dir, "analysis")
        final_model_path = os.path.join(args.output_dir, f"{experiment_name}.zip")
        
        # Create all necessary directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
    else:
        checkpoint_dir = None
        tensorboard_log_dir = None
        analysis_dir = None
        final_model_path = None

    # --- Environment Setup ---
    # Create the Gym environment
    # Rendering is disabled by default for better performance and headless compatibility
    # For pure RL, we use the dense reward function.
    print("Creating environment with DENSE reward for pure RL training.")
    render_mode = 'human' if args.render else None
    env = gym.make(env_name, render_mode=render_mode, use_dense_reward=True)
    if args.render:
        print("Rendering enabled - training will be visualized")
    else:
        print("Rendering disabled - training will run faster")
    # Apply the wrapper to flatten the dictionary observation space
    print("Applying FlattenDictObsWrapper to the environment.")
    env = FlattenDictObsWrapper(env)
    
    # --- Agent and Training Setup ---
    callback_list = None
    if args.output_dir:
        # Checkpoint callback to save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        
        # Training analysis callback for debugging and monitoring
        analysis_callback = TrainingAnalysisCallback(
            log_interval=1000,
            analysis_dir=analysis_dir,
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
    if final_model_path:
        print(f"Training complete. Saving final model to {final_model_path}")
        model.save(final_model_path)
    else:
        print("Training complete. Model not saved - no output directory specified")

    # --- Close Environment ---
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()
