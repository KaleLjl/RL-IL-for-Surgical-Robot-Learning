import os
import pickle
import time
import gymnasium as gym
import numpy as np
import torch
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from stable_baselines3.common.callbacks import CallbackList
from imitation.data import types
from imitation.util import logger as imitation_logger
from stable_baselines3.common import logger as sb3_logger

# Import our custom algorithm and wrapper
from dvrk_gym.algorithms.ppo_bc import PPOWithBCLoss
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
from dvrk_gym.utils.callbacks import TrainingAnalysisCallback
import dvrk_gym  # Import to register the environment

def train_dapg_agent(env_name, expert_data_path, model_save_path, log_dir, timesteps=300000, bc_weight=0.05, log_interval=1000, bc_model_path=None):
    """
    Trains an agent using our custom PPOWithBCLoss algorithm, which
    forms the basis of our DAPG implementation.

    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.pkl file).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
        timesteps (int): Total training timesteps.
        bc_weight (float): BC loss weight.
        log_interval (int): Steps between analysis logging.
    """
    print("--- Custom DAPG Training ---")
    
    # --- 1. Load and Flatten Expert Data ---
    print(f"Loading expert data from: {expert_data_path}")
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)

    print("Flattening Dict observations into a single array...")
    all_obs, all_next_obs, all_acts, all_dones = [], [], [], []
    for traj in trajectories:
        obs_soa = traj["obs"]
        # The last observation has no corresponding action, so we ignore it.
        num_transitions = len(traj["acts"])

        # Flatten each observation dictionary into a single numpy array
        for i in range(num_transitions):
            flat_obs = np.concatenate([
                obs_soa['observation'][i],
                obs_soa['achieved_goal'][i],
                obs_soa['desired_goal'][i]
            ])
            all_obs.append(flat_obs)
            
            # We need next_obs and dones for the Transitions object.
            flat_next_obs = np.concatenate([
                obs_soa['observation'][i+1],
                obs_soa['achieved_goal'][i+1],
                obs_soa['desired_goal'][i+1]
            ])
            all_next_obs.append(flat_next_obs)

        all_acts.extend(traj["acts"])
        # Create a dones array that is True only at the very end of the trajectory.
        dones = [False] * (num_transitions - 1) + [True]
        all_dones.extend(dones)

    # Convert to a format that can be passed to our custom algorithm
    expert_demonstrations = types.Transitions(
        obs=np.array(all_obs),
        acts=np.array(all_acts),
        next_obs=np.array(all_next_obs),
        dones=np.array(all_dones),
        infos=np.array([{} for _ in range(len(all_obs))]),
    )
    print(f"Data flattened and converted to Transitions format: {len(expert_demonstrations)} samples.")

    # --- 2. Setup Vectorized and Flattened Environment ---
    print(f"Initializing environment: {env_name}")
    # Per reward-system-guidelines.md, DAPG should use a sparse reward.
    # The default is sparse, so we don't need to set any env_kwargs.
    # Enable GUI rendering for debugging.
    venv = make_vec_env(
        env_name,
        n_envs=1,
        wrapper_class=FlattenDictObsWrapper,
        env_kwargs={'render_mode': 'human'}
    )
    print("Environment created and wrapped with FlattenDictObsWrapper.")

    # --- 3. Configure Logging ---
    os.makedirs(log_dir, exist_ok=True)
    imitation_logger.configure(folder=log_dir, format_strs=["stdout", "tensorboard"])
    sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
    print(f"Logging configured at: {log_dir}")

    # --- 4. Setup Custom DAPG (PPOWithBCLoss) Trainer ---
    print("Initializing custom PPOWithBCLoss agent...")
    
    # Use the same network architecture as BC model to enable weight loading
    policy_kwargs = dict(
        net_arch=[256, 256],  # Same as BC model
    )
    
    model = PPOWithBCLoss(
        policy=MlpPolicy,
        env=venv,
        expert_demonstrations=expert_demonstrations,
        bc_loss_weight=bc_weight,
        bc_batch_size=256,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    
    # --- 4.5. Load BC model weights if provided (Standard DAPG approach) ---
    if bc_model_path and os.path.exists(bc_model_path):
        print(f"Loading BC model weights from: {bc_model_path}")
        try:
            # BC model was saved using imitation library's policy.save() method
            # We need to load it as a policy directly, not as a stable-baselines3 model
            from gymnasium.spaces import Box
            
            # Create the same observation space as used in BC training
            flat_obs_space = Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32  # PegTransfer obs size
            )
            
            # Load BC policy directly
            bc_policy = MlpPolicy.load(bc_model_path)
            
            # Copy BC policy weights to DAPG model
            model.policy.load_state_dict(bc_policy.state_dict())
            print("BC model weights loaded successfully.")
        except Exception as e:
            print(f"Failed to load BC model: {e}")
            print("Starting from random initialization.")
    else:
        print("No BC model provided - starting from random initialization.")
    
    print("Custom agent configured.")

    # --- 4.5. Setup Training Analysis Callback ---
    analysis_callback = TrainingAnalysisCallback(
        log_interval=log_interval,
        analysis_dir=os.path.join(log_dir, "analysis"),
        algorithm="dapg",
        verbose=1
    )
    print("Training analysis callback configured.")

    # --- 5. Train the Agent ---
    print("Starting training...")
    model.learn(total_timesteps=timesteps, callback=analysis_callback)
    print("Training complete.")

    # --- 6. Save the Model ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}")

    venv.close()

if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train DAPG agent on dVRK environments")
    parser.add_argument("--env", default="NeedleReach-v0",
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--expert-data", 
                       help="Path to expert data file (auto-detected if not provided)")
    parser.add_argument("--bc-model", 
                       help="Path to BC model for initialization (auto-detected if not provided)")
    parser.add_argument("--timesteps", type=int,
                       help="Total training timesteps (auto-selected if not provided)")
    parser.add_argument("--bc-weight", type=float,
                       help="BC loss weight (auto-selected if not provided)")
    parser.add_argument("--log-interval", type=int, default=1000,
                       help="Steps between analysis logging (default: 1000)")
    
    args = parser.parse_args()
    
    # Environment-specific optimal parameters for DAPG
    if args.env == "NeedleReach-v0":
        defaults = {
            "timesteps": 300000,
            "bc_weight": 0.05,
        }
    elif args.env == "PegTransfer-v0":
        defaults = {
            "timesteps": 500000,
            "bc_weight": 0.1,  # Higher BC weight for complex task
        }
    
    # Use provided arguments or fall back to environment defaults
    timesteps = args.timesteps or defaults["timesteps"]
    bc_weight = args.bc_weight or defaults["bc_weight"]
    
    print(f"Training DAPG on {args.env} with optimized parameters:")
    print(f"  Timesteps: {timesteps}")
    print(f"  BC weight: {bc_weight}")
    
    # Auto-detect expert data path if not provided
    if args.expert_data is None:
        if args.env == "NeedleReach-v0":
            args.expert_data = os.path.join("data", "expert_data_needle_reach.pkl")
        elif args.env == "PegTransfer-v0":
            args.expert_data = os.path.join("data", "expert_data_peg_transfer.pkl")
    
    # Auto-detect BC model path if not provided
    if args.bc_model is None:
        import glob
        if args.env == "NeedleReach-v0":
            pattern = os.path.join("models", "bc_needle_reach_*.zip")
        elif args.env == "PegTransfer-v0":
            pattern = os.path.join("models", "bc_peg_transfer_*.zip")
        
        bc_models = glob.glob(pattern)
        if bc_models:
            args.bc_model = sorted(bc_models)[-1]  # Get the latest model
            print(f"Auto-detected BC model: {args.bc_model}")
        else:
            print(f"No BC model found for {args.env} - will start from random initialization")
            args.bc_model = None
    
    # Create a unique directory for this experiment
    env_suffix = args.env.lower().replace("-v0", "").replace("reach", "_reach").replace("transfer", "_transfer")
    experiment_name = f"dapg_{env_suffix}_{int(time.time())}"
    log_dir = os.path.join("logs", experiment_name)
    model_dir = "models"
    
    # Save model in the models/ dir
    model_save_path = os.path.join(model_dir, f"{experiment_name}.zip")

    train_dapg_agent(
        env_name=args.env,
        expert_data_path=args.expert_data,
        model_save_path=model_save_path,
        log_dir=log_dir,
        timesteps=timesteps,
        bc_weight=bc_weight,
        log_interval=args.log_interval,
        bc_model_path=args.bc_model,
    )
