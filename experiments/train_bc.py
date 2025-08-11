import os
import pickle
import gymnasium as gym
import numpy as np
import argparse
from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import logger as imitation_logger
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import time

import dvrk_gym  # Import to register the environment


def train_bc_agent(env_name, expert_data_path, model_save_path=None, log_dir=None):
    """
    Trains a Behavioral Cloning (BC) agent.

    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.npz file).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
    """
    print("--- Behavioral Cloning Training ---")
    
    # --- 1. Load Expert Data ---
    print(f"Loading expert data from: {expert_data_path}")
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)

    # FINAL, VERIFIED SOLUTION:
    # The `imitation` library has issues with Dict observation spaces.
    # The most robust solution is to flatten the observations at the data-loading
    # stage, before they are ever passed to the library. This avoids all bugs.
    print("Flattening Dict observations into a single array...")
    
    all_obs = []
    all_next_obs = []
    all_acts = []
    all_dones = []

    for traj in trajectories:
        obs_soa = traj["obs"]
        num_transitions = len(traj["acts"])

        # Flatten each observation dictionary into a single numpy array
        for i in range(num_transitions):
            flat_obs = np.concatenate([
                obs_soa['observation'][i],
                obs_soa['achieved_goal'][i],
                obs_soa['desired_goal'][i]
            ])
            all_obs.append(flat_obs)
            
            flat_next_obs = np.concatenate([
                obs_soa['observation'][i+1],
                obs_soa['achieved_goal'][i+1],
                obs_soa['desired_goal'][i+1]
            ])
            all_next_obs.append(flat_next_obs)

        all_acts.extend(traj["acts"])
        dones = [False] * (num_transitions - 1) + [True]
        all_dones.extend(dones)


    # Convert lists to a single large numpy array
    all_obs = np.array(all_obs)
    all_next_obs = np.array(all_next_obs)
    all_acts = np.array(all_acts)
    all_dones = np.array(all_dones)

    # The imitation library expects data in a specific format (Transitions).
    transitions = types.Transitions(
        obs=all_obs,
        acts=all_acts,
        next_obs=all_next_obs,
        dones=all_dones,
        infos=np.array([{} for _ in range(len(all_obs))]),
    )
    print(f"Data flattened and converted to Transitions format: {len(transitions)} samples.")

    # --- 2. Setup Environment ---
    print(f"Initializing environment: {env_name}")
    venv = DummyVecEnv([lambda: gym.make(env_name)])

    # --- 3. Configure Logging ---
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # The imitation logger API seems to have changed and is simpler now.
        imitation_logger.configure(folder=log_dir)
        # The SB3 logger still accepts format_strings.
        sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
        print(f"Logging configured at: {log_dir}")
    else:
        print("Logging disabled - no output directory specified")

    # --- 4. Setup BC Trainer ---
    # Since we flattened the observations, we must manually create a Box observation
    # space that matches our flattened data, and use that to initialize the policy.
    from gymnasium.spaces import Box
    from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy

    flat_obs_space = Box(
        low=-np.inf, high=np.inf, shape=all_obs.shape[1:], dtype=np.float32
    )

    # Environment-specific hyperparameters
    if env_name == "PegTransfer-v0":
        # PegTransfer: Aggressive overfitting prevention
        learning_rate = 5e-5  # Slightly higher for faster convergence
        net_arch = [128, 128]  # Keep smaller network
        n_epochs = 25  # Much fewer epochs! Stop before overfitting
        weight_decay = 1e-3  # Stronger L2 regularization (10x stronger)
    elif env_name == "NeedleReach-v0":
        # NeedleReach: Simpler task, can handle higher LR
        learning_rate = 1e-4
        net_arch = [256, 256]
        n_epochs = 200
        weight_decay = 1e-5
    else:
        # Default values
        learning_rate = 5e-5
        net_arch = [128, 128]
        n_epochs = 50
        weight_decay = 5e-4
    
    policy = MlpPolicy(
        observation_space=flat_obs_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: learning_rate,
        net_arch=net_arch,
        optimizer_kwargs={
            "weight_decay": weight_decay,  # Add L2 regularization
            "eps": 1e-8,
        },
    )

    bc_trainer = bc.BC(
        observation_space=flat_obs_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=np.random.default_rng(),
        batch_size=64,  # Slightly larger batch size for stability
    )
    print("BC trainer configured.")

    # --- 5. Train the Agent ---
    print(f"Starting training for {env_name}...")
    print(f"Hyperparameters: LR={learning_rate}, epochs={n_epochs}, net_arch={net_arch}, weight_decay={weight_decay}")
    
    # Training with monitoring for overfitting prevention
    try:
        print(f"Training with overfitting prevention...")
        print(f"Watch for: l2_norm should stay < 500, loss should stay positive")
        
        bc_trainer.train(n_epochs=n_epochs, log_interval=5)  # More frequent logging for monitoring
        print("Training complete.")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        # Continue to save the model even if training had issues

    # --- 6. Save the Policy ---
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        bc_trainer.policy.save(model_save_path)
        print(f"Trained policy saved to: {model_save_path}")
    else:
        print("Model not saved - no output directory specified")

    venv.close()

if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train BC agent on dVRK environments")
    parser.add_argument("--env", required=True,
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for logs and model (if not specified, no files are saved)")
    
    args = parser.parse_args()
    
    # Auto-detect expert data path based on environment
    if args.env == "NeedleReach-v0":
        expert_data_path = os.path.join("data", "expert_data_needle_reach.pkl")
    elif args.env == "PegTransfer-v0":
        expert_data_path = os.path.join("data", "expert_data_peg_transfer.pkl")
    
    # Set up output paths only if output_dir is specified
    if args.output_dir:
        env_suffix = args.env.lower().replace("-v0", "").replace("reach", "_reach").replace("transfer", "_transfer")
        experiment_name = f"bc_{env_suffix}_{int(time.time())}"
        log_dir = os.path.join(args.output_dir, "logs")
        model_save_path = os.path.join(args.output_dir, f"{experiment_name}.zip")
        print(f"Output directory: {args.output_dir}")
    else:
        log_dir = None
        model_save_path = None
        print("No output directory specified - running without saving files")
    
    print(f"Training BC on {args.env}")
    print(f"Expert data: {expert_data_path}")

    train_bc_agent(
        env_name=args.env,
        expert_data_path=expert_data_path,
        model_save_path=model_save_path,
        log_dir=log_dir,
    )
