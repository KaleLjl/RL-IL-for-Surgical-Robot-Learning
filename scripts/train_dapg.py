import os
import pickle
import time
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from imitation.data import types
from imitation.util import logger as imitation_logger
from stable_baselines3.common import logger as sb3_logger

# Import our custom algorithm and wrapper
from dvrk_gym.algorithms.ppo_bc import PPOWithBCLoss
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
import dvrk_gym  # Import to register the environment

def train_dapg_agent(env_name, expert_data_path, model_save_path, log_dir):
    """
    Trains an agent using our custom PPOWithBCLoss algorithm, which
    forms the basis of our DAPG implementation.

    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.pkl file).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
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
        num_transitions = len(traj["acts"])

        # Flatten each observation dictionary into a single numpy array
        for i in range(num_transitions):
            flat_obs = np.concatenate([
                obs_soa['observation'][i],
                obs_soa['achieved_goal'][i],
                obs_soa['desired_goal'][i]
            ])
            all_obs.append(flat_obs)
            
            # We need next_obs and dones for the Transitions object, even if
            # the BC loss part of DAPG doesn't use them directly.
            flat_next_obs = np.concatenate([
                obs_soa['observation'][i+1],
                obs_soa['achieved_goal'][i+1],
                obs_soa['desired_goal'][i+1]
            ])
            all_next_obs.append(flat_next_obs)

        all_acts.extend(traj["acts"])
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
    # Use make_vec_env to create a vectorized environment and apply our custom wrapper.
    # This is the modern and correct way to handle environment wrapping.
    venv = make_vec_env(
        env_name,
        n_envs=1,
        wrapper_class=FlattenDictObsWrapper,
    )
    print("Environment created and wrapped with FlattenDictObsWrapper.")

    # --- 3. Configure Logging ---
    os.makedirs(log_dir, exist_ok=True)
    imitation_logger.configure(folder=log_dir, format_strs=["stdout", "tensorboard"])
    sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
    print(f"Logging configured at: {log_dir}")

    # --- 4. Setup Custom DAPG (PPOWithBCLoss) Trainer ---
    print("Initializing custom PPOWithBCLoss agent...")
    model = PPOWithBCLoss(
        policy=MlpPolicy,
        env=venv,
        expert_demonstrations=expert_demonstrations,
        bc_loss_weight=0.05,
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
    )
    print("Custom agent configured.")

    # --- 5. Train the Agent ---
    print("Starting training...")
    # Note: Currently, this will only perform standard PPO training as the
    # BC loss logic is not yet implemented in PPOWithBCLoss.train().
    model.learn(total_timesteps=100_000)
    print("Training complete.")

    # --- 6. Save the Policy ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.policy.save(model_save_path)
    print(f"Trained policy saved to: {model_save_path}")

    venv.close()

if __name__ == "__main__":
    ENV_NAME = "NeedleReach-v0"
    EXPERT_DATA_PATH = os.path.join("data", "expert_data_needle_reach.pkl")

    # Create a unique directory for this experiment
    experiment_name = f"dapg_needle_reach_{int(time.time())}"
    log_dir = os.path.join("logs", experiment_name)
    model_dir = "models"
    # Save model in the models/ dir
    model_save_path = os.path.join(model_dir, f"{experiment_name}.zip")

    train_dapg_agent(
        env_name=ENV_NAME,
        expert_data_path=EXPERT_DATA_PATH,
        model_save_path=model_save_path,
        log_dir=log_dir,
    )
