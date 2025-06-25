import os
import pickle
import gymnasium as gym
import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import logger as imitation_logger
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

import dvrk_gym  # Import to register the environment

def train_bc_agent(env_name, expert_data_path, model_save_path, log_dir):
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

    # Convert the list of dictionaries to a list of Trajectory objects
    trajectories = [
        types.Trajectory(obs=traj["obs"], acts=traj["acts"], infos=None, terminal=True)
        for traj in trajectories
    ]

    # Use the recommended utility to flatten trajectories into transitions
    transitions = rollout.flatten_trajectories(trajectories)
    print(f"Loaded and processed {len(transitions)} transitions.")

    # --- 2. Setup Environment ---
    print(f"Initializing environment: {env_name}")
    venv = DummyVecEnv([lambda: gym.make(env_name)])

    # --- 3. Configure Logging ---
    os.makedirs(log_dir, exist_ok=True)
    imitation_format_strings = ["stdout", "tensorboard"]
    imitation_logger.configure(folder=log_dir, format_strings=imitation_format_strings)
    sb3_logger.configure(folder=log_dir, format_strings=imitation_format_strings)
    print(f"Logging configured at: {log_dir}")

    # --- 4. Setup BC Trainer ---
    # For dictionary observation spaces, we must use the MultiInputActorCriticPolicy.
    policy = MultiInputActorCriticPolicy(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max, # Effectively a constant learning rate
    )

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=np.random.default_rng(),
    )
    print("BC trainer configured.")

    # --- 5. Train the Agent ---
    print("Starting training...")
    bc_trainer.train(n_epochs=100, log_interval=10)
    print("Training complete.")

    # --- 6. Save the Policy ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bc_trainer.policy.save(model_save_path)
    print(f"Trained policy saved to: {model_save_path}")

    venv.close()

if __name__ == "__main__":
    ENV_NAME = "NeedleReach-v0"
    EXPERT_DATA_PATH = os.path.join("data", "expert_data_needle_reach.pkl")
    MODEL_SAVE_PATH = os.path.join("models", "bc_needle_reach.zip")
    LOG_DIR = os.path.join("logs", "bc_needle_reach")
    
    train_bc_agent(
        env_name=ENV_NAME,
        expert_data_path=EXPERT_DATA_PATH,
        model_save_path=MODEL_SAVE_PATH,
        log_dir=LOG_DIR,
    )
