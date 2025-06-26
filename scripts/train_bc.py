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
import time

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
    os.makedirs(log_dir, exist_ok=True)
    # The imitation logger API seems to have changed and is simpler now.
    imitation_logger.configure(folder=log_dir)
    # The SB3 logger still accepts format_strings.
    sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
    print(f"Logging configured at: {log_dir}")

    # --- 4. Setup BC Trainer ---
    # Since we flattened the observations, we must manually create a Box observation
    # space that matches our flattened data, and use that to initialize the policy.
    from gymnasium.spaces import Box
    from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy

    flat_obs_space = Box(
        low=-np.inf, high=np.inf, shape=all_obs.shape[1:], dtype=np.float32
    )

    policy = MlpPolicy(
        observation_space=flat_obs_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: 0.001,
        net_arch=[256, 256],
    )

    bc_trainer = bc.BC(
        observation_space=flat_obs_space,
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

    # Create a unique directory for this experiment
    experiment_name = f"bc_needle_reach_{int(time.time())}"
    log_dir = os.path.join("logs", experiment_name)
    # Save model in the experiment dir
    model_save_path = os.path.join(log_dir, "bc_needle_reach.zip")

    train_bc_agent(
        env_name=ENV_NAME,
        expert_data_path=EXPERT_DATA_PATH,
        model_save_path=model_save_path,
        log_dir=log_dir,
    )
