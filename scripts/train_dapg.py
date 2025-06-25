import os
import pickle
import gymnasium as gym
import numpy as np
import torch
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.dapg import DAPG
from imitation.data import rollout, types
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import logger as imitation_logger
from stable_baselines3 import PPO
from stable_baselines3.common import logger as sb3_logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecFlattenObservation
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from gymnasium.spaces import Box

import dvrk_gym  # Import to register the environment

def train_dapg_agent(env_name, expert_data_path, model_save_path, log_dir):
    """
    Trains a Demonstration-Augmented Policy Gradient (DAPG) agent.

    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.pkl file).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
    """
    print("--- DAPG Training ---")
    
    # --- 1. Load and Flatten Expert Data (same as in train_bc.py) ---
    print(f"Loading expert data from: {expert_data_path}")
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)

    print("Flattening Dict observations into a single array...")
    all_obs, all_acts, all_infos = [], [], []
    for traj in trajectories:
        obs_soa = traj["obs"]
        num_transitions = len(traj["acts"])
        for i in range(num_transitions):
            flat_obs = np.concatenate([
                obs_soa['observation'][i],
                obs_soa['achieved_goal'][i],
                obs_soa['desired_goal'][i]
            ])
            all_obs.append(flat_obs)
        all_acts.extend(traj["acts"])
        all_infos.extend(traj.get("infos", [{}] * num_transitions))

    # The imitation library expects data in a specific format (Transitions).
    # Note: DAPG does not require next_obs or dones for the demonstrations.
    transitions = types.Transitions(
        obs=np.array(all_obs),
        acts=np.array(all_acts),
        infos=np.array(all_infos),
    )
    print(f"Data flattened and converted to Transitions format: {len(transitions)} samples.")

    # --- 2. Setup Vectorized and Flattened Environment ---
    print(f"Initializing environment: {env_name}")
    # Create a vectorized environment
    venv = DummyVecEnv([lambda: gym.make(env_name)])
    # IMPORTANT: Wrap the environment to flatten Dict observations at runtime.
    # This ensures the observations seen by the policy during RL fine-tuning
    # match the format of the flattened expert data.
    venv = VecFlattenObservation(venv)
    print("Environment wrapped with VecFlattenObservation.")

    # --- 3. Configure Logging ---
    os.makedirs(log_dir, exist_ok=True)
    imitation_logger.configure(folder=log_dir, format_strs=["stdout", "tensorboard"])
    sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
    print(f"Logging configured at: {log_dir}")

    # --- 4. Setup DAPG Trainer ---
    # We use PPO as the underlying RL algorithm for fine-tuning.
    rl_algo = PPO(
        policy=MlpPolicy,
        env=venv,
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

    # DAPG requires a reward network, even if we primarily use the env reward.
    # We use a simple one here.
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )

    dapg_trainer = DAPG(
        venv=venv,
        demonstrations=transitions,
        rl_algo=rl_algo,
        rng=np.random.default_rng(),
    )
    print("DAPG trainer configured.")

    # --- 5. Train the Agent ---
    print("Starting training...")
    # Train for a total of 200,000 RL steps.
    # The BC pre-training happens automatically within the first call.
    dapg_trainer.train(total_timesteps=200_000)
    print("Training complete.")

    # --- 6. Save the Policy ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # The policy is stored within the rl_algo attribute after training
    dapg_trainer.policy.save(model_save_path)
    print(f"Trained policy saved to: {model_save_path}")

    venv.close()

if __name__ == "__main__":
    ENV_NAME = "NeedleReach-v0"
    EXPERT_DATA_PATH = os.path.join("data", "expert_data_needle_reach.pkl")
    MODEL_SAVE_PATH = os.path.join("models", "dapg_needle_reach.zip")
    LOG_DIR = os.path.join("logs", "dapg_needle_reach")
    
    train_dapg_agent(
        env_name=ENV_NAME,
        expert_data_path=EXPERT_DATA_PATH,
        model_save_path=MODEL_SAVE_PATH,
        log_dir=LOG_DIR,
    )
