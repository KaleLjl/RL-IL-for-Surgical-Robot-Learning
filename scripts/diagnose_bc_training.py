#!/usr/bin/env python3

import os
import pickle
import gymnasium as gym
import numpy as np
from imitation.algorithms import bc
from imitation.data import types
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from gymnasium.spaces import Box
import torch

import dvrk_gym

def diagnose_bc_data_and_training(env_name, expert_data_path):
    print(f"=== BC Training Diagnosis for {env_name} ===")
    
    # 1. Load expert data
    print("\n1. Loading expert data...")
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)
    
    print(f"Number of trajectories: {len(trajectories)}")
    
    # 2. Flatten observations
    print("\n2. Processing observations...")
    all_obs = []
    all_next_obs = []
    all_acts = []
    all_dones = []
    
    for i, traj in enumerate(trajectories):
        obs_soa = traj["obs"]
        num_transitions = len(traj["acts"])
        
        # Check for any invalid data in this trajectory
        for j in range(num_transitions):
            try:
                flat_obs = np.concatenate([
                    obs_soa['observation'][j],
                    obs_soa['achieved_goal'][j], 
                    obs_soa['desired_goal'][j]
                ])
                all_obs.append(flat_obs)
                
                flat_next_obs = np.concatenate([
                    obs_soa['observation'][j+1],
                    obs_soa['achieved_goal'][j+1],
                    obs_soa['desired_goal'][j+1]
                ])
                all_next_obs.append(flat_next_obs)
                
                # Check for invalid actions
                action = traj["acts"][j]
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    print(f"WARNING: Invalid action in trajectory {i}, step {j}: {action}")
                all_acts.append(action)
                
            except Exception as e:
                print(f"ERROR processing trajectory {i}, step {j}: {e}")
                return
        
        dones = [False] * (num_transitions - 1) + [True]
        all_dones.extend(dones)
    
    # Convert to arrays
    all_obs = np.array(all_obs)
    all_next_obs = np.array(all_next_obs)
    all_acts = np.array(all_acts)
    all_dones = np.array(all_dones)
    
    print(f"Flattened data shapes:")
    print(f"  Observations: {all_obs.shape}")
    print(f"  Actions: {all_acts.shape}")
    print(f"  Next observations: {all_next_obs.shape}")
    print(f"  Dones: {all_dones.shape}")
    
    # 3. Check data statistics
    print("\n3. Data statistics:")
    print(f"Observation range: [{np.min(all_obs):.3f}, {np.max(all_obs):.3f}]")
    print(f"Action range: [{np.min(all_acts):.3f}, {np.max(all_acts):.3f}]")
    print(f"Observation std: {np.std(all_obs):.3f}")
    print(f"Action std: {np.std(all_acts):.3f}")
    
    # Check for problematic values
    print("\n4. Data quality checks:")
    obs_nan = np.any(np.isnan(all_obs))
    obs_inf = np.any(np.isinf(all_obs))
    act_nan = np.any(np.isnan(all_acts))
    act_inf = np.any(np.isinf(all_acts))
    
    print(f"Observations contain NaN: {obs_nan}")
    print(f"Observations contain Inf: {obs_inf}")
    print(f"Actions contain NaN: {act_nan}")
    print(f"Actions contain Inf: {act_inf}")
    
    if obs_nan or obs_inf or act_nan or act_inf:
        print("ERROR: Data contains invalid values!")
        return
    
    # 4. Setup environment and check compatibility
    print("\n5. Environment compatibility:")
    env = gym.make(env_name)
    venv = DummyVecEnv([lambda: env])
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    print(f"Expected flattened obs shape: {sum(space.shape[0] for space in env.observation_space.spaces.values())}")
    print(f"Actual flattened obs shape: {all_obs.shape[1]}")
    print(f"Expected action shape: {env.action_space.shape[0]}")
    print(f"Actual action shape: {all_acts.shape[1]}")
    
    # 5. Create BC trainer and check for issues
    print("\n6. BC trainer setup:")
    transitions = types.Transitions(
        obs=all_obs,
        acts=all_acts,
        next_obs=all_next_obs,
        dones=all_dones,
        infos=np.array([{} for _ in range(len(all_obs))]),
    )
    
    flat_obs_space = Box(
        low=-np.inf, high=np.inf, shape=all_obs.shape[1:], dtype=np.float32
    )
    
    print(f"Created observation space: {flat_obs_space}")
    print(f"Action space: {venv.action_space}")
    
    # Test policy creation
    try:
        policy = MlpPolicy(
            observation_space=flat_obs_space,
            action_space=venv.action_space,
            lr_schedule=lambda _: 0.0001,
            net_arch=[256, 256],
        )
        print("Policy created successfully")
        
        # Test BC trainer
        bc_trainer = bc.BC(
            observation_space=flat_obs_space,
            action_space=venv.action_space,
            demonstrations=transitions,
            policy=policy,
            rng=np.random.default_rng(),
        )
        print("BC trainer created successfully")
        
        # Test a few training steps
        print("\n7. Testing training steps...")
        try:
            # Run just a few steps to see if training works
            bc_trainer.train(n_epochs=1, log_interval=1)
            print("Training test completed successfully")
        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR creating policy/trainer: {e}")
        import traceback
        traceback.print_exc()
    
    venv.close()

if __name__ == "__main__":
    # Test both environments
    envs_to_test = [
        ("NeedleReach-v0", "data/expert_data_needle_reach.pkl"),
        ("PegTransfer-v0", "data/expert_data_peg_transfer.pkl")
    ]
    
    for env_name, data_path in envs_to_test:
        if os.path.exists(data_path):
            try:
                diagnose_bc_data_and_training(env_name, data_path)
                print("\n" + "="*80 + "\n")
            except Exception as e:
                print(f"CRITICAL ERROR with {env_name}: {e}")
                import traceback
                traceback.print_exc()
                print("\n" + "="*80 + "\n")
        else:
            print(f"Data file not found: {data_path}")