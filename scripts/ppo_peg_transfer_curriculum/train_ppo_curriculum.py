#!/usr/bin/env python3
"""
PPO Curriculum Learning Training Script for PegTransfer Task

This script implements progressive curriculum learning for the PegTransfer task,
building precision through 4 levels of increasing difficulty.
"""
import os
import sys
import time
import argparse
import json
import platform
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

# Import from current directory since we're in the ppo_peg_transfer_curriculum folder
from curriculum_config import (
    get_level_config, get_ppo_params, get_max_timesteps,
    ENV_CONFIG, TRAINING_CONFIG, CURRICULUM_LEVELS
)


class ManualProgressCallback(BaseCallback):
    """Simple callback for manual curriculum learning - just tracks progress."""
    
    def __init__(self, 
                 level: int,
                 verbose: int = 1):
        super().__init__(verbose)
        self.level = level
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Record episode result
            episode_reward = self.locals.get("rewards")[0]
            episode_length = self.locals.get("episode_lengths")[0] if "episode_lengths" in self.locals else 0
            success = info.get("is_success", False)
            
            self.episode_count += 1
            if success:
                self.success_count += 1
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress every 100 episodes
            if self.episode_count % 100 == 0:
                success_rate = self.success_count / self.episode_count
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                
                print(f"\n--- Level {self.level} Progress Report ---")
                print(f"Episodes: {self.episode_count}")
                print(f"Success Rate: {success_rate:.1%} ({self.success_count}/{self.episode_count})")
                print(f"Recent Avg Reward: {avg_reward:.2f}")
                print(f"Recent Avg Length: {avg_length:.1f}")
                print("-" * 40)
        
        return True
    
    def _on_training_end(self) -> None:
        """Print final statistics."""
        if self.episode_count > 0:
            success_rate = self.success_count / self.episode_count
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            print(f"\n{'='*60}")
            print(f"FINAL LEVEL {self.level} STATISTICS")
            print(f"{'='*60}")
            print(f"Total Episodes: {self.episode_count}")
            print(f"Overall Success Rate: {success_rate:.1%}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Episode Length: {avg_length:.1f}")
            print(f"{'='*60}\n")


def create_env(env_name: str, curriculum_level: int, seed: int = 0):
    """Create environment with curriculum level."""
    def _init():
        env = gym.make(
            env_name,
            render_mode=ENV_CONFIG["render_mode"],
            use_dense_reward=ENV_CONFIG["use_dense_reward"],
            curriculum_level=curriculum_level
        )
        # Wrap for dict observations
        env = FlattenDictObsWrapper(env)
        env.reset(seed=seed)
        return env
    
    return _init


def train_level_manual(args):
    """Manual training function - train one level at a time."""
    
    level = args.level
    print(f"\n{'='*60}")
    print(f"Manual PPO Training for {args.env}")
    print(f"Level: {level} - {get_level_config(level)['name']}")
    print(f"Run: {args.run_name}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = DummyVecEnv([create_env(args.env, level, seed=i) for i in range(1)])
    env = VecMonitor(env)
    
    # Get PPO parameters for current level
    ppo_params = get_ppo_params(level)
    
    # Initialize or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env, **ppo_params)
    else:
        # Create new model
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=os.path.join(args.log_save_path, "tensorboard"),
            **ppo_params
        )
    
    # Setup callbacks
    callbacks = []
    
    # Progress tracking callback
    progress_callback = ManualProgressCallback(level=level, verbose=1)
    callbacks.append(progress_callback)
    
    # Checkpoint callback - save to run directory
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["checkpoint_frequency"],
        save_path=os.path.join(args.model_save_path, "checkpoints"),
        name_prefix=f"ppo_level_{level}"
    )
    callbacks.append(checkpoint_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Train
    try:
        total_timesteps = get_max_timesteps(level) if args.timesteps is None else args.timesteps
        
        print(f"\nTraining for {total_timesteps} timesteps...")
        print(f"PPO Parameters: {ppo_params}\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.reset_timesteps,
            progress_bar=True,
            tb_log_name=f"level_{level}"
        )
        
        # Save final model
        final_model_path = os.path.join(
            args.model_save_path,
            f"model_level_{level}_final.zip"
        )
        model.save(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
        
        # Final evaluation
        print("\nFinal evaluation...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=100, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Save training summary
        training_summary = {
            "level": level,
            "run_name": args.run_name,
            "timesteps_trained": total_timesteps,
            "final_mean_reward": float(mean_reward),
            "final_std_reward": float(std_reward),
            "total_episodes": progress_callback.episode_count,
            "success_rate": float(progress_callback.success_count / progress_callback.episode_count if progress_callback.episode_count > 0 else 0),
            "avg_episode_length": float(np.mean(progress_callback.episode_lengths) if progress_callback.episode_lengths else 0),
            "ppo_params": ppo_params,
        }
        
        summary_path = os.path.join(args.log_save_path, f"training_summary_level_{level}.json")
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        print(f"Training summary saved to {summary_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Still save the model if interrupted
        interrupted_model_path = os.path.join(
            args.model_save_path,
            f"model_level_{level}_interrupted.zip"
        )
        model.save(interrupted_model_path)
        print(f"Interrupted model saved to {interrupted_model_path}")
    
    env.close()


def generate_run_name(tag: str = None) -> str:
    """Generate a unique run name with timestamp and optional tag."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    if tag:
        run_name += f"_{tag}"
    return run_name


def save_run_metadata(run_dir: str, args, run_name: str):
    """Save comprehensive metadata about the training run."""
    metadata = {
        "run_name": run_name,
        "start_time": datetime.now().isoformat(),
        "environment": args.env,
        "curriculum_config": CURRICULUM_LEVELS,
        "env_config": ENV_CONFIG,
        "training_config": TRAINING_CONFIG,
        "system_info": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "command_args": vars(args),
        "command": " ".join(sys.argv),
    }
    
    # Try to get GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            metadata["system_info"]["gpu"] = torch.cuda.get_device_name(0)
            metadata["system_info"]["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Run metadata saved to {metadata_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Manual PPO Training for Curriculum Learning Levels"
    )
    
    parser.add_argument(
        "--env",
        default="PegTransfer-v0",
        help="Environment name"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Curriculum level to train (1-4)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: auto based on level)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model to continue training from"
    )
    
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="Reset timestep counter when loading a model (default: continue counting)"
    )
    
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default="models/ppo_curriculum",
        help="Base directory for saving models"
    )
    
    parser.add_argument(
        "--base-log-dir",
        type=str,
        default="logs/ppo_curriculum",
        help="Base directory for saving logs"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag to append to auto-generated run name"
    )
    
    args = parser.parse_args()
    
    # Generate run name if not provided
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = generate_run_name(args.run_tag)
    
    print(f"\n{'='*60}")
    print(f"Manual PPO Curriculum Training")
    print(f"Run Name: {run_name}")
    print(f"Level: {args.level}")
    print(f"{'='*60}\n")
    
    # Setup directory structure - always create new directories for each training session
    model_run_dir = os.path.join(args.base_model_dir, "runs", run_name)
    log_run_dir = os.path.join(args.base_log_dir, "runs", run_name)
    
    # Create directories
    os.makedirs(model_run_dir, exist_ok=True)
    os.makedirs(log_run_dir, exist_ok=True)
    os.makedirs(os.path.join(model_run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_run_dir, "tensorboard"), exist_ok=True)
    
    # Save metadata for this training session
    save_run_metadata(model_run_dir, args, run_name)
    
    # Update args with paths
    args.model_save_path = model_run_dir
    args.log_save_path = log_run_dir
    args.run_name = run_name
    
    # Train the specified level
    train_level_manual(args)


if __name__ == "__main__":
    main()