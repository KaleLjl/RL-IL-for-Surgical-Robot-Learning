#!/usr/bin/env python3
"""
Simplified PPO Curriculum Learning Training Script for PegTransfer Task

This is a streamlined version that focuses on core training functionality
without extensive evaluation and reporting overhead.
"""
import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper

# Import from current directory
from curriculum_config import get_level_config, get_ppo_params, get_max_timesteps, ENV_CONFIG, TRAINING_CONFIG, get_entropy_schedule_config
from entropy_scheduling import get_entropy_schedule, SCHEDULES


class SimpleProgressCallback(BaseCallback):
    """Simple callback to track success rate and rewards during training."""
    
    def __init__(self, level: int, log_freq: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.level = level
        self.log_freq = log_freq
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Record episode result
            episode_reward = self.locals.get("rewards")[0]
            success = info.get("is_success", False)
            
            self.episode_count += 1
            if success:
                self.success_count += 1
            self.episode_rewards.append(episode_reward)
            
            # Print progress at specified frequency
            if self.episode_count % self.log_freq == 0:
                elapsed_time = time.time() - self.start_time
                success_rate = self.success_count / self.episode_count
                recent_rewards = self.episode_rewards[-self.log_freq:]
                avg_reward = np.mean(recent_rewards)
                
                print(f"\n--- Level {self.level} Progress ---")
                print(f"Episodes: {self.episode_count} | Success Rate: {success_rate:.1%} ({self.success_count}/{self.episode_count})")
                print(f"Recent Avg Reward: {avg_reward:.1f} | Time: {elapsed_time/60:.1f}min")
                print("-" * 35)
        
        return True


def create_env(env_name: str, curriculum_level: int, seed: int = 0, render: bool = False):
    """Create environment with curriculum level."""
    def _init():
        env = gym.make(
            env_name,
            render_mode="human" if render else None,
            use_dense_reward=ENV_CONFIG["use_dense_reward"],
            curriculum_level=curriculum_level
        )
        # Wrap for dict observations
        env = FlattenDictObsWrapper(env)
        env.reset(seed=seed)
        return env
    
    return _init


def train_simple(args):
    """Simplified training function."""
    
    level = args.level
    print(f"\n{'='*60}")
    print(f"Simplified PPO Training for {args.env}")
    print(f"Level: {level} - {get_level_config(level)['name']}")
    print(f"{'='*60}\n")
    
    # Create run directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{args.run_name}" if args.run_name else f"run_{timestamp}"
    run_dir = os.path.join("models", "ppo_curriculum", "runs", run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([create_env(args.env, level, seed=0, render=args.render)])
    env = VecMonitor(env)
    
    # Get PPO parameters for current level
    ppo_params = get_ppo_params(level)
    
    # Initialize or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env, **ppo_params)
    else:
        print("Creating new PPO model")
        model = PPO("MlpPolicy", env, **ppo_params)
    
    # Determine timesteps
    total_timesteps = get_max_timesteps(level) if args.timesteps is None else args.timesteps
    
    # Setup callbacks
    callbacks = []
    
    # Progress tracking callback
    progress_callback = SimpleProgressCallback(level=level, log_freq=50)
    callbacks.append(progress_callback)
    
    # Entropy scheduling callback (automatic based on level config)
    entropy_callback = None
    entropy_config = get_entropy_schedule_config(level)
    
    # Use command line args if provided, otherwise use config
    if args.entropy_schedule:
        # Manual override via command line
        entropy_callback = get_entropy_schedule(
            schedule_type=args.entropy_schedule,
            total_timesteps=total_timesteps,
            start_ent=args.start_entropy,
            end_ent=args.end_entropy
        )
        callbacks.append(entropy_callback)
        print(f"  Entropy Schedule: {args.entropy_schedule} ({args.start_entropy} → {args.end_entropy}) [Manual]")
    elif entropy_config.get("enabled", False):
        # Automatic based on level configuration
        entropy_callback = get_entropy_schedule(
            schedule_type=entropy_config["schedule_type"],
            total_timesteps=total_timesteps,
            start_ent=entropy_config["start_entropy"],
            end_ent=entropy_config["end_entropy"]
        )
        callbacks.append(entropy_callback)
        print(f"  Entropy Schedule: {entropy_config['schedule_type']} ({entropy_config['start_entropy']} → {entropy_config['end_entropy']}) [Auto]")
        print(f"    Description: {entropy_config.get('description', 'N/A')}")
    else:
        print(f"  Entropy Schedule: Disabled")
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["checkpoint_frequency"],
        save_path=checkpoint_dir,
        name_prefix=f"ppo_level_{level}"
    )
    callbacks.append(checkpoint_callback)
    
    # Combine callbacks
    callback_list = CallbackList(callbacks)
    
    # Print training configuration
    print(f"Training Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Level: {level} - {get_level_config(level)['name']}")
    print(f"  Dense Rewards: {ENV_CONFIG['use_dense_reward']}")
    print(f"  Learning Rate: {ppo_params['learning_rate']}")
    print(f"  Batch Size: {ppo_params['batch_size']}")
    print(f"  Run Directory: {run_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"Starting training...\n")
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
    training_time = time.time() - start_time
    
    # Save final model in run directory
    model_path = os.path.join(run_dir, f"model_level_{level}_final.zip")
    model.save(model_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Total Episodes: {progress_callback.episode_count}")
    print(f"Final Success Rate: {progress_callback.success_count/progress_callback.episode_count:.1%} ({progress_callback.success_count}/{progress_callback.episode_count})")
    print(f"Final Avg Reward: {np.mean(progress_callback.episode_rewards[-50:]) if len(progress_callback.episode_rewards) >= 50 else np.mean(progress_callback.episode_rewards):.1f}")
    print(f"Run Directory: {run_dir}")
    print(f"Final Model: {model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}")
    
    # Cleanup
    env.close()
    return model_path, run_dir


def main():
    parser = argparse.ArgumentParser(description="Simplified PPO Curriculum Training")
    
    parser.add_argument(
        "--env",
        default="PegTransfer-v0",
        help="Environment name"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Curriculum level to train (1-7)"
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
        "--render",
        action="store_true",
        help="Enable rendering during training"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this training run (will be added to timestamp)"
    )
    
    parser.add_argument(
        "--entropy-schedule",
        type=str,
        choices=['linear', 'exponential', 'stepwise'],
        default=None,
        help="Enable entropy scheduling (reduces exploration over time)"
    )
    
    parser.add_argument(
        "--start-entropy",
        type=float,
        default=0.01,
        help="Starting entropy coefficient (default: 0.01)"
    )
    
    parser.add_argument(
        "--end-entropy", 
        type=float,
        default=0.0001,
        help="Ending entropy coefficient (default: 0.0001)"
    )
    
    args = parser.parse_args()
    
    try:
        model_path, run_dir = train_simple(args)
        print(f"\nTo evaluate this model, run:")
        print(f"docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py {model_path} --render")
        
        # Also print next level training command if not level 7
        if args.level < 7:
            print(f"\nTo train Level {args.level + 1}, run:")
            print(f"docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level {args.level + 1} --model-path {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()