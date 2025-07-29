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
                 verbose: int = 1,
                 log_freq: int = 50):
        super().__init__(verbose)
        self.level = level
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_freq = log_freq
        self.start_time = time.time()
        self.last_success_count = 0
        self.timesteps_done = 0
        
    def _on_step(self) -> bool:
        # Update timestep counter
        self.timesteps_done += 1
        
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
            
            # Print progress at specified frequency
            if self.episode_count % self.log_freq == 0:
                # Calculate metrics
                elapsed_time = time.time() - self.start_time
                episodes_per_second = self.episode_count / elapsed_time
                timesteps_per_second = self.timesteps_done / elapsed_time
                
                overall_success_rate = self.success_count / self.episode_count
                recent_episodes = min(self.log_freq, len(self.episode_rewards))
                recent_success_rate = (self.success_count - self.last_success_count) / recent_episodes
                self.last_success_count = self.success_count
                
                recent_rewards = self.episode_rewards[-recent_episodes:]
                recent_lengths = self.episode_lengths[-recent_episodes:]
                
                # Print comprehensive progress report
                print(f"\n{'='*60}")
                print(f"LEVEL {self.level} TRAINING PROGRESS - Episode {self.episode_count}")
                print(f"{'='*60}")
                print(f"Time Elapsed: {elapsed_time/60:.1f} min | Speed: {episodes_per_second:.1f} eps/sec, {timesteps_per_second:.0f} steps/sec")
                print(f"{'─'*60}")
                print(f"Overall Success Rate: {overall_success_rate:6.1%} ({self.success_count}/{self.episode_count})")
                print(f"Recent Success Rate:  {recent_success_rate:6.1%} (last {recent_episodes} episodes)")
                print(f"{'─'*60}")
                print(f"Recent Avg Reward:    {np.mean(recent_rewards):6.1f} ± {np.std(recent_rewards):.1f}")
                print(f"Recent Min/Max:       {np.min(recent_rewards):6.1f} / {np.max(recent_rewards):.1f}")
                print(f"Recent Avg Length:    {np.mean(recent_lengths):6.1f} steps")
                print(f"{'─'*60}")
                
                # Add trend indicator
                if len(self.episode_rewards) >= self.log_freq * 2:
                    old_rewards = self.episode_rewards[-2*self.log_freq:-self.log_freq]
                    reward_trend = np.mean(recent_rewards) - np.mean(old_rewards)
                    trend_symbol = "↑" if reward_trend > 1 else ("↓" if reward_trend < -1 else "→")
                    print(f"Reward Trend: {trend_symbol} {reward_trend:+.1f}")
                
                # Success indicator
                if recent_success_rate > 0.8:
                    print("Status: 🟢 Excellent progress!")
                elif recent_success_rate > 0.5:
                    print("Status: 🟡 Good progress, keep going!")
                elif recent_success_rate > 0.2:
                    print("Status: 🟠 Some progress, may need tuning")
                else:
                    print("Status: 🔴 Struggling, consider adjustments")
                    
                print(f"{'='*60}\n")
        
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


def train_level_manual(args):
    """Manual training function - train one level at a time."""
    
    start_time = time.time()  # Track training duration
    level = args.level
    print(f"\n{'='*60}")
    print(f"Manual PPO Training for {args.env}")
    print(f"Level: {level} - {get_level_config(level)['name']}")
    print(f"Run: {args.run_name}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = DummyVecEnv([create_env(args.env, level, seed=i, render=args.render) for i in range(1)])
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
        
        # Print comprehensive training configuration
        print(f"\n{'─'*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'─'*60}")
        print(f"Environment: {args.env}")
        print(f"Curriculum Level: {level} - {get_level_config(level)['name']}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Max Episode Steps: {get_level_config(level).get('max_episode_steps', 'default')}")
        print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
        print(f"Dense Rewards: {ENV_CONFIG['use_dense_reward']}")
        print(f"Early Exit: {ENV_CONFIG['early_exit_enabled']}")
        print(f"{'─'*60}")
        print(f"PPO HYPERPARAMETERS")
        print(f"{'─'*60}")
        for key, value in ppo_params.items():
            print(f"  {key}: {value}")
        print(f"{'─'*60}\n")
        
        print(f"Starting training...")
        
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
        
        # Calculate additional statistics
        success_rate = float(progress_callback.success_count / progress_callback.episode_count if progress_callback.episode_count > 0 else 0)
        avg_episode_length = float(np.mean(progress_callback.episode_lengths) if progress_callback.episode_lengths else 0)
        
        # Save detailed training summary
        training_summary = {
            "level": level,
            "level_name": get_level_config(level)['name'],
            "run_name": args.run_name,
            "timesteps_trained": total_timesteps,
            "training_time_seconds": time.time() - start_time,
            "final_mean_reward": float(mean_reward),
            "final_std_reward": float(std_reward),
            "total_episodes": progress_callback.episode_count,
            "success_count": progress_callback.success_count,
            "success_rate": success_rate,
            "avg_episode_length": avg_episode_length,
            "avg_reward_all_episodes": float(np.mean(progress_callback.episode_rewards)) if progress_callback.episode_rewards else 0,
            "reward_improvement": float(progress_callback.episode_rewards[-1] - progress_callback.episode_rewards[0]) if len(progress_callback.episode_rewards) > 1 else 0,
            "ppo_params": ppo_params,
            "environment_config": {
                "use_dense_reward": ENV_CONFIG["use_dense_reward"],
                "early_exit_enabled": ENV_CONFIG["early_exit_enabled"],
                "max_episode_steps": get_level_config(level).get('max_episode_steps', 'default')
            }
        }
        
        # Save JSON summary
        summary_path = os.path.join(args.log_save_path, f"training_summary_level_{level}.json")
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        print(f"Training summary saved to {summary_path}")
        
        # Create human-readable training report
        report_path = os.path.join(args.log_save_path, f"training_report_level_{level}.txt")
        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"PPO CURRICULUM TRAINING REPORT\n")
            f.write(f"Level {level}: {get_level_config(level)['name']}\n")
            f.write(f"Run: {args.run_name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"TRAINING RESULTS:\n")
            f.write(f"  Total Timesteps: {total_timesteps:,}\n")
            f.write(f"  Total Episodes: {progress_callback.episode_count}\n")
            f.write(f"  Training Time: {(time.time() - start_time)/60:.1f} minutes\n")
            f.write(f"  Success Rate: {success_rate:.1%} ({progress_callback.success_count}/{progress_callback.episode_count})\n")
            f.write(f"  Final Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"  Average Episode Length: {avg_episode_length:.1f} steps\n\n")
            
            f.write(f"HYPERPARAMETERS:\n")
            for key, value in ppo_params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n")
            
            f.write(f"RECOMMENDATION:\n")
            if success_rate >= 0.8:
                f.write(f"  ✓ Success rate is excellent ({success_rate:.1%})! Ready to advance to next level.\n")
            elif success_rate >= 0.5:
                f.write(f"  ⚠ Success rate is moderate ({success_rate:.1%}). Consider more training or tuning.\n")
            else:
                f.write(f"  ✗ Success rate is low ({success_rate:.1%}). Needs significant improvement.\n")
            
            f.write(f"\nNEXT STEPS:\n")
            if level < 4:
                f.write(f"  To train Level {level+1}, run:\n")
                f.write(f"  docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level {level+1} --model-path {final_model_path}\n")
            else:
                f.write(f"  Training complete! Evaluate the final model:\n")
                f.write(f"  docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py --run-name {args.run_name} --render\n")
                
        print(f"\nDetailed training report saved to {report_path}")
        
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
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training"
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