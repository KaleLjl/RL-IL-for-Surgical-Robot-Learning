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
from curriculum_manager import CurriculumManager
from curriculum_config import (
    get_level_config, get_ppo_params, get_max_timesteps,
    ENV_CONFIG, TRAINING_CONFIG
)


class CurriculumCallback(BaseCallback):
    """Custom callback for curriculum learning management."""
    
    def __init__(self, 
                 curriculum_manager: CurriculumManager,
                 env_name: str,
                 save_path: str,
                 verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.env_name = env_name
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Record episode result
            episode_reward = self.locals.get("rewards")[0]
            episode_length = self.locals.get("episode_lengths")[0] if "episode_lengths" in self.locals else 0
            success = info.get("is_success", False)
            
            self.curriculum_manager.add_episode_result(
                success=success,
                episode_length=episode_length,
                episode_reward=episode_reward,
                info={"early_exit": info.get("early_exit", False)}
            )
            
            # Print progress every 100 episodes
            if self.curriculum_manager.state["total_episodes"] % 100 == 0:
                self.curriculum_manager.print_progress()
            
            # Check for level advancement
            if self.curriculum_manager.should_advance_level():
                current_level = self.curriculum_manager.get_current_level()
                
                # Save current model
                model_path = os.path.join(
                    self.save_path, 
                    f"ppo_curriculum_level_{current_level}_final.zip"
                )
                self.model.save(model_path)
                
                # Advance to next level
                new_level = self.curriculum_manager.advance_level(model_path)
                
                # Update environment for new level
                self._update_environment_level(new_level)
                
                # Optionally update hyperparameters for new level
                # Note: This is tricky with SB3, might need to recreate model
                
                # Save curriculum state
                self.curriculum_manager.save_state()
        
        return True
    
    def _update_environment_level(self, new_level: int):
        """Update environment to new curriculum level."""
        # This is a bit hacky but works for our purposes
        # We need to update the curriculum_level in all environments
        
        # For DummyVecEnv, we can access the environments directly
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                # Navigate through wrappers to get to the base environment
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                if hasattr(base_env, 'curriculum_level'):
                    base_env.curriculum_level = new_level
                    print(f"Updated environment to curriculum level {new_level}")
    
    def _on_training_end(self) -> None:
        """Save final state when training ends."""
        self.curriculum_manager.save_state()
        print(self.curriculum_manager.get_stats_summary())


def create_env(env_name: str, curriculum_level: int, seed: int = 0):
    """Create environment with curriculum level."""
    def _init():
        env = gym.make(
            env_name,
            render_mode=ENV_CONFIG["render_mode"],
            use_dense_reward=ENV_CONFIG["use_dense_reward"],
            early_exit_enabled=ENV_CONFIG["early_exit_enabled"],
            curriculum_level=curriculum_level
        )
        # Wrap for dict observations
        env = FlattenDictObsWrapper(env)
        env.reset(seed=seed)
        return env
    
    return _init


def train_curriculum(args):
    """Main training function for curriculum learning."""
    
    # Initialize curriculum manager
    if args.resume:
        curriculum_state_path = os.path.join(args.save_path, "curriculum_state.json")
        if os.path.exists(curriculum_state_path):
            curriculum_manager = CurriculumManager(
                save_path=args.save_path,
                resume_from_file=curriculum_state_path
            )
            start_level = curriculum_manager.get_current_level()
            print(f"Resuming from curriculum level {start_level}")
        else:
            print(f"No saved state found at {curriculum_state_path}, starting fresh")
            curriculum_manager = CurriculumManager(save_path=args.save_path)
            start_level = args.start_level
    else:
        curriculum_manager = CurriculumManager(save_path=args.save_path)
        start_level = args.start_level
        if start_level != 1:
            # Manually set starting level if not 1
            curriculum_manager.state["current_level"] = start_level
    
    print(f"\n{'='*60}")
    print(f"Starting PPO Curriculum Learning for {args.env}")
    print(f"Initial Level: {start_level} - {get_level_config(start_level)['name']}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = DummyVecEnv([create_env(args.env, start_level, seed=i) for i in range(1)])
    env = VecMonitor(env)
    
    # Get PPO parameters for current level
    ppo_params = get_ppo_params(start_level)
    
    # Initialize or load model
    if args.resume and args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = PPO.load(args.model_path, env=env, **ppo_params)
    elif start_level > 1 and args.previous_model:
        # Load model from previous level
        print(f"Loading model from previous level: {args.previous_model}")
        model = PPO.load(args.previous_model, env=env, **ppo_params)
    else:
        # Create new model
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=os.path.join(args.save_path, "tensorboard"),
            **ppo_params
        )
    
    # Setup callbacks
    callbacks = []
    
    # Curriculum callback
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum_manager,
        env_name=args.env,
        save_path=args.save_path,
        verbose=1
    )
    callbacks.append(curriculum_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["checkpoint_frequency"],
        save_path=args.save_path,
        name_prefix=f"ppo_curriculum_level_{start_level}"
    )
    callbacks.append(checkpoint_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Train for current level
    try:
        total_timesteps = get_max_timesteps(start_level) if args.timesteps is None else args.timesteps
        
        print(f"\nTraining for {total_timesteps} timesteps...")
        print(f"PPO Parameters: {ppo_params}\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(
            args.save_path,
            f"ppo_curriculum_level_{curriculum_manager.get_current_level()}_final.zip"
        )
        model.save(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    finally:
        # Save curriculum state
        curriculum_manager.save_state()
        print(curriculum_manager.get_stats_summary())
        
        # Final evaluation
        print("\nFinal evaluation...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=100, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with Curriculum Learning on PegTransfer"
    )
    
    parser.add_argument(
        "--env",
        default="PegTransfer-v0",
        help="Environment name"
    )
    
    parser.add_argument(
        "--start-level",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Starting curriculum level (1-4)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: auto based on level)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved curriculum state"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to saved model to resume from"
    )
    
    parser.add_argument(
        "--previous-model",
        type=str,
        default=None,
        help="Path to model from previous curriculum level"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_curriculum/",
        help="Path to save models and logs"
    )
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Train
    train_curriculum(args)


if __name__ == "__main__":
    main()