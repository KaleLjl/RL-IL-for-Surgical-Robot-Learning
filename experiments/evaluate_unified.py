#!/usr/bin/env python3
"""Unified evaluation script for all trained models."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.algorithms import bc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from dvrk_gym.envs.peg_transfer import PegTransferEnv
from dvrk_gym.envs.needle_reach import NeedleReachEnv
from utils.config_parser import ConfigParser
from utils.logger import ExperimentLogger


class UnifiedEvaluator:
    """Unified evaluator for all algorithms."""
    
    def __init__(self,
                 model_path: str,
                 task: str,
                 algorithm: str,
                 config_path: Optional[str] = None,
                 n_episodes: int = 100,
                 render: bool = False,
                 save_videos: bool = False,
                 output_dir: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            task: Task name (needle_reach or peg_transfer)
            algorithm: Algorithm name (bc, ppo, ppo_bc)
            config_path: Path to configuration file
            n_episodes: Number of evaluation episodes
            render: Whether to render environment
            save_videos: Whether to save evaluation videos
            output_dir: Directory for outputs
        """
        self.model_path = Path(model_path)
        self.task = task
        self.algorithm = algorithm
        self.n_episodes = n_episodes
        self.render = render
        self.save_videos = save_videos
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            config_parser = ConfigParser()
            self.config = config_parser.load_config(config_path)
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.model_path.parent.parent / 'evaluations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        self.env = self.create_environment()
        
        # Load model
        self.model = self.load_model()
        
        # Initialize metrics storage
        self.metrics = {
            'success_rate': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_episodes': [],
            'failure_episodes': [],
            'failure_reasons': {}
        }
    
    def create_environment(self):
        """Create evaluation environment.
        
        Returns:
            Gymnasium environment
        """
        import gymnasium as gym
        
        # Use gym.make to get proper TimeLimit wrapper (consistent with training)
        if self.task == 'needle_reach':
            env_name = 'NeedleReach-v0'
        elif self.task == 'peg_transfer':
            env_name = 'PegTransfer-v0'
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Create environment with proper wrappers
        # Note: render_mode handling may need adjustment
        env = gym.make(env_name)
        
        return env
    
    def load_model(self):
        """Load trained model.
        
        Returns:
            Loaded model
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.algorithm == 'bc':
            # Load BC model
            model = self.load_bc_model()
        elif self.algorithm == 'ppo':
            # Load PPO model
            model = PPO.load(self.model_path, env=self.env)
        elif self.algorithm == 'ppo_bc':
            # Load PPO+BC model (same as PPO for loading)
            model = PPO.load(self.model_path, env=self.env)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return model
    
    def load_bc_model(self):
        """Load BC model.
        
        Returns:
            BC policy from SB3
        """
        # Use the exact same approach as the old working script
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        # Load the policy directly (like old script)
        policy = ActorCriticPolicy.load(str(self.model_path))
        
        return policy
    
    def flatten_observation(self, obs: Dict) -> np.ndarray:
        """Flatten dictionary observation.
        
        Args:
            obs: Dictionary observation
            
        Returns:
            Flattened observation array
        """
        return np.concatenate([
            obs['observation'],
            obs['achieved_goal'],
            obs['desired_goal']
        ])
    
    def predict_action(self, obs: Any) -> np.ndarray:
        """Predict action from observation.
        
        Args:
            obs: Observation (dict or array)
            
        Returns:
            Action array
        """
        if self.algorithm == 'bc':
            # For BC, flatten the dict observation first
            if isinstance(obs, dict):
                flat_obs = self.flatten_observation(obs)
            else:
                flat_obs = obs
            
            action, _ = self.model.predict(flat_obs, deterministic=True)
            if hasattr(action, 'numpy'):
                action = action.numpy()
        
        else:  # PPO or PPO+BC
            action, _ = self.model.predict(obs, deterministic=True)
        
        return action
    
    def run_episode(self, episode_idx: int) -> Tuple[bool, float, int, Dict]:
        """Run a single evaluation episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Tuple of (success, reward, length, info)
        """
        obs, _ = self.env.reset()
        done = False
        truncated = False
        
        episode_reward = 0
        episode_length = 0
        episode_info = {
            'trajectory': [],
            'actions': [],
            'rewards': []
        }
        
        while not (done or truncated):
            # Get action
            action = self.predict_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Store trajectory data
            episode_info['trajectory'].append(obs)
            episode_info['actions'].append(action)
            episode_info['rewards'].append(reward)
            
            # Check for timeout
            if episode_length >= 200:  # Max episode length
                truncated = True
                episode_info['failure_reason'] = 'timeout'
        
        # Determine success
        success = info.get('is_success', False) if 'is_success' in info else (episode_reward > 0)
        
        # Analyze failure if not successful
        if not success and 'failure_reason' not in episode_info:
            episode_info['failure_reason'] = self.analyze_failure(episode_info)
        
        return success, episode_reward, episode_length, episode_info
    
    def analyze_failure(self, episode_info: Dict) -> str:
        """Analyze failure reason from episode data.
        
        Args:
            episode_info: Episode information
            
        Returns:
            Failure reason string
        """
        # Task-specific failure analysis
        if self.task == 'needle_reach':
            # Check if robot got close to target
            final_obs = episode_info['trajectory'][-1] if episode_info['trajectory'] else None
            if final_obs and isinstance(final_obs, dict):
                distance = np.linalg.norm(final_obs['achieved_goal'] - final_obs['desired_goal'])
                if distance > 0.01:  # 10mm threshold
                    return 'failed_to_reach'
        
        elif self.task == 'peg_transfer':
            # Analyze based on reward progression
            rewards = episode_info['rewards']
            if len(rewards) > 0:
                # Check if grasping occurred
                if max(rewards[:50]) < 0.1:  # No early reward
                    return 'failed_to_grasp'
                elif max(rewards[50:100] if len(rewards) > 50 else [0]) < 0.5:
                    return 'dropped_object'
                else:
                    return 'failed_to_place'
        
        return 'unknown'
    
    def evaluate(self):
        """Run full evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Starting evaluation of {self.algorithm} on {self.task}")
        print(f"Running {self.n_episodes} episodes...")
        
        successes = []
        
        for i in range(self.n_episodes):
            # Run episode
            success, reward, length, info = self.run_episode(i)
            
            # Update metrics
            successes.append(success)
            self.metrics['episode_rewards'].append(reward)
            self.metrics['episode_lengths'].append(length)
            
            if success:
                self.metrics['success_episodes'].append(i)
            else:
                self.metrics['failure_episodes'].append(i)
                failure_reason = info.get('failure_reason', 'unknown')
                if failure_reason not in self.metrics['failure_reasons']:
                    self.metrics['failure_reasons'][failure_reason] = 0
                self.metrics['failure_reasons'][failure_reason] += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                current_success_rate = np.mean(successes) * 100
                print(f"Episode {i+1}/{self.n_episodes}: Success rate = {current_success_rate:.1f}%")
        
        # Calculate final metrics
        self.metrics['success_rate'] = np.mean(successes)
        self.metrics['mean_reward'] = np.mean(self.metrics['episode_rewards'])
        self.metrics['std_reward'] = np.std(self.metrics['episode_rewards'])
        self.metrics['mean_length'] = np.mean(self.metrics['episode_lengths'])
        self.metrics['std_length'] = np.std(self.metrics['episode_lengths'])
        
        # Calculate confidence interval
        from scipy import stats
        confidence = 0.95
        n = len(successes)
        se = stats.sem(successes)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        self.metrics['success_rate_ci'] = h
        
        return self.metrics
    
    def save_results(self):
        """Save evaluation results."""
        # Create results dictionary
        results = {
            'model_path': str(self.model_path),
            'task': self.task,
            'algorithm': self.algorithm,
            'n_episodes': self.n_episodes,
            'metrics': self.metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to JSON
        output_file = self.output_dir / f"evaluation_{self.algorithm}_{self.task}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*50)
        print(f"EVALUATION SUMMARY - {self.algorithm.upper()} on {self.task}")
        print("="*50)
        print(f"Success Rate: {self.metrics['success_rate']*100:.1f}% ± {self.metrics['success_rate_ci']*100:.1f}%")
        print(f"Mean Reward: {self.metrics['mean_reward']:.2f} ± {self.metrics['std_reward']:.2f}")
        print(f"Mean Episode Length: {self.metrics['mean_length']:.1f} ± {self.metrics['std_length']:.1f}")
        
        if self.metrics['failure_reasons']:
            print("\nFailure Analysis:")
            total_failures = len(self.metrics['failure_episodes'])
            for reason, count in self.metrics['failure_reasons'].items():
                percentage = (count / total_failures) * 100
                print(f"  - {reason}: {count} ({percentage:.1f}%)")
        
        print("="*50)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified evaluation script')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--task', type=str, required=True,
                       choices=['needle_reach', 'peg_transfer'],
                       help='Task to evaluate on')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['bc', 'ppo', 'ppo_bc'],
                       help='Algorithm used for training')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--save-videos', action='store_true',
                       help='Save evaluation videos')
    parser.add_argument('--output-dir', type=str,
                       help='Directory for outputs')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = UnifiedEvaluator(
        model_path=args.model,
        task=args.task,
        algorithm=args.algorithm,
        config_path=args.config,
        n_episodes=args.n_episodes,
        render=args.render,
        save_videos=args.save_videos,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Save results
    evaluator.save_results()
    
    # Close environment
    evaluator.env.close()


if __name__ == "__main__":
    main()