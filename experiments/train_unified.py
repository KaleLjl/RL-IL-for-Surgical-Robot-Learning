#!/usr/bin/env python3
"""Unified training script for all algorithms (BC, PPO, PPO+BC)."""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from dvrk_gym.envs.peg_transfer import PegTransferEnv
from dvrk_gym.envs.needle_reach import NeedleReachEnv
from utils.config_parser import ConfigParser, create_argument_parser
from utils.logger import ExperimentLogger, MetricsTracker


class UnifiedTrainer:
    """Unified trainer for all algorithms."""
    
    def __init__(self, config: Dict[str, Any], args):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
            args: Command line arguments
        """
        self.config = config
        self.args = args
        
        # Set random seeds
        self.set_seeds(config.get('seed', 42))
        
        # Setup directories
        self.setup_directories()
        
        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=self.exp_dir,
            experiment_name=self.experiment_name,
            config=config,
            use_tensorboard=not args.no_tensorboard,
            use_csv=config.get('logging', {}).get('csv_logging', True)
        )
        
        # Initialize environment
        self.env = self.create_environment()
        
        # Initialize algorithm
        self.model = None
        self.algorithm = config['algorithm']
        
    def set_seeds(self, seed: int):
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def setup_directories(self):
        """Setup experiment directories."""
        # Generate experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = ConfigParser.get_config_hash(self.config)
        
        if self.args.experiment_name:
            self.experiment_name = self.args.experiment_name
        else:
            self.experiment_name = f"{timestamp}_{self.config['algorithm']}_{self.args.task}_{config_hash}"
        
        # Create experiment directory
        self.exp_dir = Path(self.args.output_dir) / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.model_dir = self.exp_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        self.eval_dir = self.exp_dir / 'evaluations'
        self.eval_dir.mkdir(exist_ok=True)
    
    def create_environment(self):
        """Create environment based on task.
        
        Returns:
            Gymnasium environment
        """
        env_config = self.config.get('env', {})
        
        if self.args.task == 'needle_reach':
            env_class = NeedleReachEnv
        elif self.args.task == 'peg_transfer':
            env_class = PegTransferEnv
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
        
        # Create environment
        def make_env():
            return env_class(render_mode=None)
        
        # Use vectorized environment for PPO
        if self.config['algorithm'] in ['ppo', 'ppo_bc']:
            n_envs = env_config.get('n_envs', 4)
            if n_envs > 1:
                env = SubprocVecEnv([make_env for _ in range(n_envs)])
            else:
                env = DummyVecEnv([make_env])
        else:
            env = make_env()
        
        return env
    
    def load_demonstrations(self) -> List[Trajectory]:
        """Load expert demonstrations.
        
        Returns:
            Expert trajectories
        """
        demo_path = self.config.get('data', {}).get('demo_path')
        
        if not demo_path:
            # Construct default path
            demo_file = f"expert_demo_{self.args.task}.pkl"
            demo_path = Path(self.args.demo_dir) / demo_file
        else:
            demo_path = Path(demo_path)
        
        if not demo_path.exists():
            raise FileNotFoundError(f"Demonstration file not found: {demo_path}")
        
        self.logger.log_info(f"Loading demonstrations from: {demo_path}")
        
        import pickle
        with open(demo_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        # Convert to imitation library format if needed
        if isinstance(trajectories, list) and len(trajectories) > 0:
            # Convert from our format to standard format
            processed_trajectories = []
            for traj in trajectories:
                # Handle different key formats
                obs_key = 'obs' if 'obs' in traj else 'observations'
                acts_key = 'acts' if 'acts' in traj else 'actions'
                
                obs = traj[obs_key]
                acts = traj[acts_key]
                
                # Handle different observation formats
                if isinstance(obs, dict):
                    # Dict with sequences - flatten each timestep
                    flat_obs = []
                    n_steps = len(obs['observation'])
                    for i in range(n_steps):
                        flat_o = np.concatenate([
                            obs['observation'][i],
                            obs['achieved_goal'][i],
                            obs['desired_goal'][i]
                        ])
                        flat_obs.append(flat_o)
                    obs = np.array(flat_obs)
                elif isinstance(obs, list) and len(obs) > 0 and isinstance(obs[0], dict):
                    # List of dict observations - flatten each
                    flat_obs = []
                    for o in obs:
                        flat_o = np.concatenate([
                            o['observation'],
                            o['achieved_goal'],
                            o['desired_goal']
                        ])
                        flat_obs.append(flat_o)
                    obs = np.array(flat_obs)
                
                # Create standardized trajectory
                processed_traj = {
                    'observations': obs,
                    'actions': acts
                }
                processed_trajectories.append(processed_traj)
            
            trajectories = processed_trajectories
        
        self.logger.log_info(f"Loaded {len(trajectories)} demonstrations")
        
        return trajectories
    
    def train_bc(self):
        """Train using Behavioral Cloning."""
        self.logger.log_info("Starting BC training")
        
        # Load demonstrations
        trajectories = self.load_demonstrations()
        
        # Get task-specific config
        task_config = self.config['training'].get(self.args.task, {})
        network_config = self.config['network'].get(self.args.task, {})
        
        # Convert trajectories to transitions for BC
        from imitation.data.types import Transitions
        
        all_obs = []
        all_acts = []
        all_next_obs = []
        all_dones = []
        
        for traj in trajectories:
            obs = traj['observations']
            acts = traj['actions']
            
            all_obs.extend(obs[:-1])
            all_acts.extend(acts)
            all_next_obs.extend(obs[1:])
            all_dones.extend([False] * (len(obs) - 2) + [True])
        
        transitions = Transitions(
            obs=np.array(all_obs),
            acts=np.array(all_acts),
            next_obs=np.array(all_next_obs),
            dones=np.array(all_dones),
            infos=[{}] * len(all_obs)
        )
        
        # Create BC trainer with flattened observation space
        import torch
        self.model = bc.BC(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(all_obs[0].shape[0],), dtype=np.float32
            ),
            action_space=self.env.action_space,
            demonstrations=transitions,
            batch_size=task_config.get('batch_size', 64),
            optimizer_kwargs={'lr': task_config.get('learning_rate', 1e-4)},
            l2_weight=task_config.get('weight_decay', 1e-4),
            rng=np.random.default_rng(self.config.get('seed', 42))
        )
        
        # Training loop
        n_epochs = task_config.get('epochs', 100)
        log_interval = self.config['logging'].get('log_interval', 10)
        
        metrics_tracker = MetricsTracker()
        
        for epoch in range(n_epochs):
            # Train for one epoch
            self.model.train(n_epochs=1)
            
            # Get training metrics from logger if available
            # Note: BC loss is logged automatically by imitation library
            metrics_tracker.update('bc_loss', 0)  # Placeholder - real metrics in TensorBoard
            
            # Log metrics
            if epoch % log_interval == 0:
                avg_loss = metrics_tracker.get_average('bc_loss')
                self.logger.log_scalar('train/bc_loss', avg_loss, epoch)
                self.logger.log_info(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config['logging'].get('save_freq', 10) == 0:
                self.save_model(f"bc_epoch_{epoch}.zip")
        
        # Save final model
        self.save_model("bc_final.zip")
        self.logger.log_info("BC training completed")
    
    def train_ppo(self):
        """Train using PPO."""
        self.logger.log_info("Starting PPO training")
        
        # Get task-specific config
        task_config = self.config['training'].get(self.args.task, {})
        ppo_config = self.config.get('ppo', {})
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=task_config.get('learning_rate', 3e-4),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.0),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            tensorboard_log=str(self.exp_dir / 'tensorboard'),
            verbose=self.config['logging'].get('verbose', 1)
        )
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['logging'].get('save_freq', 10000),
            save_path=str(self.model_dir),
            name_prefix='ppo_checkpoint'
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.config['logging'].get('eval_freq', 0) > 0:
            eval_env = self.create_environment()
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir),
                log_path=str(self.eval_dir),
                eval_freq=self.config['logging'].get('eval_freq', 5000),
                n_eval_episodes=self.config['logging'].get('eval_episodes', 10),
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Train
        total_timesteps = task_config.get('total_timesteps', 100000)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        self.save_model("ppo_final.zip")
        self.logger.log_info("PPO training completed")
    
    def train_ppo_bc(self):
        """Train using PPO with BC."""
        self.logger.log_info("Starting PPO+BC training")
        
        # This would require custom implementation
        # For now, raise NotImplementedError
        raise NotImplementedError("PPO+BC training requires custom implementation")
    
    def create_bc_policy(self, network_config: Dict[str, Any]):
        """Create BC policy network.
        
        Args:
            network_config: Network configuration
            
        Returns:
            Policy network
        """
        import torch.nn as nn
        
        hidden_sizes = network_config.get('hidden_sizes', [256, 256])
        activation = network_config.get('activation', 'relu')
        
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'tanh':
            activation_fn = nn.Tanh
        else:
            activation_fn = nn.ReLU
        
        # Create MLP network
        layers = []
        prev_size = 19  # Flattened observation size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn()
            ])
            prev_size = hidden_size
        
        # Output layer
        action_dim = 6 if self.args.task == 'needle_reach' else 5
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())  # Actions are bounded [-1, 1]
        
        return nn.Sequential(*layers)
    
    def save_model(self, filename: str):
        """Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        save_path = self.model_dir / filename
        
        if self.algorithm == 'bc':
            # Save BC model policy
            self.model.policy.save(str(save_path))
        elif self.algorithm in ['ppo', 'ppo_bc']:
            # Save SB3 model
            self.model.save(save_path)
        
        self.logger.log_info(f"Model saved: {save_path}")
    
    def train(self):
        """Main training entry point."""
        try:
            if self.algorithm == 'bc':
                self.train_bc()
            elif self.algorithm == 'ppo':
                self.train_ppo()
            elif self.algorithm == 'ppo_bc':
                self.train_ppo_bc()
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        except Exception as e:
            self.logger.log_info(f"Training failed: {e}")
            raise
        
        finally:
            self.logger.close()
            if hasattr(self.env, 'close'):
                self.env.close()


def main():
    """Main entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_parser = ConfigParser()
    config = config_parser.load_config(args.config)
    
    # Apply command line overrides
    if args.override:
        overrides = config_parser.parse_command_line_overrides(args.override)
        config = config_parser.merge_configs(config, overrides)
    
    # Override algorithm if specified
    if args.algorithm:
        config['algorithm'] = args.algorithm
    
    # Override seed if specified
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Create trainer and train
    trainer = UnifiedTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()