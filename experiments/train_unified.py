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
        import gymnasium as gym
        
        # Use gym.make to get proper TimeLimit wrapper (like old scripts)
        # Use dense rewards for PPO/PPO_BC, sparse for BC
        if self.args.task == 'needle_reach':
            if self.config['algorithm'] in ['ppo', 'ppo_bc']:
                env_name = 'NeedleReach-Dense-v0'  # Dense rewards for RL
            else:
                env_name = 'NeedleReach-v0'  # Sparse rewards for BC
        elif self.args.task == 'peg_transfer':
            env_name = 'PegTransfer-v0'
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
        
        # Create environment with proper wrappers
        def make_env():
            return gym.make(env_name)
        
        # Use vectorized environment for PPO with flattened observations (like old script)
        if self.config['algorithm'] in ['ppo', 'ppo_bc']:
            # Import the exact wrapper used in old script
            from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
            
            def make_env_flattened():
                base_env = gym.make(env_name)
                # Apply the same wrapper as successful old script
                return FlattenDictObsWrapper(base_env)
            
            n_envs = self.config.get('env', {}).get('n_envs', 4)
            if n_envs > 1:
                env = SubprocVecEnv([make_env_flattened for _ in range(n_envs)])
            else:
                env = DummyVecEnv([make_env_flattened])
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
            # Construct default path to data directory
            demo_file = f"expert_data_{self.args.task}.pkl"
            demo_path = Path("data") / demo_file
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
        
        # Load raw demonstrations
        demo_path = f"data/expert_data_{self.args.task}.pkl"
        self.logger.log_info(f"Loading demonstrations from: {demo_path}")
        
        import pickle
        with open(demo_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        # Apply the EXACT working solution from old_scripts/train_bc.py
        # Flatten Dict observations into a single array to avoid imitation library issues
        self.logger.log_info("Flattening Dict observations into a single array...")
        
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

        # Convert lists to numpy arrays
        all_obs = np.array(all_obs)
        all_next_obs = np.array(all_next_obs)
        all_acts = np.array(all_acts)
        all_dones = np.array(all_dones)

        # Convert to imitation library format
        from imitation.data.types import Transitions
        transitions = Transitions(
            obs=all_obs,
            acts=all_acts,
            next_obs=all_next_obs,
            dones=all_dones,
            infos=np.array([{} for _ in range(len(all_obs))]),
        )
        self.logger.log_info(f"Data flattened and converted to Transitions format: {len(transitions)} samples.")
        
        # Get task-specific config
        task_config = self.config['training'].get(self.args.task, {})
        network_config = self.config['network'].get(self.args.task, {})
        
        # Create flattened observation space
        import gymnasium as gym
        from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
        
        flat_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=all_obs.shape[1:], dtype=np.float32
        )
        
        # Create custom policy with network architecture (like old script)
        learning_rate = task_config.get('learning_rate', 1e-4)
        weight_decay = task_config.get('weight_decay', 1e-4)
        hidden_sizes = network_config.get('hidden_sizes', [256, 256])
        
        policy = MlpPolicy(
            observation_space=flat_obs_space,
            action_space=self.env.action_space,
            lr_schedule=lambda _: learning_rate,
            net_arch=hidden_sizes,
            optimizer_kwargs={
                "weight_decay": weight_decay,
                "eps": 1e-8,
            },
        )
        
        # Create BC trainer with custom policy
        self.model = bc.BC(
            observation_space=flat_obs_space,
            action_space=self.env.action_space,
            demonstrations=transitions,
            policy=policy,
            batch_size=task_config.get('batch_size', 64),
            rng=np.random.default_rng(self.config.get('seed', 42))
        )
        
        # Training - use the simple approach from old script
        n_epochs = task_config.get('epochs', 100)
        log_interval = task_config.get('log_interval', 10)
        
        self.logger.log_info(f"Starting BC training for {n_epochs} epochs...")
        self.logger.log_info(f"Training data: {len(transitions)} samples")
        self.logger.log_info(f"Hyperparameters: LR={learning_rate}, net_arch={hidden_sizes}, weight_decay={weight_decay}")
        
        try:
            # Use BC's built-in training with logging
            self.model.train(n_epochs=n_epochs, log_interval=log_interval)
            self.logger.log_info("BC training completed successfully")
            
        except Exception as e:
            self.logger.log_error(f"BC training failed: {str(e)}")
            raise e
        
        # Save final model
        self.save_model("bc_final.zip")
        self.logger.log_info("BC model saved")
    
    def train_ppo(self):
        """Train using PPO."""
        self.logger.log_info("Starting PPO training")
        
        # Get task-specific config
        task_config = self.config['training'].get(self.args.task, {})
        ppo_config = self.config.get('ppo', {})
        
        # Create PPO model with MlpPolicy for flattened observations (like old script)
        policy_type = 'MlpPolicy'  # For flattened observation spaces (matching old script)
        self.model = PPO(
            policy=policy_type,
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
        
        # Import custom PPO+BC implementation
        from dvrk_gym.algorithms.ppo_bc import PPOWithBCLoss
        from imitation.data import types as imitation_types
        
        # Load expert demonstrations for BC loss
        trajectories = self.load_demonstrations()
        
        # Convert trajectories to transitions for BC loss
        all_obs = []
        all_acts = []
        for traj in trajectories:
            # Handle trajectory format - could be dict or Trajectory object
            if hasattr(traj, 'obs'):
                obs = traj.obs[:-1]  # Exclude terminal observation  
                acts = traj.acts
            else:
                # Dictionary format from load_demonstrations
                obs = traj['observations'][:-1]  # Exclude terminal observation
                acts = traj['actions']
            all_obs.extend(obs)
            all_acts.extend(acts)
        
        all_obs = np.array(all_obs, dtype=np.float32)
        all_acts = np.array(all_acts, dtype=np.float32)
        
        # Create transitions object for PPOWithBCLoss
        # Create dummy arrays for required fields
        n_samples = len(all_obs)
        dummy_infos = [{} for _ in range(n_samples)]
        dummy_next_obs = np.zeros_like(all_obs)
        dummy_dones = np.zeros(n_samples, dtype=bool)
        
        expert_transitions = imitation_types.Transitions(
            obs=all_obs,
            acts=all_acts,
            infos=dummy_infos,
            next_obs=dummy_next_obs,
            dones=dummy_dones
        )
        
        self.logger.log_info(f"Loaded {len(expert_transitions.obs)} expert samples for BC loss")
        
        # Get configuration
        task_config = self.config['training'].get(self.args.task, {})
        network_config = self.config['network'].get(self.args.task, {})
        bc_config = self.config.get('bc', {})
        
        # Get BC-specific parameters
        bc_loss_weight_dict = bc_config.get('bc_loss_weight', {})
        bc_loss_weight = bc_loss_weight_dict.get(self.args.task, 0.01)
        bc_batch_size = bc_config.get('bc_batch_size', 256)
        bc_update_frequency = bc_config.get('bc_update_frequency', 1)
        use_bc_initialization = bc_config.get('use_bc_initialization', False)
        bc_model_path = bc_config.get('bc_model_path')
        
        # Create PPO+BC model with same parameters as PPO
        self.model = PPOWithBCLoss(
            policy='MlpPolicy',
            env=self.env,
            expert_demonstrations=expert_transitions,
            bc_loss_weight=bc_loss_weight,
            bc_batch_size=bc_batch_size,
            learning_rate=task_config.get('learning_rate', 3e-4),
            n_steps=task_config.get('n_steps', 2048),
            batch_size=task_config.get('batch_size', 64),
            n_epochs=task_config.get('n_epochs', 10),
            gamma=task_config.get('gamma', 0.99),
            gae_lambda=task_config.get('gae_lambda', 0.95),
            clip_range=task_config.get('clip_range', 0.2),
            clip_range_vf=task_config.get('clip_range_vf', None),
            normalize_advantage=task_config.get('normalize_advantage', True),
            ent_coef=task_config.get('ent_coef', 0.0),
            vf_coef=task_config.get('vf_coef', 0.5),
            max_grad_norm=task_config.get('max_grad_norm', 0.5),
            use_sde=task_config.get('use_sde', False),
            sde_sample_freq=task_config.get('sde_sample_freq', -1),
            policy_kwargs=dict(
                net_arch=network_config.get('hidden_sizes', [256, 256])
            ),
            verbose=self.config['logging'].get('verbose', 1),
            seed=self.config.get('seed', 42),
            device='auto',
            tensorboard_log=self.logger.tensorboard_dir if not self.args.no_tensorboard else None
        )
        
        # Optionally initialize with BC model
        if use_bc_initialization and bc_model_path:
            if Path(bc_model_path).exists():
                self.logger.log_info(f"Initializing PPO+BC with BC model: {bc_model_path}")
                try:
                    # Load BC model weights
                    import pickle
                    with open(bc_model_path, 'rb') as f:
                        bc_data = pickle.load(f)
                    # Initialize policy network with BC weights
                    # This would require extracting weights from BC model
                    # For now, we'll skip this step
                    self.logger.log_info("BC initialization not fully implemented yet")
                except Exception as e:
                    self.logger.log_info(f"Could not load BC model: {e}")
            else:
                self.logger.log_info(f"BC model not found at {bc_model_path}, starting from scratch")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        if self.config['logging'].get('save_freq', 10000) > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config['logging']['save_freq'],
                save_path=str(self.model_dir),
                name_prefix='ppo_bc_checkpoint'
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.config['logging'].get('eval_freq', 5000) > 0:
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
        self.logger.log_info(f"Training PPO+BC for {total_timesteps} timesteps")
        self.logger.log_info(f"BC loss weight: {bc_loss_weight}")
        self.logger.log_info(f"BC batch size: {bc_batch_size}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        self.save_model("ppo_bc_final.zip")
        self.logger.log_info("PPO+BC training completed")
    
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