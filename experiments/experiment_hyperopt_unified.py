#!/usr/bin/env python3
"""Unified hyperparameter optimization using Optuna for all algorithms."""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
# import tempfile  # No longer needed

def is_in_docker():
    """Check if the script is running inside a Docker container."""
    return os.path.exists('/.dockerenv') or os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read()

# Fix matplotlib warning
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

# No longer need config parser since we use simplified training scripts


class HyperparameterOptimizer:
    """Unified hyperparameter optimizer using Optuna."""
    
    def __init__(self, 
                 algorithm: str,
                 task: str,
                 # stage parameter removed - now just algorithm-specific
                 n_trials: int = 50,
                 # study_name removed - results go directly to results_dir
                 bc_model_path: Optional[str] = None,
                 trials_dir: str = "hyperopt_trials",
                 results_dir: str = "hyperopt_results"):
        """Initialize hyperparameter optimizer.
        
        Args:
            algorithm: Algorithm to optimize ('bc', 'ppo', 'ppo_bc')
            task: Task name ('needle_reach', 'peg_transfer')
            # stage parameter removed - optimization is algorithm-specific
            n_trials: Number of optimization trials
            # study_name removed
            bc_model_path: Path to BC model for ppo_bc stage
            trials_dir: Directory for individual trial experiments
            results_dir: Directory for final analysis files
        """
        self.algorithm = algorithm
        self.task = task
        # self.stage = stage  # No longer needed
        self.n_trials = n_trials
        self.bc_model_path = bc_model_path
        self.trials_dir = Path(trials_dir)  # For individual trial results
        self.results_dir = Path(results_dir)        # For final analysis files
        
        # Create study name for Optuna (internal use only)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.study_name = f"{algorithm}_{task}_{timestamp}"
        
        # Create both directories
        self.study_dir = self.results_dir  # Results go directly into results_dir
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)  # Ensure trials dir exists
        
        # Initialize study
        self.study = optuna.create_study(
            direction='maximize',  # Maximize success rate
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print(f"Initialized Optuna study: {self.study_name}")
        print(f"Results directory: {self.study_dir}")
        print(f"Trials directory: {self.trials_dir}")
    
    def get_search_space(self, trial: optuna.trial.Trial, algorithm: str, task: str) -> Dict[str, Any]:
        """Define search space for hyperparameters.
        
        Args:
            trial: Optuna trial object
            algorithm: Algorithm name
            task: Task name
            
        Returns:
            Dictionary of hyperparameters to test
        """
        config = {}
        
        if algorithm == 'bc':
            # BC-specific hyperparameters
            config.update(self._get_bc_search_space(trial, task))
            
        elif algorithm == 'ppo':
            # PPO-specific hyperparameters
            config.update(self._get_ppo_search_space(trial, task))
            
        elif algorithm == 'ppo_bc':
            # PPO+BC hybrid hyperparameters
            config.update(self._get_ppo_bc_search_space(trial, task))
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return config
    
    def _get_bc_search_space(self, trial: optuna.trial.Trial, task: str) -> Dict[str, Any]:
        """Get BC-specific search space."""
        
        # Task-specific parameter ranges
        if task == 'needle_reach':
            hidden_sizes_options = [
                "128,128", "256,256", "512,512",
                "128,256,128", "256,512,256"
            ]
            max_epochs = 300
            base_lr_range = (1e-5, 1e-3)
            
        elif task == 'peg_transfer':
            hidden_sizes_options = [
                "64,64", "128,128", "256,256",
                "64,128,64", "128,256,128"
            ]
            max_epochs = 50
            base_lr_range = (1e-6, 1e-4)
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Network architecture
        hidden_sizes_str = trial.suggest_categorical('hidden_sizes', hidden_sizes_options)
        hidden_sizes = [int(x) for x in hidden_sizes_str.split(',')]
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', *base_lr_range, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        epochs = trial.suggest_int('epochs', max_epochs // 4, max_epochs)
        
        # Early stopping
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
        
        return {
            'algorithm': 'bc',
            'network': {
                task: {
                    'hidden_sizes': hidden_sizes,
                    'activation': activation
                }
            },
            'training': {
                task: {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'early_stopping_patience': early_stopping_patience,
                    'early_stopping_threshold': 0.001
                }
            },
            'data': {
                'num_demonstrations': 100
            },
            'logging': {
                'tensorboard': False,  # Disable for speed during optimization
                'csv_logging': True,
                'save_freq': epochs // 5,
                'verbose': 0
            }
        }
    
    def _get_ppo_search_space(self, trial: optuna.trial.Trial, task: str) -> Dict[str, Any]:
        """Get PPO-specific search space."""
        
        if task == 'needle_reach':
            # Match successful old script: 100k timesteps exactly
            total_timesteps_range = (100000, 100000)  # Fixed to 100k like old script
            hidden_sizes_options = ["128,128", "256,256", "512,512"]
            
        elif task == 'peg_transfer':
            # Match successful old script: 300k timesteps exactly  
            total_timesteps_range = (300000, 300000)  # Fixed to 300k like old script
            hidden_sizes_options = ["64,64", "128,128", "256,256"]
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Network architecture
        hidden_sizes_str = trial.suggest_categorical('hidden_sizes', hidden_sizes_options)
        hidden_sizes = [int(x) for x in hidden_sizes_str.split(',')]
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        
        # PPO hyperparameters - focused around old script proven values + faster optimization
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)  # Narrowed around 3e-4
        n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Favor smaller batch sizes
        n_epochs = trial.suggest_int('n_epochs', 5, 20)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        
        # Use 100k timesteps for fair comparison (matching old script)
        total_timesteps = trial.suggest_int('total_timesteps', 100000, 100000, step=10000)
        
        return {
            'algorithm': 'ppo',
            'network': {
                task: {
                    'hidden_sizes': hidden_sizes,
                    'activation': activation
                }
            },
            'training': {
                task: {
                    'total_timesteps': total_timesteps,
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'gamma': gamma,
                    'clip_range': clip_range,
                    'ent_coef': ent_coef
                }
            },
            'logging': {
                'tensorboard': False,
                'csv_logging': True,
                'save_freq': 10000,
                'verbose': 0
            }
        }
    
    def _get_ppo_bc_search_space(self, trial: optuna.trial.Trial, task: str) -> Dict[str, Any]:
        """Get PPO+BC hybrid search space - focused around old DAPG script values."""
        
        # PPO+BC hyperparameters focused around old script proven values
        if task == 'needle_reach':
            # Old script: timesteps=300000, bc_weight=0.05
            total_timesteps = trial.suggest_categorical('total_timesteps', [250000, 300000, 350000])
            bc_loss_weight = trial.suggest_float('bc_loss_weight', 0.02, 0.08)  # Around 0.05
            hidden_sizes_options = ["512,512"]  # Match best BC config exactly
            
        elif task == 'peg_transfer':
            # Old script: timesteps=500000, bc_weight=0.02  
            total_timesteps = trial.suggest_categorical('total_timesteps', [400000, 500000, 600000])
            bc_loss_weight = trial.suggest_float('bc_loss_weight', 0.01, 0.04)  # Around 0.02
            hidden_sizes_options = ["128,128"]  # Match peg_transfer BC config
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Network architecture - match BC exactly
        hidden_sizes_str = trial.suggest_categorical('hidden_sizes', hidden_sizes_options)
        hidden_sizes = [int(x) for x in hidden_sizes_str.split(',')]
        activation = trial.suggest_categorical('activation', ['relu'])  # BC used relu
        
        # PPO hyperparameters from old script
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 2e-4, log=True)  # Around 1e-4
        n_steps = trial.suggest_categorical('n_steps', [3072, 4096, 5120])  # Around 4096
        batch_size = trial.suggest_categorical('batch_size', [192, 256, 320])  # Around 256
        n_epochs = trial.suggest_int('n_epochs', 4, 6)  # Around 5
        gamma = trial.suggest_float('gamma', 0.98, 0.995)  # Around 0.99
        clip_range = trial.suggest_float('clip_range', 0.15, 0.25)  # Around 0.2
        ent_coef = trial.suggest_float('ent_coef', 0.0, 0.01)  # Around 0.0
        
        # BC-specific parameters from old script
        bc_update_frequency = trial.suggest_int('bc_update_frequency', 1, 1)  # Always 1 (every step)
        bc_batch_size = trial.suggest_categorical('bc_batch_size', [1024])  # Fixed from old script
        
        # Create complete config
        config = {
            'algorithm': 'ppo_bc',
            'network': {
                task: {
                    'hidden_sizes': hidden_sizes,
                    'activation': activation
                }
            },
            'training': {
                task: {
                    'total_timesteps': total_timesteps,
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'gamma': gamma,
                    'clip_range': clip_range,
                    'ent_coef': ent_coef,
                    # Additional old script parameters
                    'gae_lambda': 0.95,  # From old script
                    'vf_coef': 0.5,      # From old script
                    'max_grad_norm': 0.5, # From old script
                }
            },
            'logging': {
                'tensorboard': False,
                'csv_logging': True,
                'save_freq': 10000,
                'verbose': 1  # More verbose like old script
            }
        }
        
        # Add BC configuration
        config['bc'] = {
            'bc_loss_weight': {task: bc_loss_weight},
            'bc_update_frequency': bc_update_frequency,
            'bc_batch_size': bc_batch_size,
            'use_bc_initialization': True,
            'bc_model_path': self.bc_model_path
        }
        
        return config
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function to optimize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Success rate (to be maximized)
        """
        try:
            # Get hyperparameters for this trial
            config = self.get_search_space(trial, self.algorithm, self.task)
            config['seed'] = trial.suggest_int('seed', 1, 1000)
            
            # Create temporary config file
            try:
                # Create experiment name for this trial
                experiment_name = f"{self.study_name}_trial_{trial.number:03d}"
                
                # Determine which training script to use
                if self.algorithm == 'bc':
                    train_script = 'experiments/train_bc.py'
                    env_name = 'NeedleReach-v0' if self.task == 'needle_reach' else 'PegTransfer-v0'
                    base_train_cmd = [
                        'python3', train_script,
                        '--env', env_name,
                        '--output-dir', str(self.trials_dir / experiment_name)
                    ]
                    # Wrap with Docker if running outside container
                    train_cmd = base_train_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_train_cmd
                elif self.algorithm == 'ppo':
                    train_script = 'experiments/train_ppo.py'
                    env_name = 'NeedleReach-v0' if self.task == 'needle_reach' else 'PegTransfer-v0'
                    base_train_cmd = [
                        'python3', train_script,
                        '--env', env_name,
                        '--output-dir', str(self.trials_dir / experiment_name)
                    ]
                    # Wrap with Docker if running outside container
                    train_cmd = base_train_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_train_cmd
                elif self.algorithm == 'ppo_bc' or self.algorithm == 'ppo_il':
                    train_script = 'experiments/train_ppo+il.py'
                    env_name = 'NeedleReach-v0' if self.task == 'needle_reach' else 'PegTransfer-v0'
                    if not self.bc_model_path:
                        raise ValueError("BC model path required for PPO+IL training")
                    base_train_cmd = [
                        'python3', train_script,
                        '--env', env_name,
                        '--bc-model', str(self.bc_model_path),
                        '--expert-data', 'experiments/data/expert_data_needle_reach.pkl' if self.task == 'needle_reach' else 'experiments/data/expert_data_peg_transfer.pkl',
                        '--output-dir', str(self.trials_dir / experiment_name)
                    ]
                    # Wrap with Docker if running outside container
                    train_cmd = base_train_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_train_cmd
                else:
                    raise ValueError(f"Unknown algorithm: {self.algorithm}")
                
                print(f"\n{'='*50}")
                print(f"TRIAL {trial.number}: {self.algorithm.upper()} on {self.task}")
                print(f"{'='*50}")
                print("Parameters:")
                for key, value in trial.params.items():
                    print(f"  {key}: {value}")
                print(f"{'='*50}")
                
                # Run training with timeout (increased for PPO) and show output
                training_timeout = 14400 if self.algorithm == 'ppo' else 7200  # 4 hours for PPO, 2 hours for BC
                print(f"\nRunning training command: {' '.join(train_cmd)}")
                print(f"Training logs will be shown below:")
                print("-" * 60)
                
                result = subprocess.run(
                    train_cmd,
                    timeout=training_timeout,
                    cwd=os.getcwd()  # Use current working directory where script was invoked
                )
                
                print("-" * 60)
                if result.returncode != 0:
                    print(f"Training failed for trial {trial.number} with return code {result.returncode}")
                    return 0.0
                
                print(f"Training completed successfully for trial {trial.number}")
                
                # Find the trained model
                trial_dir = self.trials_dir / experiment_name
                if not trial_dir.exists():
                    print(f"Trial directory not found: {trial_dir}")
                    return 0.0
                
                # Find the model file (should be the .zip file in the trial directory)
                model_files = list(trial_dir.glob('*.zip'))
                if not model_files:
                    print(f"No model file found in {trial_dir}")
                    return 0.0
                model_path = model_files[0]  # Take the first .zip file
                
                # Determine which evaluation script to use
                eval_output_dir = trial_dir / "evaluation"
                if self.algorithm == 'bc':
                    eval_script = 'experiments/evaluate_bc.py'
                    base_eval_cmd = [
                        'python3', eval_script,
                        '--model', str(model_path),
                        '--episodes', '50',  # Faster evaluation during optimization
                        '--no-render',
                        '--output-dir', str(eval_output_dir),
                        '--action-noise-test'  # Enable robustness testing
                    ]
                    # Wrap with Docker if running outside container
                    eval_cmd = base_eval_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_eval_cmd
                elif self.algorithm == 'ppo':
                    eval_script = 'experiments/evaluate_ppo.py'
                    base_eval_cmd = [
                        'python3', eval_script,
                        '--model', str(model_path),
                        '--episodes', '50',
                        '--no-render',
                        '--output-dir', str(eval_output_dir)
                    ]
                    # Wrap with Docker if running outside container
                    eval_cmd = base_eval_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_eval_cmd
                elif self.algorithm == 'ppo_bc' or self.algorithm == 'ppo_il':
                    eval_script = 'experiments/evaluate_ppo_il.py'
                    base_eval_cmd = [
                        'python3', eval_script,
                        '--model', str(model_path),
                        '--episodes', '50',
                        '--no-render',
                        '--output-dir', str(eval_output_dir)
                    ]
                    # Wrap with Docker if running outside container
                    eval_cmd = base_eval_cmd if is_in_docker() else ['docker', 'exec', 'dvrk_dev'] + base_eval_cmd
                else:
                    raise ValueError(f"Unknown algorithm: {self.algorithm}")
                
                print(f"\nRunning evaluation command: {' '.join(eval_cmd)}")
                print(f"Evaluation logs:")
                print("-" * 40)
                
                eval_result = subprocess.run(
                    eval_cmd,
                    timeout=600,  # 10 minute timeout
                    cwd=os.getcwd()  # Use current working directory where script was invoked
                )
                
                print("-" * 40)
                if eval_result.returncode != 0:
                    print(f"Evaluation failed for trial {trial.number} with return code {eval_result.returncode}")
                    return 0.0
                
                print(f"Evaluation completed successfully for trial {trial.number}")
                
                # Load evaluation results
                if self.algorithm == 'bc':
                    eval_file = eval_output_dir / 'evaluation_results.json'
                elif self.algorithm == 'ppo':
                    eval_file = eval_output_dir / 'ppo_evaluation_results.json'
                elif self.algorithm == 'ppo_bc' or self.algorithm == 'ppo_il':
                    eval_file = eval_output_dir / 'ppo_il_evaluation_results.json'
                
                if not eval_file.exists():
                    print(f"Evaluation file not found: {eval_file}")
                    return 0.0
                
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                # Use composite score for BC (includes robustness), standard success rate for others
                if self.algorithm == 'bc':
                    success_rate = eval_data['composite_score'] / 100.0  # Use composite score for BC
                else:
                    success_rate = eval_data['success_rate'] / 100.0  # Standard for PPO/PPO+IL
                mean_reward = eval_data['average_reward']
                
                print(f"Trial {trial.number} Results:")
                if self.algorithm == 'bc':
                    standard_success = eval_data.get('success_rate', 0)
                    composite_score = eval_data.get('composite_score', 0)
                    print(f"  Standard Success Rate: {standard_success:.1f}%")
                    print(f"  Composite Score (used): {composite_score:.1f}%")
                    if 'action_noise_test' in eval_data and eval_data['action_noise_test']['enabled']:
                        noise_success = eval_data['action_noise_test']['success_rate']
                        print(f"  Action Noise Success Rate: {noise_success:.1f}%")
                else:
                    print(f"  Success Rate: {success_rate:.3f}")
                print(f"  Mean Reward: {mean_reward:.2f}")
                print(f"  📁 Trial data saved to: {trial_dir}")
                print(f"  🤖 Model saved to: {model_path}")
                print(f"  📊 Evaluation results: {eval_file}")
                
                # Store additional metrics in trial user attributes
                trial.set_user_attr('mean_reward', mean_reward)
                trial.set_user_attr('experiment_name', experiment_name)
                trial.set_user_attr('model_path', str(model_path))
                
                return success_rate
                
            finally:
                # Clean up no longer needed with simplified training scripts
                pass
                
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            return 0.0
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Run hyperparameter optimization.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"\nStarting hyperparameter optimization:")
        print(f"Algorithm: {self.algorithm}")
        print(f"Task: {self.task}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Study name: {self.study_name}")
        print(f"Execution mode: {'Inside Docker container' if is_in_docker() else 'Outside Docker (using docker exec)'}")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        print(f"\n{'='*50}")
        print("OPTIMIZATION COMPLETED")
        print(f"{'='*50}")
        print(f"Best Success Rate: {best_score:.3f}")
        print(f"Best Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        self.save_results(best_trial)
        self.generate_visualizations()
        
        print(f"\n📋 Study results saved to: {self.study_dir}")
        print(f"📈 Visualizations saved to: {self.study_dir}")
        print(f"🏆 Best model located at: {best_trial.user_attrs.get('model_path', 'N/A')}")
        
        return best_params, best_score
    
    def save_results(self, best_trial: optuna.trial.FrozenTrial):
        """Save optimization results."""
        
        # Save study
        study_file = self.study_dir / 'study.json'
        study_data = {
            'study_name': self.study_name,
            'algorithm': self.algorithm,
            'task': self.task,
            'n_trials': self.n_trials,
            'best_trial': {
                'number': best_trial.number,
                'value': best_trial.value,
                'params': best_trial.params,
                'user_attrs': best_trial.user_attrs
            },
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': str(trial.state),
                    'user_attrs': trial.user_attrs
                }
                for trial in self.study.trials
            ]
        }
        
        with open(study_file, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        # Create optimal config file
        optimal_config = self.get_search_space(best_trial, self.algorithm, self.task)
        optimal_config['seed'] = best_trial.params['seed']
        
        config_file = self.study_dir / f'optimal_config_{self.algorithm}_{self.task}.json'
        with open(config_file, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        # Save trials DataFrame
        df = self.study.trials_dataframe()
        df.to_csv(self.study_dir / 'trials.csv', index=False)
        
        print(f"\nResults saved to: {self.study_dir}")
        print(f"Optimal config: {config_file}")
    
    def generate_visualizations(self):
        """Generate optimization visualizations."""
        try:
            import plotly.offline as pyo
            
            # Optimization history
            fig_history = plot_optimization_history(self.study)
            fig_history.update_layout(title=f'Optimization History - {self.algorithm.upper()} on {self.task}')
            pyo.plot(fig_history, filename=str(self.study_dir / 'optimization_history.html'), auto_open=False)
            
            # Parameter importance
            if len(self.study.trials) > 10:
                fig_importance = plot_param_importances(self.study)
                fig_importance.update_layout(title=f'Parameter Importance - {self.algorithm.upper()} on {self.task}')
                pyo.plot(fig_importance, filename=str(self.study_dir / 'parameter_importance.html'), auto_open=False)
            
            # Parallel coordinate plot
            if len(self.study.trials) > 5:
                fig_parallel = plot_parallel_coordinate(self.study)
                fig_parallel.update_layout(title=f'Parallel Coordinates - {self.algorithm.upper()} on {self.task}')
                pyo.plot(fig_parallel, filename=str(self.study_dir / 'parallel_coordinates.html'), auto_open=False)
            
            print(f"Visualizations saved to: {self.study_dir}")
            
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")


# Sequential optimization removed - users will manually select best BC model for PPO+IL optimization


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for surgical robot learning')
    
    parser.add_argument('--algorithm', type=str, choices=['bc', 'ppo', 'ppo_bc'], required=True,
                        help='Algorithm to optimize')
    parser.add_argument('--env', type=str, choices=['NeedleReach-v0', 'PegTransfer-v0'], required=True,
                        help='Environment to optimize for')
    # --stage removed - algorithm determines optimization type
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials')
    # --study-name removed - results go directly to --results-dir
    parser.add_argument('--bc-model-path', type=str, default=None,
                        help='Path to BC model for ppo_bc stage')
    parser.add_argument('--trials-dir', type=str, default='hyperopt_trials',
                        help='Directory for individual trial experiments')
    parser.add_argument('--results-dir', type=str, default='hyperopt_results',
                        help='Directory for final analysis files')
    # Sequential parameters removed - no longer needed
    
    args = parser.parse_args()
    
    if args.algorithm == 'ppo_bc' and not args.bc_model_path:
        parser.error('--bc-model-path is required for ppo_bc algorithm')
    
    # Convert env name to task name for internal use
    task = 'needle_reach' if args.env == 'NeedleReach-v0' else 'peg_transfer'
    
    optimizer = HyperparameterOptimizer(
        algorithm=args.algorithm,
        task=task,
        n_trials=args.n_trials,
        # study_name parameter removed
        bc_model_path=args.bc_model_path,
        trials_dir=args.trials_dir,
        results_dir=args.results_dir
    )
    
    optimizer.optimize()


if __name__ == '__main__':
    main()