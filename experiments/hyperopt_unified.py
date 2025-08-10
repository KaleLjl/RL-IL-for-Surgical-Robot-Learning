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
import tempfile

import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from utils.config_parser import ConfigParser


class HyperparameterOptimizer:
    """Unified hyperparameter optimizer using Optuna."""
    
    def __init__(self, 
                 algorithm: str,
                 task: str,
                 stage: str = "bc",
                 n_trials: int = 50,
                 study_name: Optional[str] = None,
                 bc_model_path: Optional[str] = None,
                 output_dir: str = "results/hyperopt"):
        """Initialize hyperparameter optimizer.
        
        Args:
            algorithm: Algorithm to optimize ('bc', 'ppo', 'ppo_bc')
            task: Task name ('needle_reach', 'peg_transfer')
            stage: Optimization stage ('bc', 'ppo_bc', 'sequential')
            n_trials: Number of optimization trials
            study_name: Custom study name
            bc_model_path: Path to BC model for ppo_bc stage
            output_dir: Output directory for results
        """
        self.algorithm = algorithm
        self.task = task
        self.stage = stage
        self.n_trials = n_trials
        self.bc_model_path = bc_model_path
        self.output_dir = Path(output_dir)
        
        # Create study name
        if study_name:
            self.study_name = study_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.study_name = f"{algorithm}_{task}_{stage}_{timestamp}"
        
        # Create output directory
        self.study_dir = self.output_dir / self.study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = optuna.create_study(
            direction='maximize',  # Maximize success rate
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print(f"Initialized Optuna study: {self.study_name}")
        print(f"Output directory: {self.study_dir}")
    
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
                [128, 128], [256, 256], [512, 512],
                [128, 256, 128], [256, 512, 256]
            ]
            max_epochs = 300
            base_lr_range = (1e-5, 1e-3)
            
        elif task == 'peg_transfer':
            hidden_sizes_options = [
                [64, 64], [128, 128], [256, 256],
                [64, 128, 64], [128, 256, 128]
            ]
            max_epochs = 50
            base_lr_range = (1e-6, 1e-4)
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Network architecture
        hidden_sizes = trial.suggest_categorical('hidden_sizes', hidden_sizes_options)
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
            hidden_sizes_options = [[128, 128], [256, 256], [512, 512]]
            
        elif task == 'peg_transfer':
            # Match successful old script: 300k timesteps exactly  
            total_timesteps_range = (300000, 300000)  # Fixed to 300k like old script
            hidden_sizes_options = [[64, 64], [128, 128], [256, 256]]
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Network architecture
        hidden_sizes = trial.suggest_categorical('hidden_sizes', hidden_sizes_options)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        
        # PPO hyperparameters - focused around old script proven values + faster optimization
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)  # Narrowed around 3e-4
        n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Favor smaller batch sizes
        n_epochs = trial.suggest_int('n_epochs', 5, 20)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        
        # Reduced timesteps for faster optimization (20-50k instead of 50-100k)
        total_timesteps = trial.suggest_int('total_timesteps', 20000, 50000, step=10000)
        
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
        """Get PPO+BC hybrid search space."""
        
        # Start with PPO search space
        config = self._get_ppo_search_space(trial, task)
        config['algorithm'] = 'ppo_bc'
        
        # Add BC-specific parameters
        bc_loss_weight = trial.suggest_float('bc_loss_weight', 0.001, 0.1, log=True)
        bc_update_frequency = trial.suggest_int('bc_update_frequency', 1, 10)
        
        # Add BC configuration
        config['bc'] = {
            'bc_loss_weight': {task: bc_loss_weight},
            'bc_update_frequency': bc_update_frequency,
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
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2)
                temp_config_path = f.name
            
            try:
                # Create experiment name for this trial
                experiment_name = f"{self.study_name}_trial_{trial.number:03d}"
                
                # Run training
                train_cmd = [
                    'python3', 'train_unified.py',
                    '--config', temp_config_path,
                    '--task', self.task,
                    '--algorithm', self.algorithm,
                    '--experiment-name', experiment_name,
                    '--no-tensorboard'
                ]
                
                print(f"\n{'='*50}")
                print(f"TRIAL {trial.number}: {self.algorithm.upper()} on {self.task}")
                print(f"{'='*50}")
                print("Parameters:")
                for key, value in trial.params.items():
                    print(f"  {key}: {value}")
                print(f"{'='*50}")
                
                # Run training with timeout (increased for PPO)
                training_timeout = 14400 if self.algorithm == 'ppo' else 7200  # 4 hours for PPO, 2 hours for BC
                result = subprocess.run(
                    train_cmd,
                    capture_output=True,
                    text=True,
                    timeout=training_timeout,
                    cwd=Path(__file__).parent
                )
                
                if result.returncode != 0:
                    print(f"Training failed for trial {trial.number}")
                    print(f"Error: {result.stderr}")
                    return 0.0
                
                # Find the experiment directory
                exp_dir = Path(f"results/experiments/{experiment_name}")
                if not exp_dir.exists():
                    print(f"Experiment directory not found: {exp_dir}")
                    return 0.0
                
                # Run evaluation
                model_path = exp_dir / 'models' / f'{self.algorithm}_final.zip'
                if not model_path.exists():
                    print(f"Model not found: {model_path}")
                    return 0.0
                
                eval_cmd = [
                    'python3', 'evaluate_unified.py',
                    '--model', str(model_path),
                    '--task', self.task,
                    '--algorithm', self.algorithm,
                    '--n-episodes', '50',  # Faster evaluation during optimization
                    '--output-dir', str(exp_dir / 'evaluations')
                ]
                
                eval_result = subprocess.run(
                    eval_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=Path(__file__).parent
                )
                
                if eval_result.returncode != 0:
                    print(f"Evaluation failed for trial {trial.number}")
                    print(f"Error: {eval_result.stderr}")
                    return 0.0
                
                # Load evaluation results
                eval_file = exp_dir / 'evaluations' / f'evaluation_{self.algorithm}_{self.task}.json'
                if not eval_file.exists():
                    print(f"Evaluation file not found: {eval_file}")
                    return 0.0
                
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                success_rate = eval_data['metrics']['success_rate']
                mean_reward = eval_data['metrics']['mean_reward']
                
                print(f"Trial {trial.number} Results:")
                print(f"  Success Rate: {success_rate:.3f}")
                print(f"  Mean Reward: {mean_reward:.2f}")
                
                # Store additional metrics in trial user attributes
                trial.set_user_attr('mean_reward', mean_reward)
                trial.set_user_attr('experiment_name', experiment_name)
                trial.set_user_attr('model_path', str(model_path))
                
                return success_rate
                
            finally:
                # Clean up temporary config file
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)
                
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
        print(f"Stage: {self.stage}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Study name: {self.study_name}")
        
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
        
        return best_params, best_score
    
    def save_results(self, best_trial: optuna.trial.FrozenTrial):
        """Save optimization results."""
        
        # Save study
        study_file = self.study_dir / 'study.json'
        study_data = {
            'study_name': self.study_name,
            'algorithm': self.algorithm,
            'task': self.task,
            'stage': self.stage,
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
        
        config_file = self.study_dir / f'optimal_config_{self.algorithm}_{self.task}.yaml'
        ConfigParser.save_config(optimal_config, config_file)
        
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


def sequential_optimization(task: str, bc_trials: int = 50, ppo_bc_trials: int = 30, output_dir: str = "results/hyperopt"):
    """Run sequential optimization: BC first, then PPO+BC.
    
    Args:
        task: Task name ('needle_reach', 'peg_transfer')
        bc_trials: Number of trials for BC optimization
        ppo_bc_trials: Number of trials for PPO+BC optimization
        output_dir: Output directory
    """
    print(f"\n{'='*60}")
    print("SEQUENTIAL HYPERPARAMETER OPTIMIZATION")
    print(f"Task: {task}")
    print(f"BC Trials: {bc_trials}")
    print(f"PPO+BC Trials: {ppo_bc_trials}")
    print(f"{'='*60}")
    
    # Stage 1: Optimize BC
    print(f"\n{'='*30}")
    print("STAGE 1: BC OPTIMIZATION")
    print(f"{'='*30}")
    
    bc_optimizer = HyperparameterOptimizer(
        algorithm='bc',
        task=task,
        stage='bc',
        n_trials=bc_trials,
        output_dir=output_dir
    )
    
    bc_params, bc_score = bc_optimizer.optimize()
    
    # Get best BC model path
    best_bc_trial = bc_optimizer.study.best_trial
    bc_model_path = best_bc_trial.user_attrs.get('model_path')
    
    if not bc_model_path or not Path(bc_model_path).exists():
        print("ERROR: Best BC model not found. Cannot proceed to PPO+BC optimization.")
        return
    
    print(f"\nBC optimization completed. Best model: {bc_model_path}")
    
    # Stage 2: Optimize PPO+BC using best BC model
    print(f"\n{'='*30}")
    print("STAGE 2: PPO+BC OPTIMIZATION")
    print(f"{'='*30}")
    
    ppo_bc_optimizer = HyperparameterOptimizer(
        algorithm='ppo_bc',
        task=task,
        stage='ppo_bc',
        n_trials=ppo_bc_trials,
        bc_model_path=bc_model_path,
        output_dir=output_dir
    )
    
    ppo_bc_params, ppo_bc_score = ppo_bc_optimizer.optimize()
    
    # Generate summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = Path(output_dir) / f"sequential_summary_{task}_{timestamp}.json"
    
    summary = {
        'task': task,
        'timestamp': timestamp,
        'bc_optimization': {
            'trials': bc_trials,
            'best_score': bc_score,
            'best_params': bc_params,
            'study_name': bc_optimizer.study_name
        },
        'ppo_bc_optimization': {
            'trials': ppo_bc_trials,
            'best_score': ppo_bc_score,
            'best_params': ppo_bc_params,
            'study_name': ppo_bc_optimizer.study_name,
            'bc_model_path': bc_model_path
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SEQUENTIAL OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"BC Best Score: {bc_score:.3f}")
    print(f"PPO+BC Best Score: {ppo_bc_score:.3f}")
    print(f"Improvement: {ppo_bc_score - bc_score:.3f}")
    print(f"Summary saved to: {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for surgical robot learning')
    
    parser.add_argument('--algorithm', type=str, choices=['bc', 'ppo', 'ppo_bc'], required=True,
                        help='Algorithm to optimize')
    parser.add_argument('--task', type=str, choices=['needle_reach', 'peg_transfer'], required=True,
                        help='Task to optimize for')
    parser.add_argument('--stage', type=str, choices=['bc', 'ppo_bc', 'sequential'], default='bc',
                        help='Optimization stage')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Custom study name')
    parser.add_argument('--bc-model-path', type=str, default=None,
                        help='Path to BC model for ppo_bc stage')
    parser.add_argument('--output-dir', type=str, default='results/hyperopt',
                        help='Output directory')
    parser.add_argument('--bc-trials', type=int, default=50,
                        help='Number of BC trials for sequential optimization')
    parser.add_argument('--ppo-bc-trials', type=int, default=30,
                        help='Number of PPO+BC trials for sequential optimization')
    
    args = parser.parse_args()
    
    if args.stage == 'sequential':
        sequential_optimization(
            task=args.task,
            bc_trials=args.bc_trials,
            ppo_bc_trials=args.ppo_bc_trials,
            output_dir=args.output_dir
        )
    else:
        if args.algorithm == 'ppo_bc' and not args.bc_model_path:
            parser.error('--bc-model-path is required for ppo_bc algorithm')
        
        optimizer = HyperparameterOptimizer(
            algorithm=args.algorithm,
            task=args.task,
            stage=args.stage,
            n_trials=args.n_trials,
            study_name=args.study_name,
            bc_model_path=args.bc_model_path,
            output_dir=args.output_dir
        )
        
        optimizer.optimize()


if __name__ == '__main__':
    main()