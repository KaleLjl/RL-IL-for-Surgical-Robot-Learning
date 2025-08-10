#!/usr/bin/env python3
"""Generate PPO hyperparameter optimization analysis for hyperopt_phase3."""

import json
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Try to import optional packages
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def collect_ppo_results(results_dir: Path) -> Dict[str, Any]:
    """Collect all PPO trial results."""
    trials = []
    trial_dirs = sorted(results_dir.glob("ppo_needle_reach_final_trial_*"))
    
    print(f"Found {len(trial_dirs)} PPO trials")
    
    for trial_dir in trial_dirs:
        trial_num = int(trial_dir.name.split('_')[-1])
        
        # Load config
        config_file = trial_dir / "config.json"
        eval_file = trial_dir / "evaluations" / "evaluation_ppo_needle_reach.json"
        
        if not config_file.exists() or not eval_file.exists():
            print(f"Skipping incomplete trial {trial_num}")
            continue
            
        # Load configuration
        with open(config_file) as f:
            config = json.load(f)
            
        # Load evaluation results
        with open(eval_file) as f:
            eval_results = json.load(f)
        
        # Extract metrics from nested structure
        metrics = eval_results.get('metrics', {})
        
        # Extract hyperparameters
        training_config = config['training']['needle_reach']
        network_config = config['network']['needle_reach']
        
        trial_data = {
            'trial_number': trial_num,
            'success_rate': metrics.get('success_rate', 0.0),
            'mean_reward': metrics.get('mean_reward', 0.0),
            'std_reward': metrics.get('std_reward', 0.0),
            'mean_episode_length': metrics.get('mean_length', 0.0),
            # Hyperparameters
            'learning_rate': training_config.get('learning_rate', 0.0),
            'n_steps': training_config.get('n_steps', 0),
            'batch_size': training_config.get('batch_size', 0),
            'n_epochs': training_config.get('n_epochs', 0),
            'gamma': training_config.get('gamma', 0.0),
            'clip_range': training_config.get('clip_range', 0.0),
            'ent_coef': training_config.get('ent_coef', 0.0),
            'total_timesteps': training_config.get('total_timesteps', 0),
            'hidden_sizes': network_config.get('hidden_sizes', []),
            'activation': network_config.get('activation', 'relu'),
            'seed': config.get('seed', 0)
        }
        
        trials.append(trial_data)
        print(f"Trial {trial_num}: Success Rate = {trial_data['success_rate']:.1%}")
    
    return {
        'trials': trials,
        'n_trials': len(trials),
        'best_trial': max(trials, key=lambda x: x['success_rate']) if trials else None
    }

def generate_study_json(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate Optuna-style study.json file."""
    trials = results['trials']
    best_trial = results['best_trial']
    
    if not best_trial:
        return
        
    study_data = {
        'study_name': 'ppo_needle_reach_hyperopt',
        'algorithm': 'ppo',
        'task': 'needle_reach', 
        'stage': 'ppo',
        'n_trials': len(trials),
        'best_trial': {
            'number': best_trial['trial_number'],
            'value': best_trial['success_rate'],
            'params': {
                'hidden_sizes': best_trial['hidden_sizes'],
                'activation': best_trial['activation'],
                'learning_rate': best_trial['learning_rate'],
                'n_steps': best_trial['n_steps'],
                'batch_size': best_trial['batch_size'],
                'n_epochs': best_trial['n_epochs'],
                'gamma': best_trial['gamma'],
                'clip_range': best_trial['clip_range'],
                'ent_coef': best_trial['ent_coef'],
                'total_timesteps': best_trial['total_timesteps'],
                'seed': best_trial['seed']
            },
            'user_attrs': {
                'mean_reward': best_trial['mean_reward'],
                'experiment_name': f"ppo_needle_reach_final_trial_{best_trial['trial_number']:03d}",
                'model_path': f"results/experiments/ppo_needle_reach_final_trial_{best_trial['trial_number']:03d}/models/ppo_final.zip"
            }
        },
        'all_trials': []
    }
    
    # Add all trials
    for trial in trials:
        trial_data = {
            'number': trial['trial_number'],
            'value': trial['success_rate'],
            'params': {
                'hidden_sizes': trial['hidden_sizes'],
                'activation': trial['activation'],
                'learning_rate': trial['learning_rate'],
                'n_steps': trial['n_steps'],
                'batch_size': trial['batch_size'],
                'n_epochs': trial['n_epochs'],
                'gamma': trial['gamma'],
                'clip_range': trial['clip_range'],
                'ent_coef': trial['ent_coef'],
                'total_timesteps': trial['total_timesteps'],
                'seed': trial['seed']
            },
            'state': 'TrialState.COMPLETE',
            'user_attrs': {
                'mean_reward': trial['mean_reward'],
                'experiment_name': f"ppo_needle_reach_final_trial_{trial['trial_number']:03d}",
                'model_path': f"results/experiments/ppo_needle_reach_final_trial_{trial['trial_number']:03d}/models/ppo_final.zip"
            }
        }
        study_data['all_trials'].append(trial_data)
    
    # Save study.json
    with open(output_dir / 'study.json', 'w') as f:
        json.dump(study_data, f, indent=2)
    
    print(f"Generated study.json with {len(trials)} trials")
    print(f"Best trial: {best_trial['trial_number']} with {best_trial['success_rate']:.1%} success rate")

def generate_optimal_config(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate optimal config YAML file."""
    best_trial = results['best_trial']
    if not best_trial:
        return
        
    config = {
        'algorithm': 'ppo',
        'network': {
            'needle_reach': {
                'hidden_sizes': best_trial['hidden_sizes'],
                'activation': best_trial['activation']
            }
        },
        'training': {
            'needle_reach': {
                'learning_rate': best_trial['learning_rate'],
                'n_steps': best_trial['n_steps'],
                'batch_size': best_trial['batch_size'],
                'n_epochs': best_trial['n_epochs'],
                'gamma': best_trial['gamma'],
                'clip_range': best_trial['clip_range'],
                'ent_coef': best_trial['ent_coef'],
                'total_timesteps': best_trial['total_timesteps']
            }
        },
        'data': {
            'num_demonstrations': 100
        },
        'logging': {
            'tensorboard': False,
            'csv_logging': True,
            'save_freq': 5000,
            'verbose': 1
        },
        'seed': best_trial['seed']
    }
    
    with open(output_dir / 'optimal_config_ppo_needle_reach.yaml', 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)
    
    print(f"Generated optimal config for trial {best_trial['trial_number']}")

def generate_trials_csv(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate trials CSV file."""
    if not HAS_PANDAS:
        print("Pandas not available, skipping CSV generation")
        return
        
    df = pd.DataFrame(results['trials'])
    
    # Reorder columns for better readability
    column_order = [
        'trial_number', 'success_rate', 'mean_reward', 'std_reward', 
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
        'gamma', 'clip_range', 'ent_coef', 'total_timesteps',
        'hidden_sizes', 'activation', 'seed'
    ]
    
    df = df[column_order]
    df.to_csv(output_dir / 'trials.csv', index=False)
    
    print(f"Generated trials.csv with {len(df)} trials")
    print(f"Success rate range: {df['success_rate'].min():.1%} - {df['success_rate'].max():.1%}")

def generate_visualizations(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate HTML visualizations."""
    if not HAS_PLOTLY or not HAS_PANDAS:
        print("Plotly or Pandas not available, skipping visualizations")
        return
        
    trials = results['trials']
    df = pd.DataFrame(trials)
    
    # Optimization History
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['trial_number'],
        y=df['success_rate'],
        mode='lines+markers',
        name='Success Rate',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add best value line
    best_so_far = df['success_rate'].cummax()
    fig.add_trace(go.Scatter(
        x=df['trial_number'],
        y=best_so_far,
        mode='lines',
        name='Best Value',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='PPO Hyperparameter Optimization History - Needle Reach',
        xaxis_title='Trial Number',
        yaxis_title='Success Rate',
        hovermode='x unified'
    )
    
    fig.write_html(output_dir / 'optimization_history.html')
    
    # Parameter Importance (correlations with success rate)
    numeric_params = ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 
                     'gamma', 'clip_range', 'ent_coef', 'total_timesteps']
    
    correlations = []
    for param in numeric_params:
        if param in df.columns:
            corr = df[param].corr(df['success_rate'])
            correlations.append({'parameter': param, 'correlation': abs(corr)})
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=True)
    
    fig2 = go.Figure(go.Bar(
        x=corr_df['correlation'],
        y=corr_df['parameter'],
        orientation='h',
        marker=dict(color='lightblue')
    ))
    
    fig2.update_layout(
        title='Parameter Importance (Absolute Correlation with Success Rate)',
        xaxis_title='Absolute Correlation',
        yaxis_title='Parameter'
    )
    
    fig2.write_html(output_dir / 'parameter_importance.html')
    
    # Parallel Coordinates
    # Select key parameters for parallel coordinates
    key_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'success_rate']
    
    # Convert list columns to string for visualization
    viz_df = df.copy()
    viz_df['hidden_sizes_str'] = viz_df['hidden_sizes'].astype(str)
    
    fig3 = go.Figure(data=go.Parcoords(
        line=dict(color=viz_df['success_rate'], 
                 colorscale='Viridis',
                 showscale=True,
                 cmin=viz_df['success_rate'].min(),
                 cmax=viz_df['success_rate'].max()),
        dimensions=[
            dict(range=[viz_df['learning_rate'].min(), viz_df['learning_rate'].max()],
                 label='Learning Rate', values=viz_df['learning_rate']),
            dict(range=[viz_df['n_steps'].min(), viz_df['n_steps'].max()],
                 label='N Steps', values=viz_df['n_steps']),
            dict(range=[viz_df['batch_size'].min(), viz_df['batch_size'].max()],
                 label='Batch Size', values=viz_df['batch_size']),
            dict(range=[viz_df['gamma'].min(), viz_df['gamma'].max()],
                 label='Gamma', values=viz_df['gamma']),
            dict(range=[viz_df['success_rate'].min(), viz_df['success_rate'].max()],
                 label='Success Rate', values=viz_df['success_rate'])
        ]
    ))
    
    fig3.update_layout(title='PPO Hyperparameter Parallel Coordinates - Needle Reach')
    fig3.write_html(output_dir / 'parallel_coordinates.html')
    
    print("Generated visualizations: optimization_history.html, parameter_importance.html, parallel_coordinates.html")

def main():
    """Main function."""
    # Paths (relative to /app in Docker container)
    results_dir = Path("experiments/results/experiments")
    output_dir = Path("experiments/results/hyperopt_phase3/ppo_needle_reach_hyperopt")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting PPO results...")
    results = collect_ppo_results(results_dir)
    
    if not results['trials']:
        print("No PPO trials found!")
        return
    
    print(f"\nGenerating analysis for {results['n_trials']} PPO trials...")
    
    # Generate all analysis files
    generate_study_json(results, output_dir)
    generate_optimal_config(results, output_dir)
    generate_trials_csv(results, output_dir)
    generate_visualizations(results, output_dir)
    
    # Print summary
    best_trial = results['best_trial']
    print(f"\n{'='*60}")
    print("PPO HYPERPARAMETER OPTIMIZATION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total Trials: {results['n_trials']}")
    print(f"Best Trial: {best_trial['trial_number']}")
    print(f"Best Success Rate: {best_trial['success_rate']:.1%}")
    print(f"Best Mean Reward: {best_trial['mean_reward']:.2f}")
    print(f"Best Parameters:")
    print(f"  Learning Rate: {best_trial['learning_rate']}")
    print(f"  N Steps: {best_trial['n_steps']}")
    print(f"  Batch Size: {best_trial['batch_size']}")
    print(f"  Network: {best_trial['hidden_sizes']}")
    print(f"  Total Timesteps: {best_trial['total_timesteps']}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()