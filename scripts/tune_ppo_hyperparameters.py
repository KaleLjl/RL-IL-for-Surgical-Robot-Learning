#!/usr/bin/env python3
"""
Hyperparameter tuning script for PPO on PegTransfer using Optuna.
This script uses Bayesian optimization to efficiently find optimal hyperparameters.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
import os
import time
from datetime import datetime
import argparse
import pickle


def create_env(env_name: str, render: bool = False):
    """Create and wrap the environment."""
    render_mode = 'human' if render else None
    env = gym.make(env_name, render_mode=render_mode, use_dense_reward=True)
    env = FlattenDictObsWrapper(env)
    return env


def objective(trial: optuna.Trial, env_name: str, n_timesteps: int, n_eval_episodes: int = 5) -> float:
    """
    Objective function for Optuna optimization.
    Returns the mean reward over evaluation episodes.
    """
    # Suggest hyperparameters
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 30),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 0.0001, 0.1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
    }
    
    # Ensure batch_size is compatible with n_steps
    if hyperparams['batch_size'] > hyperparams['n_steps']:
        hyperparams['batch_size'] = hyperparams['n_steps']
    
    # Create environment
    env = create_env(env_name, render=False)
    eval_env = create_env(env_name, render=False)
    
    # Log trial info
    print(f"\nTrial {trial.number} started with params:")
    print(f"  LR: {hyperparams['learning_rate']:.2e}, Steps: {hyperparams['n_steps']}, "
          f"Batch: {hyperparams['batch_size']}, Entropy: {hyperparams['ent_coef']:.4f}")
    
    try:
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            **hyperparams,
            verbose=0,
            seed=trial.number  # For reproducibility
        )
        
        # Create evaluation callback for early stopping
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Train model
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        
        # Check success rate
        successes = []
        for _ in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            successes.append(info.get('is_success', False))
        
        success_rate = np.mean(successes)
        
        # Report intermediate result for pruning
        trial.report(mean_reward, n_timesteps)
        
        # Prune if the trial is not promising
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned. Mean reward: {mean_reward:.2f}")
            raise optuna.TrialPruned()
        
        # Log results
        print(f"  Trial {trial.number} finished. Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, "
              f"Success rate: {success_rate:.1%}")
        
        # Store additional metrics
        trial.set_user_attr("std_reward", std_reward)
        trial.set_user_attr("success_rate", success_rate)
        
        # Use a combined metric: prioritize success rate but consider reward
        if success_rate > 0:
            # If we have any success, heavily weight it
            combined_score = success_rate * 1000 + mean_reward
        else:
            # Otherwise just use mean reward
            combined_score = mean_reward
        
        return combined_score
        
    except Exception as e:
        print(f"  Trial {trial.number} failed with error: {e}")
        return -float('inf')
    finally:
        env.close()
        eval_env.close()


def optimize_hyperparameters(
    env_name: str,
    n_trials: int,
    n_timesteps: int,
    n_eval_episodes: int,
    study_name: str,
    storage_path: str
):
    """Run Optuna hyperparameter optimization."""
    
    # Create or load study
    storage = f"sqlite:///{storage_path}"
    
    # Use TPE sampler for better performance with RL
    sampler = TPESampler(n_startup_trials=5, multivariate=True)
    
    # Use median pruner to stop unpromising trials
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )
    
    # Optimize
    print(f"\nStarting optimization with {n_trials} trials...")
    print(f"Each trial will train for {n_timesteps} timesteps")
    
    study.optimize(
        lambda trial: objective(trial, env_name, n_timesteps, n_eval_episodes),
        n_trials=n_trials,
        catch=(Exception,),  # Continue even if some trials fail
        gc_after_trial=True  # Free memory after each trial
    )
    
    return study


def save_results(study: optuna.Study, output_dir: str, env_name: str = "PegTransfer-v0"):
    """Save optimization results and generate reports."""
    
    # Save study object
    with open(os.path.join(output_dir, "study.pkl"), "wb") as f:
        pickle.dump(study, f)
    
    # Get all trials sorted by value
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)
    
    # Save detailed results
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "datetime": datetime.now().isoformat(),
        "trials": []
    }
    
    for trial in trials:
        if trial.value is not None:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
                "state": str(trial.state)
            }
            results["trials"].append(trial_data)
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    print(f"\n{'='*70}")
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nCompleted {len([t for t in trials if t.value is not None])} trials successfully")
    
    # Best trial
    if study.best_trial.value is not None:
        print(f"\n🏆 BEST TRIAL (Trial {study.best_trial.number}):")
        print(f"   Combined Score: {study.best_value:.2f}")
        if "success_rate" in study.best_trial.user_attrs:
            print(f"   Success Rate: {study.best_trial.user_attrs['success_rate']:.1%}")
        if "std_reward" in study.best_trial.user_attrs:
            print(f"   Reward: {study.best_value:.2f} ± {study.best_trial.user_attrs['std_reward']:.2f}")
        
        print(f"\n   Best Hyperparameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float) and value < 0.01:
                print(f"   - {param}: {value:.2e}")
            else:
                print(f"   - {param}: {value}")
    
    # Top 5 trials
    print(f"\n📊 TOP 5 TRIALS:")
    print(f"{'Trial':<8}{'Score':<12}{'Success':<12}{'LR':<12}{'Steps':<8}{'Batch':<8}{'Entropy':<10}")
    print("-" * 80)
    
    for i, trial in enumerate(trials[:5]):
        if trial.value is not None:
            success_rate = trial.user_attrs.get('success_rate', 0.0)
            print(f"{trial.number:<8}{trial.value:<12.2f}{success_rate:<12.1%}"
                  f"{trial.params['learning_rate']:<12.2e}{trial.params['n_steps']:<8}"
                  f"{trial.params['batch_size']:<8}{trial.params['ent_coef']:<10.4f}")
    
    # Parameter importance (if enough trials)
    if len([t for t in trials if t.value is not None]) >= 10:
        print(f"\n📈 PARAMETER IMPORTANCE:")
        try:
            importance = optuna.importance.get_param_importances(study)
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for param, imp in sorted_importance[:5]:
                print(f"   {param}: {imp:.2%}")
        except:
            print("   (Not enough trials for importance analysis)")
    
    # Training command
    if study.best_trial.value is not None:
        print(f"\n🚀 TO TRAIN WITH BEST PARAMETERS:")
        print(f"   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py \\")
        print(f"     --env {env_name} \\")
        print(f"     --timesteps 500000 \\")
        print(f"     --learning-rate {study.best_params['learning_rate']} \\")
        print(f"     --n-steps {study.best_params['n_steps']} \\")
        print(f"     --batch-size {study.best_params['batch_size']}")
        
        # Save best config for easy reuse
        best_config_path = os.path.join(output_dir, "best_config.sh")
        with open(best_config_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Best hyperparameters from optimization\n")
            f.write(f"docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py \\\n")
            f.write(f"  --env {args.env} \\\n")
            f.write(f"  --timesteps 500000 \\\n")
            f.write(f"  --learning-rate {study.best_params['learning_rate']} \\\n")
            f.write(f"  --n-steps {study.best_params['n_steps']} \\\n")
            f.write(f"  --batch-size {study.best_params['batch_size']}\n")
        os.chmod(best_config_path, 0o755)
        print(f"\n   (Command saved to: {best_config_path})")


def visualize_optimization(study: optuna.Study, output_dir: str):
    """Generate optimization visualizations (requires optuna-dashboard)."""
    try:
        import optuna.visualization as vis
        import plotly
        
        # Create visualizations
        figs = {
            "optimization_history": vis.plot_optimization_history(study),
            "param_importances": vis.plot_param_importances(study),
            "parallel_coordinate": vis.plot_parallel_coordinate(study),
            "slice": vis.plot_slice(study),
        }
        
        # Save as HTML
        for name, fig in figs.items():
            plotly.offline.plot(fig, filename=os.path.join(output_dir, f"{name}.html"), auto_open=False)
        
        print(f"\n📊 Visualizations saved to {output_dir}")
        
    except ImportError:
        print("\n💡 Install plotly for visualizations: pip install plotly")


def main():
    parser = argparse.ArgumentParser(description="Optimize PPO hyperparameters using Optuna")
    parser.add_argument("--env", default="PegTransfer-v0",
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment to optimize")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of optimization trials")
    parser.add_argument("--timesteps", type=int, default=30000,
                       help="Timesteps per trial")
    parser.add_argument("--eval-episodes", type=int, default=5,
                       help="Episodes for evaluation")
    parser.add_argument("--study-name", type=str, default=None,
                       help="Name for the Optuna study")
    parser.add_argument("--output-dir", default="logs/optuna_ppo",
                       help="Directory to save results")
    parser.add_argument("--resume", action="store_true",
                       help="Resume previous study")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"ppo_{args.env}_{timestamp}"
    output_dir = os.path.join(args.output_dir, study_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage path for Optuna
    storage_path = os.path.join(output_dir, "optuna_study.db")
    
    print(f"{'='*70}")
    print("PPO HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print(f"{'='*70}")
    print(f"Environment: {args.env}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Timesteps per trial: {args.timesteps}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Output directory: {output_dir}")
    if args.resume:
        print(f"Resuming study: {study_name}")
    
    # Run optimization
    start_time = time.time()
    
    study = optimize_hyperparameters(
        env_name=args.env,
        n_trials=args.n_trials,
        n_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        study_name=study_name,
        storage_path=storage_path
    )
    
    optimization_time = time.time() - start_time
    
    # Save and display results
    save_results(study, output_dir, args.env)
    
    # Generate visualizations
    visualize_optimization(study, output_dir)
    
    print(f"\n⏱️  Total optimization time: {optimization_time/3600:.1f} hours")
    print(f"📁 Results saved to: {output_dir}")
    
    # Optuna dashboard command
    print(f"\n💻 To view interactive dashboard:")
    print(f"   optuna-dashboard sqlite:///{storage_path}")


if __name__ == "__main__":
    main()