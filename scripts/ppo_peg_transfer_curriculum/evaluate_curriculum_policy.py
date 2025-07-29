#!/usr/bin/env python3
"""
Evaluation Script for PPO Curriculum Learning Policies

This script evaluates trained models from each curriculum level and generates
performance reports.
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
from curriculum_config import get_level_config, CURRICULUM_LEVELS


def evaluate_model_on_level(model_path: str, env_name: str, level: int, 
                           n_episodes: int = 100, render: bool = False) -> dict:
    """Evaluate a trained model on a specific curriculum level."""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(
        env_name,
        render_mode=render_mode,
        use_dense_reward=False,
        curriculum_level=level
    )
    env = FlattenDictObsWrapper(env)
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluate
    print(f"Evaluating {os.path.basename(model_path)} on Level {level}...")
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    early_exits = 0
    
    # Detailed tracking
    approach_successes = 0
    grasp_successes = 0
    transport_successes = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # Episode tracking
        achieved_approach = False
        achieved_grasp = False
        achieved_transport = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Track progress (need to access base env)
            base_env = env.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Check approach
            eef_pos = obs[:3]  # First 3 elements in flattened observation
            if hasattr(base_env, 'obj_id'):
                from dvrk_gym.utils.pybullet_utils import get_body_pose
                obj_pos, _ = get_body_pose(base_env.obj_id)
                distance = np.linalg.norm(eef_pos - np.array(obj_pos))
                
                if distance < 0.01 * base_env.SCALING:
                    achieved_approach = True
            
            # Check grasp
            if (hasattr(base_env, '_activated') and 
                base_env._activated >= 0 and 
                base_env._contact_constraint is not None):
                achieved_grasp = True
            
            # Check transport
            if achieved_grasp and hasattr(base_env, 'goal'):
                goal_distance = np.linalg.norm(np.array(obj_pos) - base_env.goal)
                if goal_distance < 0.02 * base_env.SCALING:
                    achieved_transport = True
        
        # Record results
        success = info.get('is_success', False)
        early_exit = info.get('early_exit', False)
        
        if success:
            successes += 1
        if early_exit:
            early_exits += 1
        if achieved_approach:
            approach_successes += 1
        if achieved_grasp:
            grasp_successes += 1
        if achieved_transport:
            transport_successes += 1
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if (episode + 1) % 20 == 0:
            print(f"  Progress: {episode + 1}/{n_episodes} episodes completed")
    
    env.close()
    
    # Calculate statistics
    results = {
        "model_path": model_path,
        "level": level,
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "early_exit_rate": early_exits / n_episodes,
        "approach_rate": approach_successes / n_episodes,
        "grasp_rate": grasp_successes / n_episodes,
        "transport_rate": transport_successes / n_episodes,
    }
    
    return results


def evaluate_cross_level_performance(model_paths: dict, env_name: str, 
                                   n_episodes: int = 50) -> dict:
    """Evaluate each level's model on all curriculum levels."""
    
    print("\n" + "="*60)
    print("Cross-Level Performance Evaluation")
    print("="*60 + "\n")
    
    results = {}
    
    for trained_level, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Skipping Level {trained_level} - model not found")
            continue
            
        print(f"\nEvaluating Level {trained_level} model across all levels:")
        level_results = {}
        
        for test_level in range(1, 5):
            result = evaluate_model_on_level(
                model_path, env_name, test_level, n_episodes, render=False
            )
            if result:
                level_results[test_level] = result
        
        results[trained_level] = level_results
    
    return results


def generate_performance_matrix(cross_results: dict):
    """Generate and display performance matrix."""
    
    print("\n" + "="*60)
    print("Performance Matrix (Rows: Trained Level, Cols: Test Level)")
    print("="*60 + "\n")
    
    # Create matrix
    levels = [1, 2, 3, 4]
    matrix = np.zeros((4, 4))
    
    for trained_level in levels:
        if trained_level not in cross_results:
            continue
        for test_level in levels:
            if test_level in cross_results[trained_level]:
                matrix[trained_level-1, test_level-1] = \
                    cross_results[trained_level][test_level]['success_rate'] * 100
    
    # Print matrix
    print("Success Rates (%):")
    print("       Level 1   Level 2   Level 3   Level 4")
    for i, trained_level in enumerate(levels):
        print(f"L{trained_level}:    ", end="")
        for j in range(4):
            print(f"{matrix[i, j]:7.1f}", end="   ")
        print()
    
    # Analysis
    print("\nAnalysis:")
    for i in range(4):
        if matrix[i, i] > 0:
            print(f"Level {i+1}: {matrix[i, i]:.1f}% on own level")
            
            # Check forward transfer
            if i < 3 and matrix[i, i+1] > 0:
                print(f"  → Forward transfer to L{i+2}: {matrix[i, i+1]:.1f}%")
            
            # Check if it still solves previous levels
            if i > 0:
                prev_performance = [matrix[i, j] for j in range(i)]
                if all(p > 50 for p in prev_performance):
                    print(f"  ✓ Maintains performance on previous levels")
                else:
                    print(f"  ✗ Degraded performance on some previous levels")
    
    return matrix


def plot_learning_curves(results: dict, save_path: str = None):
    """Plot success rates across curriculum levels."""
    
    plt.figure(figsize=(10, 6))
    
    # Extract data
    levels = []
    success_rates = []
    
    for level in sorted(results.keys()):
        if level in results and level in results[level]:
            levels.append(level)
            success_rates.append(results[level][level]['success_rate'] * 100)
    
    # Plot
    plt.plot(levels, success_rates, 'b-o', linewidth=2, markersize=10)
    
    # Add level names
    for i, level in enumerate(levels):
        config = get_level_config(level)
        plt.annotate(config['name'], 
                    (level, success_rates[i]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)
    
    # Formatting
    plt.xlabel('Curriculum Level', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('PPO Curriculum Learning Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3, 4])
    plt.ylim(0, 105)
    
    # Add advancement thresholds
    for level in range(1, 5):
        config = get_level_config(level)
        threshold = config['advancement']['success_rate_threshold'] * 100
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3, 
                   label=f'L{level} threshold' if level == 1 else None)
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def generate_report(all_results: dict, save_path: str = None):
    """Generate comprehensive evaluation report."""
    
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "results": all_results,
        "summary": {}
    }
    
    # Generate summary
    for level in range(1, 5):
        if level in all_results and level in all_results[level]:
            result = all_results[level][level]
            config = get_level_config(level)
            
            report["summary"][f"level_{level}"] = {
                "name": config["name"],
                "success_rate": result["success_rate"],
                "meets_threshold": result["success_rate"] >= config["advancement"]["success_rate_threshold"],
                "mean_reward": result["mean_reward"],
                "mean_episode_length": result["mean_length"],
                "sub_tasks": {
                    "approach": result["approach_rate"],
                    "grasp": result["grasp_rate"],
                    "transport": result["transport_rate"],
                }
            }
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {save_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    for level, summary in report["summary"].items():
        print(f"\n{summary['name']}:")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Meets Threshold: {'✓' if summary['meets_threshold'] else '✗'}")
        print(f"  Mean Reward: {summary['mean_reward']:.2f}")
        print(f"  Sub-tasks: Approach {summary['sub_tasks']['approach']:.1%}, "
              f"Grasp {summary['sub_tasks']['grasp']:.1%}, "
              f"Transport {summary['sub_tasks']['transport']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO Curriculum Learning Policies"
    )
    
    parser.add_argument(
        "--env",
        default="PegTransfer-v0",
        help="Environment name"
    )
    
    parser.add_argument(
        "--model-dir",
        default="models/ppo_curriculum/",
        help="Directory containing trained models (deprecated, use --run-name)"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the training run to evaluate"
    )
    
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default="models/ppo_curriculum",
        help="Base directory containing model runs"
    )
    
    parser.add_argument(
        "--base-results-dir",
        type=str,
        default="results/ppo_curriculum",
        help="Base directory for saving results"
    )
    
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        help="Evaluate specific level only"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to specific model file"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--cross-eval",
        action="store_true",
        help="Perform cross-level evaluation"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation"
    )
    
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save results (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Determine run to evaluate
    if args.run_name:
        run_name = args.run_name
        model_run_dir = os.path.join(args.base_model_dir, "runs", run_name)
        if not os.path.exists(model_run_dir):
            raise ValueError(f"Run directory not found: {model_run_dir}")
    elif args.model_path:
        # Using specific model path
        run_name = "custom_eval"
        model_run_dir = None
    else:
        # List available runs
        runs_dir = os.path.join(args.base_model_dir, "runs")
        if os.path.exists(runs_dir):
            available_runs = [d for d in os.listdir(runs_dir) 
                            if os.path.isdir(os.path.join(runs_dir, d))]
            if available_runs:
                print("Available runs to evaluate:")
                for run in sorted(available_runs):
                    print(f"  - {run}")
                print("\nPlease specify --run-name to evaluate a specific run")
                return
            else:
                print("No runs found in", runs_dir)
                return
        else:
            print("No runs directory found")
            return
    
    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(args.base_results_dir, "runs", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")
    
    # Find model paths
    model_paths = {}
    
    if args.model_path and args.level:
        # Specific model provided
        model_paths[args.level] = args.model_path
    elif model_run_dir:
        # Search for models in run directory (new manual training structure)
        for level in range(1, 5):
            # Look for level-specific final models
            model_path = os.path.join(model_run_dir, f"model_level_{level}_final.zip")
            if os.path.exists(model_path):
                model_paths[level] = model_path
            else:
                # Try to find checkpoint
                import glob
                checkpoint_dir = os.path.join(model_run_dir, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    pattern = os.path.join(checkpoint_dir, f"ppo_level_{level}_*.zip")
                    matches = glob.glob(pattern)
                    if matches:
                        # Use the latest checkpoint
                        model_paths[level] = sorted(matches)[-1]
    
    if not model_paths:
        print("No models found to evaluate!")
        return
    
    print(f"Found models for levels: {list(model_paths.keys())}")
    
    if args.cross_eval:
        # Cross-level evaluation
        cross_results = evaluate_cross_level_performance(
            model_paths, args.env, args.episodes // 2
        )
        
        # Generate performance matrix
        generate_performance_matrix(cross_results)
        
        # Save cross-evaluation results
        cross_save_path = os.path.join(save_dir, "cross_evaluation_results.json")
        with open(cross_save_path, 'w') as f:
            json.dump(cross_results, f, indent=2)
        
    else:
        # Standard evaluation
        all_results = {}
        
        if args.level:
            # Evaluate specific level
            if args.level in model_paths:
                result = evaluate_model_on_level(
                    model_paths[args.level], args.env, args.level, 
                    args.episodes, args.render
                )
                if result:
                    all_results[args.level] = {args.level: result}
        else:
            # Evaluate all levels
            for level, model_path in model_paths.items():
                result = evaluate_model_on_level(
                    model_path, args.env, level, args.episodes, args.render
                )
                if result:
                    if level not in all_results:
                        all_results[level] = {}
                    all_results[level][level] = result
        
        # Generate report
        report_path = os.path.join(save_dir, "evaluation_report.json")
        generate_report(all_results, report_path)
        
        # Plot results
        if len(all_results) > 1:
            plot_save_dir = os.path.join(save_dir, "plots")
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_path = os.path.join(plot_save_dir, "curriculum_performance.png")
            plot_learning_curves(all_results, plot_path)


if __name__ == "__main__":
    main()