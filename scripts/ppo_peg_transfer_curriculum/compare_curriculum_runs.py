#!/usr/bin/env python3
"""
Compare Multiple PPO Curriculum Learning Runs

This script enables comparison of different curriculum learning runs to identify
the best performing configurations and hyperparameters.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from curriculum_config import CURRICULUM_LEVELS


def load_run_data(base_model_dir: str, base_log_dir: str, run_name: str) -> Dict:
    """Load all data for a specific run."""
    run_data = {
        "run_name": run_name,
        "metadata": None,
        "curriculum_state": None,
        "level_metrics": {},
        "model_paths": {},
    }
    
    # Load metadata
    metadata_path = os.path.join(base_model_dir, "runs", run_name, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            run_data["metadata"] = json.load(f)
    
    # Load curriculum state
    state_path = os.path.join(base_log_dir, "runs", run_name, "curriculum_state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            curriculum_data = json.load(f)
            run_data["curriculum_state"] = curriculum_data["state"]
            run_data["level_stats"] = curriculum_data.get("level_stats", {})
    
    # Load level metrics
    for level in range(1, 5):
        metrics_path = os.path.join(base_log_dir, "runs", run_name, f"level_{level}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                run_data["level_metrics"][level] = json.load(f)
        
        # Find model paths
        model_path = os.path.join(base_model_dir, "runs", run_name, f"level_{level}", "model_final.zip")
        if os.path.exists(model_path):
            run_data["model_paths"][level] = model_path
    
    return run_data


def extract_comparison_metrics(run_data: Dict) -> Dict:
    """Extract key metrics for comparison."""
    metrics = {
        "run_name": run_data["run_name"],
        "start_time": run_data["metadata"]["start_time"] if run_data["metadata"] else None,
        "total_episodes": run_data["curriculum_state"]["total_episodes"] if run_data["curriculum_state"] else 0,
        "total_timesteps": run_data["curriculum_state"]["total_timesteps"] if run_data["curriculum_state"] else 0,
        "max_level_reached": run_data["curriculum_state"]["current_level"] if run_data["curriculum_state"] else 0,
    }
    
    # Extract per-level metrics
    for level in range(1, 5):
        level_key = f"level_{level}"
        
        # From level stats
        if run_data.get("level_stats") and str(level) in run_data["level_stats"]:
            level_stats = run_data["level_stats"][str(level)]
            metrics[f"{level_key}_episodes"] = level_stats.get("episodes", 0)
            metrics[f"{level_key}_success_rate"] = (
                level_stats["successes"] / level_stats["episodes"] 
                if level_stats.get("episodes", 0) > 0 else 0
            )
            metrics[f"{level_key}_completed"] = level_stats.get("advancement_time") is not None
        else:
            metrics[f"{level_key}_episodes"] = 0
            metrics[f"{level_key}_success_rate"] = 0
            metrics[f"{level_key}_completed"] = False
        
        # From level metrics
        if level in run_data["level_metrics"]:
            level_metrics = run_data["level_metrics"][level]
            metrics[f"{level_key}_final_reward"] = level_metrics.get("final_mean_reward", None)
    
    # Extract hyperparameters
    if run_data["metadata"] and "command_args" in run_data["metadata"]:
        args = run_data["metadata"]["command_args"]
        metrics["learning_rate"] = args.get("learning_rate", "default")
        metrics["timesteps"] = args.get("timesteps", "default")
        metrics["run_tag"] = args.get("run_tag", "")
    
    return metrics


def create_comparison_table(runs_data: List[Dict]) -> pd.DataFrame:
    """Create a comparison table of all runs."""
    comparison_data = []
    
    for run_data in runs_data:
        metrics = extract_comparison_metrics(run_data)
        comparison_data.append(metrics)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by max level reached and total success
    if not df.empty:
        df = df.sort_values(["max_level_reached", "total_episodes"], ascending=[False, True])
    
    return df


def plot_success_rates_comparison(runs_data: List[Dict], save_path: str = None):
    """Plot success rates across levels for multiple runs."""
    plt.figure(figsize=(12, 6))
    
    for run_data in runs_data:
        run_name = run_data["run_name"]
        levels = []
        success_rates = []
        
        for level in range(1, 5):
            if run_data.get("level_stats") and str(level) in run_data["level_stats"]:
                level_stats = run_data["level_stats"][str(level)]
                if level_stats.get("episodes", 0) > 0:
                    levels.append(level)
                    success_rate = level_stats["successes"] / level_stats["episodes"]
                    success_rates.append(success_rate * 100)
        
        if levels:
            plt.plot(levels, success_rates, marker='o', label=run_name, linewidth=2)
    
    # Add advancement thresholds
    for level in range(1, 5):
        threshold = CURRICULUM_LEVELS[level]["advancement"]["success_rate_threshold"] * 100
        plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.3)
        plt.text(0.5, threshold + 1, f'L{level} threshold', fontsize=8, alpha=0.5)
    
    plt.xlabel('Curriculum Level', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rates Comparison Across Runs', fontsize=14)
    plt.xticks([1, 2, 3, 4])
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_training_efficiency(runs_data: List[Dict], save_path: str = None):
    """Plot training efficiency (episodes to reach each level)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    run_names = []
    episodes_per_level = {1: [], 2: [], 3: [], 4: []}
    
    for run_data in runs_data:
        run_name = run_data["run_name"]
        run_names.append(run_name)
        
        for level in range(1, 5):
            if run_data.get("level_stats") and str(level) in run_data["level_stats"]:
                episodes = run_data["level_stats"][str(level)].get("episodes", 0)
            else:
                episodes = 0
            episodes_per_level[level].append(episodes)
    
    # Stacked bar chart for episodes per level
    x = np.arange(len(run_names))
    width = 0.6
    
    bottom = np.zeros(len(run_names))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for level in range(1, 5):
        ax1.bar(x, episodes_per_level[level], width, label=f'Level {level}',
                bottom=bottom, color=colors[level-1])
        bottom += episodes_per_level[level]
    
    ax1.set_xlabel('Run Name')
    ax1.set_ylabel('Episodes')
    ax1.set_title('Total Episodes per Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Success rate heatmap
    success_matrix = []
    for run_data in runs_data:
        row = []
        for level in range(1, 5):
            if run_data.get("level_stats") and str(level) in run_data["level_stats"]:
                level_stats = run_data["level_stats"][str(level)]
                if level_stats.get("episodes", 0) > 0:
                    success_rate = level_stats["successes"] / level_stats["episodes"]
                    row.append(success_rate * 100)
                else:
                    row.append(0)
            else:
                row.append(0)
        success_matrix.append(row)
    
    im = ax2.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(np.arange(4))
    ax2.set_yticks(np.arange(len(run_names)))
    ax2.set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4'])
    ax2.set_yticklabels(run_names)
    ax2.set_title('Success Rate Heatmap (%)')
    
    # Add text annotations
    for i in range(len(run_names)):
        for j in range(4):
            text = ax2.text(j, i, f'{success_matrix[i][j]:.1f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Success Rate (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def generate_comparison_report(runs_data: List[Dict], save_path: str):
    """Generate comprehensive comparison report."""
    report = {
        "comparison_date": datetime.now().isoformat(),
        "num_runs": len(runs_data),
        "runs": {},
        "summary": {
            "best_overall": None,
            "best_per_level": {},
            "most_efficient": None,
        }
    }
    
    # Extract metrics for all runs
    all_metrics = []
    for run_data in runs_data:
        metrics = extract_comparison_metrics(run_data)
        all_metrics.append(metrics)
        report["runs"][run_data["run_name"]] = metrics
    
    # Find best overall (reached highest level with good success rate)
    best_score = -1
    for metrics in all_metrics:
        # Score based on max level reached and average success rate
        score = metrics["max_level_reached"] * 100
        for level in range(1, metrics["max_level_reached"] + 1):
            score += metrics.get(f"level_{level}_success_rate", 0) * 10
        
        if score > best_score:
            best_score = score
            report["summary"]["best_overall"] = metrics["run_name"]
    
    # Find best per level
    for level in range(1, 5):
        best_success = -1
        best_run = None
        
        for metrics in all_metrics:
            success_rate = metrics.get(f"level_{level}_success_rate", 0)
            if success_rate > best_success:
                best_success = success_rate
                best_run = metrics["run_name"]
        
        if best_run:
            report["summary"]["best_per_level"][f"level_{level}"] = {
                "run_name": best_run,
                "success_rate": best_success
            }
    
    # Find most efficient (least episodes to reach furthest)
    best_efficiency = float('inf')
    for metrics in all_metrics:
        if metrics["max_level_reached"] >= 3:  # At least reached level 3
            efficiency = metrics["total_episodes"] / metrics["max_level_reached"]
            if efficiency < best_efficiency:
                best_efficiency = efficiency
                report["summary"]["most_efficient"] = {
                    "run_name": metrics["run_name"],
                    "episodes_per_level": efficiency,
                    "max_level": metrics["max_level_reached"]
                }
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def print_comparison_summary(report: Dict):
    """Print a formatted summary of the comparison."""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING COMPARISON SUMMARY")
    print("="*60)
    print(f"\nTotal runs compared: {report['num_runs']}")
    print(f"Comparison date: {report['comparison_date'][:19]}")
    
    if report["summary"]["best_overall"]:
        print(f"\nBest Overall Run: {report['summary']['best_overall']}")
        best_metrics = report["runs"][report["summary"]["best_overall"]]
        print(f"  - Max Level Reached: {best_metrics['max_level_reached']}")
        print(f"  - Total Episodes: {best_metrics['total_episodes']}")
    
    print("\nBest Success Rate per Level:")
    for level in range(1, 5):
        level_key = f"level_{level}"
        if level_key in report["summary"]["best_per_level"]:
            best = report["summary"]["best_per_level"][level_key]
            print(f"  Level {level}: {best['run_name']} ({best['success_rate']:.1%})")
    
    if report["summary"]["most_efficient"]:
        efficient = report["summary"]["most_efficient"]
        print(f"\nMost Efficient Run: {efficient['run_name']}")
        print(f"  - Episodes per Level: {efficient['episodes_per_level']:.1f}")
        print(f"  - Reached Level: {efficient['max_level']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Multiple PPO Curriculum Learning Runs"
    )
    
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default="models/ppo_curriculum",
        help="Base directory containing model runs"
    )
    
    parser.add_argument(
        "--base-log-dir",
        type=str,
        default="logs/ppo_curriculum",
        help="Base directory containing log runs"
    )
    
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run names to compare (if not specified, compares all)"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Find runs to compare
    if args.runs:
        run_names = args.runs
    else:
        # Find all available runs
        runs_dir = os.path.join(args.base_model_dir, "runs")
        if os.path.exists(runs_dir):
            run_names = [d for d in os.listdir(runs_dir) 
                        if os.path.isdir(os.path.join(runs_dir, d))]
        else:
            print(f"No runs directory found at {runs_dir}")
            return
    
    if not run_names:
        print("No runs found to compare")
        return
    
    print(f"Comparing {len(run_names)} runs: {', '.join(run_names)}")
    
    # Load data for all runs
    runs_data = []
    for run_name in run_names:
        print(f"\nLoading data for run: {run_name}")
        run_data = load_run_data(args.base_model_dir, args.base_log_dir, run_name)
        if run_data["metadata"] or run_data["curriculum_state"]:
            runs_data.append(run_data)
        else:
            print(f"  Warning: No data found for {run_name}")
    
    if not runs_data:
        print("No valid run data found")
        return
    
    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("results/ppo_curriculum/comparisons", f"comparison_{timestamp}")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving results to: {save_dir}")
    
    # Create comparison table
    comparison_df = create_comparison_table(runs_data)
    comparison_df.to_csv(os.path.join(save_dir, "comparison_table.csv"), index=False)
    print("\nComparison table saved")
    
    # Generate plots
    if not args.no_plots:
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        print("Generating comparison plots...")
        plot_success_rates_comparison(
            runs_data, 
            os.path.join(plots_dir, "success_rates_comparison.png")
        )
        
        plot_training_efficiency(
            runs_data,
            os.path.join(plots_dir, "training_efficiency.png")
        )
        print("Plots saved")
    
    # Generate report
    report_path = os.path.join(save_dir, "comparison_report.json")
    report = generate_comparison_report(runs_data, report_path)
    print(f"\nComparison report saved to {report_path}")
    
    # Print summary
    print_comparison_summary(report)


if __name__ == "__main__":
    main()