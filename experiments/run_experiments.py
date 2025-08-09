#!/usr/bin/env python3
"""Batch experiment runner for systematic evaluation."""

import os
import sys
import yaml
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse


class ExperimentRunner:
    """Run batches of experiments."""
    
    def __init__(self, experiment_config: str, base_dir: str = 'results/experiments'):
        """Initialize experiment runner.
        
        Args:
            experiment_config: Path to experiment configuration
            base_dir: Base directory for results
        """
        self.config_path = Path(experiment_config)
        self.base_dir = Path(base_dir)
        
        # Load experiment configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create results directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.base_dir / f"batch_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        with open(self.run_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def run_single_experiment(self, 
                            name: str, 
                            config: Dict[str, Any],
                            task: str,
                            algorithm: str) -> Dict[str, Any]:
        """Run a single experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            task: Task name
            algorithm: Algorithm name
            
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"Task: {task}, Algorithm: {algorithm}")
        print('='*60)
        
        # Create experiment directory
        exp_dir = self.run_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        config_file = exp_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Build command
        cmd = [
            'python3', 'train_unified.py',
            '--config', str(config_file),
            '--task', task,
            '--algorithm', algorithm,
            '--output-dir', str(exp_dir),
            '--experiment-name', name,
            '--demo-dir', 'demonstrations',
            '--no-tensorboard'
        ]
        
        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                check=True
            )
            
            training_time = time.time() - start_time
            
            # Find trained model
            model_path = self.find_model(exp_dir)
            
            if model_path:
                # Run evaluation
                eval_results = self.evaluate_model(
                    model_path, task, algorithm, exp_dir
                )
            else:
                eval_results = {'error': 'Model not found'}
            
            return {
                'name': name,
                'status': 'success',
                'training_time': training_time,
                'evaluation': eval_results,
                'output': result.stdout
            }
        
        except subprocess.CalledProcessError as e:
            return {
                'name': name,
                'status': 'failed',
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def find_model(self, exp_dir: Path) -> Path:
        """Find trained model in experiment directory.
        
        Args:
            exp_dir: Experiment directory
            
        Returns:
            Path to model or None
        """
        model_dir = exp_dir / 'models'
        if not model_dir.exists():
            return None
        
        # Look for final model
        for pattern in ['*_final.zip', '*_best.zip', '*.zip']:
            models = list(model_dir.glob(pattern))
            if models:
                return models[0]
        
        return None
    
    def evaluate_model(self,
                       model_path: Path,
                       task: str,
                       algorithm: str,
                       output_dir: Path) -> Dict[str, Any]:
        """Evaluate a trained model.
        
        Args:
            model_path: Path to model
            task: Task name
            algorithm: Algorithm name
            output_dir: Output directory
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating model: {model_path}")
        
        cmd = [
            'python3', 'evaluate_unified.py',
            '--model', str(model_path),
            '--task', task,
            '--algorithm', algorithm,
            '--n-episodes', '100',
            '--output-dir', str(output_dir / 'evaluations')
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Load evaluation results
            eval_file = output_dir / 'evaluations' / f'evaluation_{algorithm}_{task}.json'
            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    return json.load(f)
            else:
                return {'error': 'Evaluation file not found'}
        
        except subprocess.CalledProcessError as e:
            return {'error': str(e)}
    
    def run_all_experiments(self):
        """Run all experiments defined in configuration."""
        results = []
        
        # Get experiment list
        experiments = self.config.get('experiments', [])
        
        # Get default task and algorithm
        default_task = self.config.get('task', 'needle_reach')
        default_algorithm = self.config.get('algorithm', 'bc')
        
        # Run each experiment
        for i, exp in enumerate(experiments):
            name = exp.get('name', f'exp_{i}')
            task = exp.get('task', default_task)
            algorithm = exp.get('algorithm', default_algorithm)
            
            # Merge configurations
            base_config = {}
            if 'base_config' in self.config:
                base_config_path = self.config_path.parent / self.config['base_config']
                with open(base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f)
            
            # Merge with experiment-specific config
            exp_config = self.merge_configs(base_config, exp)
            
            # Run experiment
            result = self.run_single_experiment(name, exp_config, task, algorithm)
            results.append(result)
            
            # Save intermediate results
            self.save_results(results)
        
        return results
    
    def merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in ['name', 'task', 'algorithm']:
                continue  # Skip metadata fields
            
            if isinstance(value, dict) and key in result:
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results.
        
        Args:
            results: List of experiment results
        """
        # Save as JSON
        with open(self.run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary CSV
        import csv
        csv_file = self.run_dir / 'summary.csv'
        
        if results:
            with open(csv_file, 'w', newline='') as f:
                # Extract metrics for CSV
                fieldnames = ['name', 'status', 'training_time']
                
                # Add evaluation metrics if available
                if 'evaluation' in results[0] and 'metrics' in results[0]['evaluation']:
                    fieldnames.extend(['success_rate', 'mean_reward', 'mean_length'])
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'name': result['name'],
                        'status': result['status'],
                        'training_time': result.get('training_time', 0)
                    }
                    
                    if 'evaluation' in result and 'metrics' in result['evaluation']:
                        metrics = result['evaluation']['metrics']
                        row['success_rate'] = metrics.get('success_rate', 0)
                        row['mean_reward'] = metrics.get('mean_reward', 0)
                        row['mean_length'] = metrics.get('mean_length', 0)
                    
                    writer.writerow(row)
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of experiment results.
        
        Args:
            results: List of experiment results
        """
        print("\n" + "="*80)
        print("EXPERIMENT BATCH SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\nSuccessful Experiments:")
            for result in successful:
                print(f"  - {result['name']}")
                if 'evaluation' in result and 'metrics' in result['evaluation']:
                    metrics = result['evaluation']['metrics']
                    success_rate = metrics.get('success_rate', 0) * 100
                    print(f"    Success Rate: {success_rate:.1f}%")
        
        if failed:
            print("\nFailed Experiments:")
            for result in failed:
                print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\nResults saved to: {self.run_dir}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Batch experiment runner')
    parser.add_argument('config', type=str,
                       help='Path to experiment configuration file')
    parser.add_argument('--base-dir', type=str, default='results/experiments',
                       help='Base directory for results')
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner(args.config, args.base_dir)
    
    # Run experiments
    results = runner.run_all_experiments()
    
    # Print summary
    runner.print_summary(results)


if __name__ == "__main__":
    main()