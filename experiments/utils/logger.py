"""Unified logging utilities for experiment tracking."""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Unified logger for experiment tracking."""
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 config: Dict[str, Any],
                 use_tensorboard: bool = True,
                 use_csv: bool = True):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            config: Experiment configuration
            use_tensorboard: Whether to use TensorBoard
            use_csv: Whether to log to CSV
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config
        
        # Initialize TensorBoard
        self.writer = None
        if use_tensorboard:
            tb_dir = self.log_dir / 'tensorboard'
            tb_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tb_dir))
        
        # Initialize CSV logging
        self.csv_logger = None
        if use_csv:
            self.csv_path = self.log_dir / 'metrics.csv'
            self.csv_logger = CSVLogger(str(self.csv_path))
        
        # Save configuration
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Log file for text output
        self.log_file = open(self.log_dir / 'training.log', 'w')
        
        # Metrics buffer for aggregation
        self.metrics_buffer = {}
        
        # Start time for duration tracking
        self.start_time = datetime.now()
        
        self.log_info(f"Experiment started: {experiment_name}")
        self.log_info(f"Log directory: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Training step
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        
        if self.csv_logger:
            self.csv_logger.log({tag: value, 'step': step})
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values.
        
        Args:
            tag: Group name
            values: Dictionary of metric names and values
            step: Training step
        """
        if self.writer:
            self.writer.add_scalars(tag, values, step)
        
        if self.csv_logger:
            log_dict = {f"{tag}/{k}": v for k, v in values.items()}
            log_dict['step'] = step
            self.csv_logger.log(log_dict)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log a histogram.
        
        Args:
            tag: Histogram name
            values: Values to histogram
            step: Training step
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_info(self, message: str):
        """Log an info message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_file.write(log_message + '\n')
        self.log_file.flush()
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(key, value, step)
            elif isinstance(value, dict):
                self.log_scalars(key, value, step)
            elif isinstance(value, (list, np.ndarray)):
                self.log_histogram(key, np.array(value), step)
    
    def add_to_buffer(self, key: str, value: float):
        """Add value to metrics buffer for aggregation.
        
        Args:
            key: Metric key
            value: Metric value
        """
        if key not in self.metrics_buffer:
            self.metrics_buffer[key] = []
        self.metrics_buffer[key].append(value)
    
    def flush_buffer(self, step: int, prefix: str = ''):
        """Flush metrics buffer and log aggregated values.
        
        Args:
            step: Training step
            prefix: Prefix for metric names
        """
        for key, values in self.metrics_buffer.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                metric_name = f"{prefix}/{key}" if prefix else key
                self.log_scalar(f"{metric_name}/mean", mean_val, step)
                self.log_scalar(f"{metric_name}/std", std_val, step)
        
        self.metrics_buffer.clear()
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_name: str):
        """Save a checkpoint.
        
        Args:
            checkpoint_data: Data to save
            checkpoint_name: Name of checkpoint file
        """
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # Save using appropriate method based on data type
        import torch
        if any(isinstance(v, torch.nn.Module) for v in checkpoint_data.values()):
            torch.save(checkpoint_data, checkpoint_path)
        else:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        self.log_info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_evaluation_results(self, results: Dict[str, Any], step: int):
        """Log evaluation results.
        
        Args:
            results: Evaluation results dictionary
            step: Training step
        """
        # Log to tensorboard
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"eval/{key}", value, step)
        
        # Save to JSON file
        eval_file = self.log_dir / f'evaluation_step_{step}.json'
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        self.log_info(f"Evaluation at step {step}: {results}")
    
    def close(self):
        """Close all logging resources."""
        if self.writer:
            self.writer.close()
        
        if self.csv_logger:
            self.csv_logger.close()
        
        # Log duration
        duration = datetime.now() - self.start_time
        self.log_info(f"Experiment completed. Duration: {duration}")
        
        self.log_file.close()


class CSVLogger:
    """CSV logger for metrics tracking."""
    
    def __init__(self, csv_path: str):
        """Initialize CSV logger.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.file = open(self.csv_path, 'w', newline='')
        self.writer = None
        self.headers_written = False
    
    def log(self, data: Dict[str, Any]):
        """Log data to CSV.
        
        Args:
            data: Dictionary of data to log
        """
        if not self.headers_written:
            self.writer = csv.DictWriter(self.file, fieldnames=data.keys())
            self.writer.writeheader()
            self.headers_written = True
        
        # Ensure all fields are present
        if self.writer:
            # Add missing fields with None
            for field in self.writer.fieldnames:
                if field not in data:
                    data[field] = None
            
            self.writer.writerow(data)
            self.file.flush()
    
    def close(self):
        """Close CSV file."""
        self.file.close()


class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for averaging
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
    
    def update(self, key: str, value: float):
        """Update a metric.
        
        Args:
            key: Metric key
            value: Metric value
        """
        if key not in self.metrics:
            self.metrics[key] = []
            self.history[key] = []
        
        self.metrics[key].append(value)
        self.history[key].append(value)
        
        # Keep window size
        if len(self.metrics[key]) > self.window_size:
            self.metrics[key].pop(0)
    
    def get_average(self, key: str) -> Optional[float]:
        """Get average of metric over window.
        
        Args:
            key: Metric key
            
        Returns:
            Average value or None if not found
        """
        if key in self.metrics and self.metrics[key]:
            return np.mean(self.metrics[key])
        return None
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value of metric.
        
        Args:
            key: Metric key
            
        Returns:
            Latest value or None if not found
        """
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_statistics(self, key: str) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            key: Metric key
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if key not in self.metrics or not self.metrics[key]:
            return {}
        
        values = self.metrics[key]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'latest': values[-1]
        }
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages of all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {key: self.get_average(key) for key in self.metrics}