"""
Custom callbacks for training monitoring and analysis.
Provides interval-based logging and early warning detection for debugging.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TrainingAnalysisCallback(BaseCallback):
    """
    Custom callback for interval-based training analysis and logging.
    
    This callback collects training metrics at fixed intervals and generates
    structured JSON logs for debugging and parameter tuning assistance.
    """
    
    def __init__(self, 
                 log_interval: int = 1000,
                 analysis_dir: Optional[str] = None,
                 algorithm: str = "ppo",
                 verbose: int = 0):
        """
        Initialize the training analysis callback.
        
        Args:
            log_interval: Steps between analysis logging
            analysis_dir: Directory to save analysis logs (auto-created if None)
            algorithm: Algorithm type ('ppo' or 'dapg')
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.algorithm = algorithm.lower()
        self.analysis_dir = analysis_dir
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.policy_losses = []
        self.value_losses = []
        self.bc_losses = []  # For DAPG
        self.gradient_norms = []
        
        # Analysis state
        self.last_analysis_step = 0
        self.analysis_counter = 0
        self.start_time = time.time()
        
        # Warning and recommendation tracking
        self.warnings = []
        self.recommendations = []
        
        print(f"🔍 Training Analysis Callback initialized for {algorithm.upper()}")
        print(f"📊 Log interval: {log_interval} steps")
        
    def _init_callback(self) -> None:
        """Initialize callback when training starts."""
        # Create analysis directory if not provided
        if self.analysis_dir is None:
            log_dir = Path(self.logger.dir) if self.logger.dir else Path("./logs")
            self.analysis_dir = log_dir / "analysis"
        else:
            self.analysis_dir = Path(self.analysis_dir)
            
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "algorithm": self.algorithm,
            "log_interval": self.log_interval,
            "start_time": datetime.now().isoformat(),
            "model_config": self._get_model_config()
        }
        
        config_path = self.analysis_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)
            
        if self.verbose > 0:
            print(f"📁 Analysis logs will be saved to: {self.analysis_dir}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration for logging."""
        config = {}
        
        # Helper function to safely extract config values
        def safe_extract(obj, attr):
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                # Make sure value is JSON serializable
                if callable(value):
                    # If it's a function, try to get the actual value
                    try:
                        return value()
                    except:
                        return str(value)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    return value
                else:
                    return str(value)
            return None
        
        # Extract basic PPO config
        config['learning_rate'] = safe_extract(self.model, 'learning_rate')
        config['n_steps'] = safe_extract(self.model, 'n_steps')
        config['batch_size'] = safe_extract(self.model, 'batch_size')
        config['gamma'] = safe_extract(self.model, 'gamma')
        config['gae_lambda'] = safe_extract(self.model, 'gae_lambda')
        config['clip_range'] = safe_extract(self.model, 'clip_range')
        config['ent_coef'] = safe_extract(self.model, 'ent_coef')
        
        # DAPG-specific config
        if self.algorithm == "dapg":
            config['bc_loss_weight'] = safe_extract(self.model, 'bc_loss_weight')
            config['bc_batch_size'] = safe_extract(self.model, 'bc_batch_size')
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return config
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Collect episode-level metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Episode completed
            if 'episode' in info:
                episode_info = info['episode']
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                
                # Success rate (if available)
                if 'is_success' in info:
                    self.episode_successes.append(info['is_success'])
        
        # Collect training metrics from logger
        if self.logger is not None:
            # Policy loss
            if 'train/policy_gradient_loss' in self.logger.name_to_value:
                self.policy_losses.append(self.logger.name_to_value['train/policy_gradient_loss'])
            
            # Value loss
            if 'train/value_loss' in self.logger.name_to_value:
                self.value_losses.append(self.logger.name_to_value['train/value_loss'])
            
            # BC loss (DAPG only)
            if self.algorithm == "dapg" and 'train/bc_loss' in self.logger.name_to_value:
                self.bc_losses.append(self.logger.name_to_value['train/bc_loss'])
        
        # Run analysis at intervals
        if self.num_timesteps - self.last_analysis_step >= self.log_interval:
            self._run_analysis()
            self.last_analysis_step = self.num_timesteps
        
        return True
    
    def _run_analysis(self) -> None:
        """Run training analysis and generate log."""
        self.analysis_counter += 1
        
        # Prepare analysis data
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": self.analysis_counter,
            "timestep": self.num_timesteps,
            "elapsed_time": time.time() - self.start_time,
            "algorithm": self.algorithm,
            "metrics": self._compute_metrics(),
            "health_check": self._check_training_health(),
            "recommendations": self._generate_recommendations(),
            "warnings": self._collect_warnings()
        }
        
        # Save analysis log
        log_filename = f"analysis_{self.analysis_counter:04d}_step_{self.num_timesteps}.json"
        log_path = self.analysis_dir / log_filename
        
        with open(log_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, cls=NumpyEncoder)
        
        # Print summary
        if self.verbose > 0:
            self._print_analysis_summary(analysis_data)
            
        # Clear old data to prevent memory issues
        self._clear_old_data()
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute training metrics for analysis."""
        metrics = {}
        
        # Episode metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
            metrics['episode_reward'] = {
                'mean': np.mean(recent_rewards),
                'std': np.std(recent_rewards),
                'min': np.min(recent_rewards),
                'max': np.max(recent_rewards),
                'count': len(recent_rewards)
            }
        
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-100:]
            metrics['episode_length'] = {
                'mean': np.mean(recent_lengths),
                'std': np.std(recent_lengths),
                'min': np.min(recent_lengths),
                'max': np.max(recent_lengths)
            }
        
        if self.episode_successes:
            recent_successes = self.episode_successes[-100:]
            metrics['success_rate'] = {
                'rate': np.mean(recent_successes),
                'count': len(recent_successes)
            }
        
        # Training loss metrics
        if self.policy_losses:
            recent_policy_losses = self.policy_losses[-50:]
            metrics['policy_loss'] = {
                'mean': np.mean(recent_policy_losses),
                'std': np.std(recent_policy_losses),
                'trend': self._compute_trend(recent_policy_losses)
            }
        
        if self.value_losses:
            recent_value_losses = self.value_losses[-50:]
            metrics['value_loss'] = {
                'mean': np.mean(recent_value_losses),
                'std': np.std(recent_value_losses),
                'trend': self._compute_trend(recent_value_losses)
            }
        
        # DAPG-specific metrics
        if self.algorithm == "dapg" and self.bc_losses:
            recent_bc_losses = self.bc_losses[-20:]
            metrics['bc_loss'] = {
                'mean': np.mean(recent_bc_losses),
                'std': np.std(recent_bc_losses),
                'trend': self._compute_trend(recent_bc_losses)
            }
        
        return metrics
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for a list of values."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _check_training_health(self) -> Dict[str, Any]:
        """Check training health and identify issues."""
        health = {
            "status": "healthy",
            "issues": []
        }
        
        # Check reward variance
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            if len(recent_rewards) > 50:
                reward_std = np.std(recent_rewards)
                if reward_std < 0.1:
                    health["issues"].append({
                        "type": "low_reward_variance",
                        "severity": "warning",
                        "message": "Reward variance is very low - policy may be stuck"
                    })
        
        # Check policy loss explosion
        if self.policy_losses:
            recent_policy_losses = self.policy_losses[-20:]
            if len(recent_policy_losses) > 5:
                if any(loss > 100 for loss in recent_policy_losses):
                    health["issues"].append({
                        "type": "exploding_policy_loss",
                        "severity": "critical",
                        "message": "Policy loss is exploding - consider reducing learning rate"
                    })
        
        # Check value function issues
        if self.value_losses:
            recent_value_losses = self.value_losses[-20:]
            if len(recent_value_losses) > 5:
                if np.mean(recent_value_losses) > 10:
                    health["issues"].append({
                        "type": "high_value_loss",
                        "severity": "warning",
                        "message": "Value function loss is high - may need more training data"
                    })
        
        # DAPG-specific health checks
        if self.algorithm == "dapg" and self.bc_losses:
            recent_bc_losses = self.bc_losses[-10:]
            if len(recent_bc_losses) > 3:
                mean_bc_loss = np.mean(recent_bc_losses)
                if mean_bc_loss > 2.0:
                    health["issues"].append({
                        "type": "high_bc_loss",
                        "severity": "warning",
                        "message": "BC loss is high - policy may be diverging from expert"
                    })
        
        if health["issues"]:
            health["status"] = "needs_attention"
        
        return health
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate parameter tuning recommendations."""
        recommendations = []
        
        # Learning rate recommendations
        if self.policy_losses:
            recent_losses = self.policy_losses[-50:]
            if len(recent_losses) > 20:
                trend = self._compute_trend(recent_losses)
                if trend == "increasing":
                    recommendations.append({
                        "type": "learning_rate",
                        "action": "decrease",
                        "reason": "Policy loss is increasing - consider reducing learning rate",
                        "priority": "high"
                    })
                elif trend == "stable" and np.mean(recent_losses) > 1.0:
                    recommendations.append({
                        "type": "learning_rate",
                        "action": "increase",
                        "reason": "Policy loss is stable but high - consider increasing learning rate",
                        "priority": "medium"
                    })
        
        # Batch size recommendations
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            if len(recent_rewards) > 50:
                reward_variance = np.var(recent_rewards)
                if reward_variance > 1000:
                    recommendations.append({
                        "type": "batch_size",
                        "action": "increase",
                        "reason": "High reward variance - consider increasing batch size",
                        "priority": "medium"
                    })
        
        # DAPG-specific recommendations
        if self.algorithm == "dapg":
            if self.bc_losses and self.episode_rewards:
                recent_bc_loss = np.mean(self.bc_losses[-10:]) if len(self.bc_losses) >= 10 else 0
                recent_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
                
                if recent_bc_loss < 0.1 and recent_reward < 0:
                    recommendations.append({
                        "type": "bc_weight",
                        "action": "decrease",
                        "reason": "BC loss low but poor performance - reduce BC weight",
                        "priority": "high"
                    })
                elif recent_bc_loss > 2.0:
                    recommendations.append({
                        "type": "bc_weight",
                        "action": "increase",
                        "reason": "BC loss high - increase BC weight for better expert following",
                        "priority": "high"
                    })
        
        return recommendations
    
    def _collect_warnings(self) -> List[str]:
        """Collect current warnings."""
        warnings = []
        
        # Check for recent issues
        if self.policy_losses:
            recent_losses = self.policy_losses[-10:]
            if len(recent_losses) > 5:
                if np.mean(recent_losses) > 50:
                    warnings.append("Policy loss is very high - training may be unstable")
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-20:]
            if len(recent_rewards) > 10:
                if np.mean(recent_rewards) < -1000:
                    warnings.append("Very low episode rewards - check reward function")
        
        return warnings
    
    def _print_analysis_summary(self, analysis_data: Dict[str, Any]) -> None:
        """Print a concise analysis summary."""
        print(f"\n📊 Training Analysis #{analysis_data['analysis_id']} (Step {analysis_data['timestep']})")
        print(f"⏱️  Elapsed time: {analysis_data['elapsed_time']:.1f}s")
        
        # Print key metrics
        metrics = analysis_data['metrics']
        if 'episode_reward' in metrics:
            reward_info = metrics['episode_reward']
            print(f"🎯 Reward: {reward_info['mean']:.3f} ± {reward_info['std']:.3f} (n={reward_info['count']})")
        
        if 'success_rate' in metrics:
            success_info = metrics['success_rate']
            print(f"✅ Success rate: {success_info['rate']:.3f} (n={success_info['count']})")
        
        if 'policy_loss' in metrics:
            policy_info = metrics['policy_loss']
            print(f"📉 Policy loss: {policy_info['mean']:.6f} ({policy_info['trend']})")
        
        if 'bc_loss' in metrics:
            bc_info = metrics['bc_loss']
            print(f"🎓 BC loss: {bc_info['mean']:.6f} ({bc_info['trend']})")
        
        # Print health status
        health = analysis_data['health_check']
        if health['status'] == 'healthy':
            print("💚 Training status: Healthy")
        else:
            print(f"⚠️  Training status: {health['status']}")
            for issue in health['issues']:
                print(f"   - {issue['message']}")
        
        # Print recommendations
        recommendations = analysis_data['recommendations']
        if recommendations:
            print("💡 Recommendations:")
            for rec in recommendations:
                print(f"   - {rec['reason']}")
        
        print("-" * 60)
    
    def _clear_old_data(self) -> None:
        """Clear old data to prevent memory issues."""
        max_keep = 200  # Keep last 200 entries
        
        if len(self.episode_rewards) > max_keep:
            self.episode_rewards = self.episode_rewards[-max_keep:]
        if len(self.episode_lengths) > max_keep:
            self.episode_lengths = self.episode_lengths[-max_keep:]
        if len(self.episode_successes) > max_keep:
            self.episode_successes = self.episode_successes[-max_keep:]
        if len(self.policy_losses) > max_keep:
            self.policy_losses = self.policy_losses[-max_keep:]
        if len(self.value_losses) > max_keep:
            self.value_losses = self.value_losses[-max_keep:]
        if len(self.bc_losses) > max_keep:
            self.bc_losses = self.bc_losses[-max_keep:]
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Generate final analysis
        self._run_analysis()
        
        # Create training summary
        summary = {
            "training_completed": True,
            "final_timestamp": datetime.now().isoformat(),
            "total_timesteps": self.num_timesteps,
            "total_time": time.time() - self.start_time,
            "total_analyses": self.analysis_counter,
            "final_metrics": self._compute_metrics(),
            "final_health": self._check_training_health()
        }
        
        summary_path = self.analysis_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n🎉 Training completed! Final analysis saved to: {summary_path}")