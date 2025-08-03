#!/usr/bin/env python3
"""
Entropy Scheduling for PPO to solve exploration noise dependency.

This module provides callbacks to gradually reduce entropy coefficient during training,
forcing the policy to become more deterministic over time.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LinearEntropySchedule(BaseCallback):
    """
    Callback for linear entropy coefficient scheduling.
    
    Gradually reduces entropy from start_value to end_value over total_timesteps.
    This forces the policy to become more deterministic as training progresses.
    """
    
    def __init__(self, start_ent_coef: float, end_ent_coef: float, total_timesteps: int, 
                 verbose: int = 0):
        """
        Args:
            start_ent_coef: Initial entropy coefficient (e.g., 0.01)
            end_ent_coef: Final entropy coefficient (e.g., 0.0001)
            total_timesteps: Total training timesteps for scheduling
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps
        self.current_ent_coef = start_ent_coef
        
    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        
        # Linear interpolation
        self.current_ent_coef = self.start_ent_coef - (self.start_ent_coef - self.end_ent_coef) * progress
        
        # Update the model's entropy coefficient
        self.model.ent_coef = self.current_ent_coef
        
        # Log occasionally
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Entropy coefficient: {self.current_ent_coef:.6f} (progress: {progress:.1%})")
        
        return True


class ExponentialEntropySchedule(BaseCallback):
    """
    Callback for exponential entropy coefficient scheduling.
    
    Reduces entropy exponentially, with faster reduction early in training.
    Good for tasks where you want to explore heavily at the start.
    """
    
    def __init__(self, start_ent_coef: float, end_ent_coef: float, total_timesteps: int,
                 decay_rate: float = 5.0, verbose: int = 0):
        """
        Args:
            start_ent_coef: Initial entropy coefficient
            end_ent_coef: Final entropy coefficient
            total_timesteps: Total training timesteps
            decay_rate: How fast to decay (higher = faster initial decay)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.total_timesteps = total_timesteps
        self.decay_rate = decay_rate
        self.current_ent_coef = start_ent_coef
        
    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        
        # Exponential decay
        decay_factor = np.exp(-self.decay_rate * progress)
        self.current_ent_coef = self.end_ent_coef + (self.start_ent_coef - self.end_ent_coef) * decay_factor
        
        # Update the model's entropy coefficient
        self.model.ent_coef = self.current_ent_coef
        
        # Log occasionally
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"Entropy coefficient: {self.current_ent_coef:.6f} (progress: {progress:.1%})")
        
        return True


class StepwiseEntropySchedule(BaseCallback):
    """
    Callback for stepwise entropy coefficient scheduling.
    
    Reduces entropy in discrete steps at specified milestones.
    Good for curriculum learning where you want stable phases.
    """
    
    def __init__(self, entropy_schedule: dict, verbose: int = 0):
        """
        Args:
            entropy_schedule: Dict mapping timesteps to entropy coefficients
                             e.g., {0: 0.01, 50000: 0.005, 100000: 0.001}
            verbose: Verbosity level
        """
        super().__init__(verbose)
        # Sort schedule by timesteps
        self.schedule = sorted(entropy_schedule.items())
        self.current_ent_coef = self.schedule[0][1]
        
    def _on_step(self) -> bool:
        # Find current entropy coefficient based on timesteps
        for timestep, ent_coef in reversed(self.schedule):
            if self.num_timesteps >= timestep:
                if self.current_ent_coef != ent_coef:
                    self.current_ent_coef = ent_coef
                    if self.verbose > 0:
                        print(f"Entropy coefficient changed to: {self.current_ent_coef:.6f} at timestep {self.num_timesteps}")
                break
        
        # Update the model's entropy coefficient
        self.model.ent_coef = self.current_ent_coef
        
        return True


def get_entropy_schedule(schedule_type: str, total_timesteps: int, 
                        start_ent: float = 0.01, end_ent: float = 0.0001) -> BaseCallback:
    """
    Get an entropy scheduling callback.
    
    Args:
        schedule_type: Type of schedule ('linear', 'exponential', 'stepwise')
        total_timesteps: Total training timesteps
        start_ent: Starting entropy coefficient
        end_ent: Ending entropy coefficient
    
    Returns:
        Entropy scheduling callback
    """
    if schedule_type == 'linear':
        return LinearEntropySchedule(start_ent, end_ent, total_timesteps, verbose=1)
    
    elif schedule_type == 'exponential':
        return ExponentialEntropySchedule(start_ent, end_ent, total_timesteps, 
                                        decay_rate=5.0, verbose=1)
    
    elif schedule_type == 'stepwise':
        # Default stepwise schedule
        schedule = {
            0: start_ent,
            int(total_timesteps * 0.3): start_ent * 0.5,
            int(total_timesteps * 0.6): start_ent * 0.1,
            int(total_timesteps * 0.8): end_ent
        }
        return StepwiseEntropySchedule(schedule, verbose=1)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# Recommended schedules for different scenarios
SCHEDULES = {
    'precision_task': {
        'type': 'linear',
        'start_ent': 0.01,
        'end_ent': 0.0001,
        'description': 'For tasks requiring precise final positioning'
    },
    'exploration_heavy': {
        'type': 'exponential', 
        'start_ent': 0.02,
        'end_ent': 0.001,
        'description': 'For tasks needing heavy initial exploration'
    },
    'curriculum': {
        'type': 'stepwise',
        'start_ent': 0.01,
        'end_ent': 0.0001,
        'description': 'For curriculum learning with distinct phases'
    }
}