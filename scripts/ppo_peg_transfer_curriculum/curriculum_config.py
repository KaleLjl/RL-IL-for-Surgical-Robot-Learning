"""
Configuration for PPO Curriculum Learning on PegTransfer Task

This file defines the curriculum levels and their parameters for progressive
skill building in the PegTransfer manipulation task.
"""

# Curriculum Level Definitions
CURRICULUM_LEVELS = {
    1: {
        "name": "Waypoint 1: Approach",
        "description": "Reach grasp position with open gripper",
        "max_episode_steps": 100,  # Increased from 50 to allow more exploration
        "success_criteria": {
            "distance_threshold": 0.01,  # 1cm in scaled units
            "stable_steps_required": 5,   # Must maintain position for 5 steps
            "orientation_tolerance": 15,  # degrees (not enforced in current implementation)
        },
        "advancement": {
            "success_rate_threshold": 0.8,  # 80% success rate
            "min_episodes": 1000,          # Minimum episodes before advancement
            "evaluation_window": 100,       # Last N episodes for success rate
        },
        "ppo_params": {
            "learning_rate": 3e-4,  # Use NeedleReach proven value (was working well)
            "n_steps": 2048,        # Keep this - works well for both
            "batch_size": 64,       # Use NeedleReach value (simpler, less overfitting)
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
        }
    },
    
    2: {
        "name": "Waypoint 2: Grasp",
        "description": "Close gripper to grasp object at position",
        "max_episode_steps": 80,
        "success_criteria": {
            "approach_required": True,      # Must achieve Level 1 criteria first
            "grasp_stable_steps": 10,      # Maintain grasp for 10 steps
            "lift_height": 0.005,          # Must lift object 0.5cm
        },
        "advancement": {
            "success_rate_threshold": 0.7,  # 70% success rate
            "min_episodes": 1000,
            "evaluation_window": 100,
        },
        "ppo_params": {
            "learning_rate": 2e-4,  # Slightly lower for fine control
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
        }
    },
    
    3: {
        "name": "Waypoint 3: Lift",
        "description": "Lift grasped object to above_height",
        "max_episode_steps": 120,
        "success_criteria": {
            "grasp_required": True,         # Must have stable grasp
            "goal_distance_threshold": 0.02, # Within 2cm of goal
            "no_drops_allowed": True,       # Constraint must be maintained
        },
        "advancement": {
            "success_rate_threshold": 0.6,  # 60% success rate
            "min_episodes": 1000,
            "evaluation_window": 100,
        },
        "ppo_params": {
            "learning_rate": 1e-4,  # Lower for complex coordination
            "n_steps": 4096,        # Longer rollouts for transport
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,         # Higher discount for longer episodes
            "clip_range": 0.15,     # Tighter clipping for stability
        }
    },
    
    4: {
        "name": "Waypoint 3: Lift",
        "description": "Lift grasped object to above position",
        "max_episode_steps": 120,
        "success_criteria": {
            "grasp_required": True,
            "lift_height": 0.045,           # Must lift to above_height
        },
        "advancement": {
            "success_rate_threshold": 0.7,
            "min_episodes": 1000,
            "evaluation_window": 100,
        },
        "ppo_params": {
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "clip_range": 0.15,
        }
    },
    
    5: {
        "name": "Waypoint 4: Transport",
        "description": "Transport grasped object to above goal position",
        "max_episode_steps": 140,
        "success_criteria": {
            "grasp_required": True,
            "transport_required": True,
        },
        "advancement": {
            "success_rate_threshold": 0.6,
            "min_episodes": 1000,
            "evaluation_window": 100,
        },
        "ppo_params": {
            "learning_rate": 8e-5,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "clip_range": 0.15,
        }
    },
    
    6: {
        "name": "Waypoint 5: Lower",
        "description": "Lower to release height while maintaining grasp",
        "max_episode_steps": 100,
        "success_criteria": {
            "grasp_required": True,
            "release_height_required": True,
        },
        "advancement": {
            "success_rate_threshold": 0.7,
            "min_episodes": 1000,
            "evaluation_window": 100,
        },
        "ppo_params": {
            "learning_rate": 8e-5,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "clip_range": 0.15,
        }
    },
    
    7: {
        "name": "Waypoint 6: Release",
        "description": "Release object at correct position to complete task",
        "max_episode_steps": 80,
        "success_criteria": {
            "position_tolerance": 0.005,    # 0.5cm position accuracy
            "height_tolerance": 0.004,      # 0.4cm height accuracy
            "complete_task": True,          # Full task completion required
        },
        "advancement": {
            "success_rate_threshold": 0.8,  # Target 80% success rate
            "min_episodes": 1500,           # More episodes for final precision
            "evaluation_window": 200,
        },
        "ppo_params": {
            "learning_rate": 5e-5,          # Slowest learning for precision
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "clip_range": 0.1,  # Most conservative for final precision
        }
    }
}

# Default PPO parameters (used as fallback)
DEFAULT_PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,   # Use NeedleReach default (PPO default)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "verbose": 1,
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps_per_level": {
        1: 100000,  # Safe approach above object
        2: 100000,  # Precise positioning for grasp  
        3: 100000,  # Grasping action
        4: 120000,  # Lifting coordination
        5: 150000,  # Transport coordination
        6: 100000,  # Lowering precision
        7: 120000,  # Release timing
    },
    "checkpoint_frequency": 10000,  # Save model every N steps
    "evaluation_frequency": 5000,    # Evaluate progress every N steps
    "log_interval": 1000,           # Log stats every N steps
    "save_best_model": True,        # Save best model per level
    "model_save_path": "models/ppo_curriculum/",
    "log_path": "logs/ppo_curriculum/",
    "tensorboard_log": "logs/ppo_curriculum/tensorboard/",
}

# Environment configuration
ENV_CONFIG = {
    "render_mode": None,  # Set to "human" for visualization during training
    "use_dense_reward": True,   # Use dense rewards for Level 1 (like NeedleReach)
}

def get_level_config(level: int) -> dict:
    """Get configuration for a specific curriculum level."""
    if level not in CURRICULUM_LEVELS:
        raise ValueError(f"Invalid curriculum level: {level}. Must be 1-7.")
    return CURRICULUM_LEVELS[level]

def get_ppo_params(level: int) -> dict:
    """Get PPO hyperparameters for a specific level."""
    level_config = get_level_config(level)
    params = DEFAULT_PPO_PARAMS.copy()
    params.update(level_config.get("ppo_params", {}))
    return params

def get_max_timesteps(level: int) -> int:
    """Get maximum training timesteps for a level."""
    return TRAINING_CONFIG["total_timesteps_per_level"].get(level, 100000)

def get_advancement_criteria(level: int) -> dict:
    """Get advancement criteria for a level."""
    return get_level_config(level)["advancement"]