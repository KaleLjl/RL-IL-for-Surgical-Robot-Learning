"""
Configuration for PPO Curriculum Learning on PegTransfer Task

This file defines the curriculum levels and their parameters for progressive
skill building in the PegTransfer manipulation task.
"""

# Curriculum Level Definitions
CURRICULUM_LEVELS = {
    1: {
        "name": "Precise Approach",
        "description": "Master reaching object with perfect positioning",
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
            "learning_rate": 1e-4,  # Reduced from 3e-4 for more stable learning
            "n_steps": 2048,        # Increased from 1024 for better value estimates
            "batch_size": 256,      # Increased from 64 for better gradient estimates
            "n_epochs": 10,
            "gamma": 0.99,
            "clip_range": 0.2,
        }
    },
    
    2: {
        "name": "Precise Grasp",
        "description": "Master stable grasping given good approach",
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
        "name": "Precise Transport",
        "description": "Master transport without dropping",
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
        "name": "Full Task Mastery",
        "description": "Complete full task with release precision",
        "max_episode_steps": 150,
        "success_criteria": {
            "position_tolerance": 0.005,    # 0.5cm position accuracy
            "height_tolerance": 0.004,      # 0.4cm height accuracy
            "complete_task": True,          # Full task completion required
        },
        "advancement": {
            "success_rate_threshold": 0.8,  # Target 80% success rate
            "min_episodes": 2000,           # More episodes for final mastery
            "evaluation_window": 200,
        },
        "ppo_params": {
            "learning_rate": 1e-4,
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
    "ent_coef": 0.01,  # Added entropy for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "verbose": 1,
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps_per_level": {
        1: 100000,  # Increased from 50k - approach needs more time with current success rate
        2: 100000,  # Grasping requires more exploration
        3: 150000,  # Transport is complex
        4: 200000,  # Full task integration
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
        raise ValueError(f"Invalid curriculum level: {level}. Must be 1-4.")
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