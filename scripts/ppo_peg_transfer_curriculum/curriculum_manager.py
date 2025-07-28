"""
Curriculum Manager for PPO PegTransfer Training

Manages curriculum state, tracks progress, and handles level advancement.
"""
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

from curriculum_config import (
    get_level_config, 
    get_advancement_criteria,
    TRAINING_CONFIG
)

class CurriculumManager:
    """Manages curriculum learning state and progression."""
    
    def __init__(self, 
                 save_path: str = "logs/ppo_curriculum/",
                 resume_from_file: Optional[str] = None):
        """
        Initialize curriculum manager.
        
        Args:
            save_path: Directory to save curriculum state
            resume_from_file: Path to resume from existing state
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        if resume_from_file and os.path.exists(resume_from_file):
            self.load_state(resume_from_file)
        else:
            self.reset_state()
    
    def reset_state(self):
        """Initialize curriculum state from scratch."""
        self.state = {
            "current_level": 1,
            "start_time": datetime.now().isoformat(),
            "level_history": [],
            "episode_results": deque(maxlen=1000),  # Keep last 1000 episodes
            "total_episodes": 0,
            "total_timesteps": 0,
            "level_episodes": 0,
            "level_timesteps": 0,
            "best_success_rate": 0.0,
            "current_success_rate": 0.0,
        }
        
        # Per-level tracking
        self.level_stats = {
            level: {
                "episodes": 0,
                "timesteps": 0,
                "successes": 0,
                "best_success_rate": 0.0,
                "advancement_time": None,
                "model_path": None,
            } for level in range(1, 5)
        }
    
    def add_episode_result(self, 
                          success: bool, 
                          episode_length: int,
                          episode_reward: float,
                          info: Dict = None):
        """
        Record episode result and update statistics.
        
        Args:
            success: Whether episode was successful
            episode_length: Number of steps in episode
            episode_reward: Total episode reward
            info: Additional episode information
        """
        result = {
            "level": self.state["current_level"],
            "success": success,
            "length": episode_length,
            "reward": episode_reward,
            "timestamp": datetime.now().isoformat(),
            "info": info or {}
        }
        
        self.state["episode_results"].append(result)
        self.state["total_episodes"] += 1
        self.state["total_timesteps"] += episode_length
        self.state["level_episodes"] += 1
        self.state["level_timesteps"] += episode_length
        
        # Update level-specific stats
        level = self.state["current_level"]
        self.level_stats[level]["episodes"] += 1
        self.level_stats[level]["timesteps"] += episode_length
        if success:
            self.level_stats[level]["successes"] += 1
        
        # Update success rate
        self._update_success_rate()
    
    def _update_success_rate(self):
        """Calculate current success rate based on evaluation window."""
        level = self.state["current_level"]
        criteria = get_advancement_criteria(level)
        window_size = criteria["evaluation_window"]
        
        # Get recent episodes for current level
        recent_episodes = [
            r for r in list(self.state["episode_results"])[-window_size:]
            if r["level"] == level
        ]
        
        if len(recent_episodes) >= window_size:
            successes = sum(1 for r in recent_episodes if r["success"])
            self.state["current_success_rate"] = successes / len(recent_episodes)
        else:
            # Not enough episodes yet
            if recent_episodes:
                successes = sum(1 for r in recent_episodes if r["success"])
                self.state["current_success_rate"] = successes / len(recent_episodes)
            else:
                self.state["current_success_rate"] = 0.0
        
        # Update best success rate
        if self.state["current_success_rate"] > self.state["best_success_rate"]:
            self.state["best_success_rate"] = self.state["current_success_rate"]
            self.level_stats[level]["best_success_rate"] = self.state["current_success_rate"]
    
    def should_advance_level(self) -> bool:
        """Check if criteria are met to advance to next level."""
        level = self.state["current_level"]
        if level >= 4:  # Already at max level
            return False
        
        criteria = get_advancement_criteria(level)
        
        # Check minimum episodes
        if self.state["level_episodes"] < criteria["min_episodes"]:
            return False
        
        # Check success rate threshold
        if self.state["current_success_rate"] >= criteria["success_rate_threshold"]:
            return True
        
        return False
    
    def advance_level(self, model_path: Optional[str] = None) -> int:
        """
        Advance to next curriculum level.
        
        Args:
            model_path: Path to saved model for current level
            
        Returns:
            New curriculum level
        """
        current_level = self.state["current_level"]
        
        # Save level completion info
        self.level_stats[current_level]["advancement_time"] = datetime.now().isoformat()
        self.level_stats[current_level]["model_path"] = model_path
        
        # Record level transition
        self.state["level_history"].append({
            "from_level": current_level,
            "to_level": current_level + 1,
            "timestamp": datetime.now().isoformat(),
            "episodes": self.state["level_episodes"],
            "timesteps": self.state["level_timesteps"],
            "success_rate": self.state["current_success_rate"],
        })
        
        # Advance to next level
        self.state["current_level"] = min(current_level + 1, 4)
        self.state["level_episodes"] = 0
        self.state["level_timesteps"] = 0
        self.state["best_success_rate"] = 0.0
        self.state["current_success_rate"] = 0.0
        
        print(f"\n{'='*60}")
        print(f"CURRICULUM ADVANCEMENT: Level {current_level} → Level {self.state['current_level']}")
        print(f"Success rate achieved: {self.state['level_history'][-1]['success_rate']:.1%}")
        print(f"Episodes completed: {self.state['level_history'][-1]['episodes']}")
        print(f"{'='*60}\n")
        
        return self.state["current_level"]
    
    def get_current_level(self) -> int:
        """Get current curriculum level."""
        return self.state["current_level"]
    
    def get_progress_report(self) -> Dict:
        """Generate comprehensive progress report."""
        level = self.state["current_level"]
        level_config = get_level_config(level)
        criteria = get_advancement_criteria(level)
        
        report = {
            "current_level": level,
            "level_name": level_config["name"],
            "total_episodes": self.state["total_episodes"],
            "total_timesteps": self.state["total_timesteps"],
            "level_episodes": self.state["level_episodes"],
            "level_timesteps": self.state["level_timesteps"],
            "current_success_rate": self.state["current_success_rate"],
            "best_success_rate": self.state["best_success_rate"],
            "advancement_progress": {
                "episodes": f"{self.state['level_episodes']}/{criteria['min_episodes']}",
                "success_rate": f"{self.state['current_success_rate']:.1%}/{criteria['success_rate_threshold']:.0%}",
                "ready_to_advance": self.should_advance_level(),
            },
            "level_stats": self.level_stats,
        }
        
        return report
    
    def print_progress(self):
        """Print formatted progress report."""
        report = self.get_progress_report()
        
        print(f"\n--- Curriculum Progress Report ---")
        print(f"Current Level: {report['current_level']} - {report['level_name']}")
        print(f"Total Progress: {report['total_episodes']} episodes, {report['total_timesteps']} steps")
        print(f"Level Progress: {report['level_episodes']} episodes, {report['level_timesteps']} steps")
        print(f"Success Rate: {report['current_success_rate']:.1%} (best: {report['best_success_rate']:.1%})")
        print(f"Advancement: Episodes {report['advancement_progress']['episodes']}, "
              f"Success {report['advancement_progress']['success_rate']}")
        
        if report['advancement_progress']['ready_to_advance']:
            print("✓ Ready to advance to next level!")
        
        print("-" * 40)
    
    def save_state(self, filename: Optional[str] = None):
        """Save curriculum state to file."""
        if filename is None:
            filename = os.path.join(self.save_path, "curriculum_state.json")
        
        # Convert deque to list for JSON serialization
        state_to_save = self.state.copy()
        state_to_save["episode_results"] = list(self.state["episode_results"])
        
        save_data = {
            "state": state_to_save,
            "level_stats": self.level_stats,
            "save_time": datetime.now().isoformat(),
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Curriculum state saved to {filename}")
    
    def load_state(self, filename: str):
        """Load curriculum state from file."""
        with open(filename, 'r') as f:
            save_data = json.load(f)
        
        self.state = save_data["state"]
        self.state["episode_results"] = deque(
            self.state["episode_results"], 
            maxlen=1000
        )
        self.level_stats = save_data["level_stats"]
        
        print(f"Curriculum state loaded from {filename}")
        print(f"Resuming from Level {self.state['current_level']}")
    
    def get_stats_summary(self) -> str:
        """Get formatted summary of all level statistics."""
        summary = "\n=== Curriculum Learning Summary ===\n"
        
        for level in range(1, 5):
            stats = self.level_stats[level]
            config = get_level_config(level)
            
            summary += f"\nLevel {level}: {config['name']}\n"
            summary += f"  Episodes: {stats['episodes']}\n"
            summary += f"  Timesteps: {stats['timesteps']}\n"
            summary += f"  Successes: {stats['successes']}\n"
            
            if stats['episodes'] > 0:
                success_rate = stats['successes'] / stats['episodes']
                summary += f"  Success Rate: {success_rate:.1%}\n"
            
            if stats['advancement_time']:
                summary += f"  Completed: {stats['advancement_time']}\n"
            
            if level == self.state['current_level']:
                summary += f"  Status: CURRENT (Success rate: {self.state['current_success_rate']:.1%})\n"
        
        summary += "\n" + "=" * 35 + "\n"
        return summary