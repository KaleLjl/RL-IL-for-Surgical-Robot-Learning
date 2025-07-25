#!/usr/bin/env python3
"""
Test different reward schemes for PPO on PegTransfer environment.
This script runs short training sessions with different reward functions
to identify which reward design leads to better learning.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
import json
import os
import time
from typing import Dict, Any, Callable
import argparse


class RewardTestCallback(BaseCallback):
    """Custom callback to track metrics for reward comparison."""
    
    def __init__(self, log_dir: str, reward_name: str):
        super().__init__()
        self.log_dir = log_dir
        self.reward_name = reward_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.first_success_step = None
        self.grasp_attempts = 0
        self.successful_grasps = 0
        
    def _on_step(self) -> bool:
        # Track episode completion
        if self.locals.get("dones")[0]:
            info = self.locals.get("infos")[0]
            
            # Track metrics
            self.episode_rewards.append(info.get("episode", {}).get("r", 0))
            self.episode_lengths.append(info.get("episode", {}).get("l", 0))
            
            # Track success
            is_success = info.get("is_success", False)
            self.successes.append(is_success)
            
            # Track first success
            if is_success and self.first_success_step is None:
                self.first_success_step = self.num_timesteps
                
            # Track grasp attempts (if available in info)
            if "grasp_attempted" in info:
                self.grasp_attempts += 1
                if info.get("grasp_successful", False):
                    self.successful_grasps += 1
                    
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this reward scheme."""
        if not self.successes:
            success_rate = 0.0
        else:
            success_rate = sum(self.successes) / len(self.successes)
            
        if not self.episode_rewards:
            mean_reward = 0.0
            std_reward = 0.0
        else:
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            
        return {
            "reward_scheme": self.reward_name,
            "total_episodes": int(len(self.successes)),
            "success_rate": float(success_rate),
            "total_successes": int(sum(self.successes)),
            "first_success_step": int(self.first_success_step) if self.first_success_step else None,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "grasp_success_rate": float(self.successful_grasps / max(1, self.grasp_attempts)),
            "total_timesteps": int(self.num_timesteps)
        }


class RewardWrapper(gym.Wrapper):
    """Wrapper to override the reward function of an environment."""
    
    def __init__(self, env, reward_fn: Callable[[Dict], float], reward_name: str):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.reward_name = reward_name
        self._grasp_attempted = False
        self._grasp_successful = False
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get the base environment (unwrap if needed)
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Track grasp attempts
        if hasattr(base_env, '_activated'):
            if base_env._activated >= 0 and not self._grasp_attempted:
                self._grasp_attempted = True
                info["grasp_attempted"] = True
                
                # Check if grasp was successful
                if hasattr(base_env, '_contact_constraint') and base_env._contact_constraint is not None:
                    self._grasp_successful = True
                    info["grasp_successful"] = True
        
        # Override reward with custom function - pass base_env
        reward = self.reward_fn(obs, base_env, info)
        
        # Reset grasp tracking on episode end
        if terminated or truncated:
            self._grasp_attempted = False
            self._grasp_successful = False
            
        return obs, reward, terminated, truncated, info


def sparse_reward(obs: Dict, env: Any, info: Dict) -> float:
    """Pure sparse reward: +1 for success, 0 otherwise."""
    return 1.0 if info.get("is_success", False) else 0.0


def sparse_negative_reward(obs: Dict, env: Any, info: Dict) -> float:
    """Sparse reward with negative: 0 for success, -1 otherwise."""
    return 0.0 if info.get("is_success", False) else -1.0


def hybrid_reward_v1(obs: Dict, env: Any, info: Dict) -> float:
    """Hybrid: Sparse + small dense guidance only when grasping."""
    # Base sparse reward
    sparse = 1.0 if info.get("is_success", False) else 0.0
    
    # Add small dense component only when object is grasped
    if hasattr(env, '_activated') and env._activated >= 0 and hasattr(env, '_contact_constraint') and env._contact_constraint is not None:
        # Object is grasped, provide guidance to goal
        obj_pos = obs['achieved_goal']
        goal_pos = obs['desired_goal']
        dist = np.linalg.norm(obj_pos - goal_pos)
        dense_bonus = -dist * 0.1  # Small weight on dense component
        return sparse + dense_bonus
    
    return sparse


def hybrid_reward_v2(obs: Dict, env: Any, info: Dict) -> float:
    """Hybrid: Shaped rewards with strong success bonus."""
    # Strong success bonus
    if info.get("is_success", False):
        return 10.0
    
    # Get positions
    eef_pos = obs['observation'][:3]
    obj_pos = obs['achieved_goal']
    goal_pos = obs['desired_goal']
    
    # Check grasp status
    is_grasped = hasattr(env, '_activated') and env._activated >= 0 and hasattr(env, '_contact_constraint') and env._contact_constraint is not None
    
    if not is_grasped:
        # Approach phase: encourage getting close to object
        dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
        if dist_to_obj < 0.01 * getattr(env, 'SCALING', 5):  # Very close
            return 1.0  # Bonus for being close
        else:
            return -dist_to_obj * 0.5  # Gentle guidance
    else:
        # Transport phase: strong reward for moving toward goal
        dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
        return 3.0 - dist_to_goal  # Base reward for grasping minus distance


def progressive_reward(obs: Dict, env: Any, info: Dict) -> float:
    """Progressive reward: gives intermediate rewards for sub-goals."""
    reward = 0.0
    
    # Success gets highest reward
    if info.get("is_success", False):
        return 10.0
    
    # Get positions and states
    eef_pos = obs['observation'][:3]
    obj_pos = obs['achieved_goal']
    goal_pos = obs['desired_goal']
    jaw_angle = obs['observation'][6]
    
    # Sub-goal 1: Approach object (max 1 point)
    dist_to_obj = np.linalg.norm(eef_pos - obj_pos)
    if dist_to_obj < 0.02 * getattr(env, 'SCALING', 5):
        reward += 1.0
    
    # Sub-goal 2: Grasp attempt (max 2 points)
    if jaw_angle < 0 and dist_to_obj < 0.02 * getattr(env, 'SCALING', 5):
        reward += 2.0
    
    # Sub-goal 3: Successful grasp (max 3 points)
    is_grasped = hasattr(env, '_activated') and env._activated >= 0 and hasattr(env, '_contact_constraint') and env._contact_constraint is not None
    if is_grasped:
        reward += 3.0
        
        # Sub-goal 4: Transport (max 3 points based on progress)
        initial_dist = 0.2 * getattr(env, 'SCALING', 5)  # Approximate initial distance
        current_dist = np.linalg.norm(obj_pos - goal_pos)
        progress = max(0, (initial_dist - current_dist) / initial_dist)
        reward += progress * 3.0
    
    return reward


def curiosity_based_reward(obs: Dict, env: Any, info: Dict) -> float:
    """Adds intrinsic curiosity bonus to sparse reward."""
    # Base sparse reward
    sparse = 1.0 if info.get("is_success", False) else 0.0
    
    # Simple curiosity: reward for reaching new states
    # In practice, you'd use a learned model, but here we use position novelty
    eef_pos = obs['observation'][:3]
    obj_pos = obs['achieved_goal']
    
    # Create a simple state representation
    state_key = f"{eef_pos[0]:.2f},{eef_pos[1]:.2f},{eef_pos[2]:.2f},{obj_pos[0]:.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f}"
    
    # Initialize visit count if needed
    if not hasattr(env, '_state_visits'):
        env._state_visits = {}
    
    # Curiosity bonus inversely proportional to visit count
    visit_count = env._state_visits.get(state_key, 0)
    curiosity_bonus = 0.1 / (1 + visit_count)
    env._state_visits[state_key] = visit_count + 1
    
    return sparse + curiosity_bonus


def test_reward_scheme(
    reward_fn: Callable,
    reward_name: str,
    env_name: str,
    timesteps: int,
    log_dir: str,
    render: bool = True
) -> Dict[str, Any]:
    """Test a single reward scheme and return results."""
    print(f"\n{'='*60}")
    print(f"Testing reward scheme: {reward_name}")
    print(f"{'='*60}")
    
    # Create environment with custom reward
    render_mode = 'human' if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = RewardWrapper(env, reward_fn, reward_name)
    env = FlattenDictObsWrapper(env)
    
    # Create callback
    callback = RewardTestCallback(log_dir, reward_name)
    
    # Create PPO model with standard parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Add entropy for exploration
        verbose=1
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    train_time = time.time() - start_time
    
    # Get results
    results = callback.get_summary()
    results["training_time"] = train_time
    
    # Save model if it achieved any success
    if results["total_successes"] > 0:
        model_path = os.path.join(log_dir, f"{reward_name}_model.zip")
        model.save(model_path)
        results["model_path"] = model_path
    
    env.close()
    
    # Print summary
    print(f"\nResults for {reward_name}:")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Total successes: {results['total_successes']}")
    print(f"  First success at: {results['first_success_step'] or 'Never'}")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Training time: {results['training_time']:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test different reward schemes for PPO")
    parser.add_argument("--env", default="PegTransfer-v0", 
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment to test")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Timesteps per reward scheme test")
    parser.add_argument("--output-dir", default="logs/reward_tests",
                       help="Directory to save results")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for faster testing")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = int(time.time())
    output_dir = os.path.join(args.output_dir, f"{args.env}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define reward schemes to test
    reward_schemes = [
        (sparse_reward, "sparse_positive"),
        (sparse_negative_reward, "sparse_negative"),
        (hybrid_reward_v1, "hybrid_sparse_dense"),
        (hybrid_reward_v2, "hybrid_shaped"),
        (progressive_reward, "progressive_subgoals"),
        (curiosity_based_reward, "curiosity_bonus"),
    ]
    
    # Also test the original dense reward
    def original_dense(obs: Dict, env: Any, info: Dict) -> float:
        """Use environment's original dense reward."""
        # Environment is already unwrapped by RewardWrapper
        return env._get_dense_reward(obs)
    
    reward_schemes.append((original_dense, "original_dense"))
    
    # Test each reward scheme
    all_results = []
    for reward_fn, reward_name in reward_schemes:
        scheme_dir = os.path.join(output_dir, reward_name)
        os.makedirs(scheme_dir, exist_ok=True)
        
        results = test_reward_scheme(
            reward_fn=reward_fn,
            reward_name=reward_name,
            env_name=args.env,
            timesteps=args.timesteps,
            log_dir=scheme_dir,
            render=not args.no_render
        )
        
        all_results.append(results)
        
        # Save individual results
        with open(os.path.join(scheme_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    # Rank reward schemes
    print(f"\n{'='*60}")
    print("FINAL RANKINGS")
    print(f"{'='*60}")
    
    # Sort by success rate first, then by first success step
    ranked = sorted(all_results, key=lambda x: (
        -x["success_rate"],  # Higher is better
        x["first_success_step"] if x["first_success_step"] else float('inf')  # Lower is better
    ))
    
    print(f"\n{'Rank':<6}{'Reward Scheme':<25}{'Success Rate':<15}{'First Success':<15}{'Mean Reward':<15}")
    print("-" * 80)
    
    for i, result in enumerate(ranked, 1):
        first_success = str(result["first_success_step"]) if result["first_success_step"] else "Never"
        print(f"{i:<6}{result['reward_scheme']:<25}{result['success_rate']:<15.2%}{first_success:<15}{result['mean_reward']:<15.2f}")
    
    # Save summary
    summary = {
        "environment": args.env,
        "timesteps_per_test": args.timesteps,
        "timestamp": timestamp,
        "rankings": ranked
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if ranked[0]["success_rate"] > 0:
        print(f"✓ Best performing reward: {ranked[0]['reward_scheme']}")
        print(f"  - Success rate: {ranked[0]['success_rate']:.2%}")
        print(f"  - First success at step: {ranked[0]['first_success_step']}")
        print(f"\n  Next steps:")
        print(f"  1. Run full training with {ranked[0]['reward_scheme']} reward")
        print(f"  2. Fine-tune hyperparameters with this reward function")
    else:
        print("⚠ No reward scheme achieved success in this short test.")
        print("  Recommendations:")
        print("  1. Increase test duration to 50k steps")
        print("  2. Add more exploration (higher entropy coefficient)")
        print("  3. Consider curriculum learning approach")


if __name__ == "__main__":
    main()