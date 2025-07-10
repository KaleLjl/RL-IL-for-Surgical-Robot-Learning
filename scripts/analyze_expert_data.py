#!/usr/bin/env python3
"""
Comprehensive analysis of PegTransfer expert data quality.
This script analyzes the expert data file to identify potential issues
that could cause BC training problems.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

def load_expert_data(file_path: str) -> Dict[str, Any]:
    """Load expert data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded expert data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"✗ Expert data file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading expert data: {e}")
        sys.exit(1)

def analyze_data_structure(data: Any) -> None:
    """Analyze the overall structure of the expert data."""
    print("\n" + "="*60)
    print("DATA STRUCTURE ANALYSIS")
    print("="*60)
    
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            if isinstance(value, (list, np.ndarray)):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    if hasattr(value[0], 'shape'):
                        print(f"  First element shape: {value[0].shape}")
            elif isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
    
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"First element keys: {list(data[0].keys())}")
                # Analyze structure of first episode
                for key, value in data[0].items():
                    print(f"\n{key} (from first episode):")
                    print(f"  Type: {type(value)}")
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  Length: {len(value)}")
                        if len(value) > 0:
                            print(f"  First element type: {type(value[0])}")
                            if hasattr(value[0], 'shape'):
                                print(f"  First element shape: {value[0].shape}")
                    elif isinstance(value, np.ndarray):
                        print(f"  Shape: {value.shape}")
                        print(f"  Dtype: {value.dtype}")
            elif hasattr(data[0], 'shape'):
                print(f"First element shape: {data[0].shape}")
    
    else:
        print(f"Unexpected data type: {type(data)}")
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
        if hasattr(data, '__len__'):
            print(f"Length: {len(data)}")

def check_for_anomalies(arr: np.ndarray, name: str) -> Dict[str, Any]:
    """Check for NaN, inf, and extreme values in array."""
    anomalies = {}
    
    # Check for NaN values
    nan_count = np.sum(np.isnan(arr))
    anomalies['nan_count'] = nan_count
    
    # Check for infinite values
    inf_count = np.sum(np.isinf(arr))
    anomalies['inf_count'] = inf_count
    
    # Check for extreme values (beyond 3 standard deviations)
    if not np.all(np.isnan(arr)) and not np.all(np.isinf(arr)):
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        extreme_mask = np.abs(arr - mean) > 3 * std
        extreme_count = np.sum(extreme_mask)
        anomalies['extreme_count'] = extreme_count
        anomalies['extreme_threshold'] = 3 * std
    else:
        anomalies['extreme_count'] = 0
        anomalies['extreme_threshold'] = 0
    
    # Print results
    if nan_count > 0 or inf_count > 0 or anomalies['extreme_count'] > 0:
        print(f"\n⚠️  ANOMALIES DETECTED in {name}:")
        if nan_count > 0:
            print(f"  - NaN values: {nan_count}")
        if inf_count > 0:
            print(f"  - Infinite values: {inf_count}")
        if anomalies['extreme_count'] > 0:
            print(f"  - Extreme values (>3σ): {anomalies['extreme_count']}")
    else:
        print(f"✓ No anomalies detected in {name}")
    
    return anomalies

def analyze_observations(observations: List[np.ndarray]) -> None:
    """Analyze observation data."""
    print("\n" + "="*60)
    print("OBSERVATION ANALYSIS")
    print("="*60)
    
    if not observations:
        print("✗ No observations found!")
        return
    
    # Convert to numpy array for analysis
    obs_array = np.array(observations)
    print(f"Observations shape: {obs_array.shape}")
    print(f"Observation dimension: {obs_array.shape[1] if len(obs_array.shape) > 1 else 'scalar'}")
    
    # Basic statistics
    print(f"\nObservation Statistics:")
    print(f"  Mean: {np.mean(obs_array, axis=0)}")
    print(f"  Std:  {np.std(obs_array, axis=0)}")
    print(f"  Min:  {np.min(obs_array, axis=0)}")
    print(f"  Max:  {np.max(obs_array, axis=0)}")
    
    # Check for anomalies
    check_for_anomalies(obs_array, "observations")
    
    # Check value ranges for specific components (if known structure)
    if obs_array.shape[1] >= 3:
        print(f"\nPosition components (assuming first 3 are XYZ):")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            if i < obs_array.shape[1]:
                print(f"  {axis}: [{np.min(obs_array[:, i]):.4f}, {np.max(obs_array[:, i]):.4f}]")
    
    # Check for constant values (might indicate sensor issues)
    constant_dims = []
    for i in range(obs_array.shape[1]):
        if np.std(obs_array[:, i]) < 1e-10:
            constant_dims.append(i)
    
    if constant_dims:
        print(f"\n⚠️  Constant observation dimensions: {constant_dims}")
        print("   This might indicate sensor issues or unnecessary features")
    else:
        print(f"✓ All observation dimensions show variation")

def analyze_actions(actions: List[np.ndarray]) -> None:
    """Analyze action data."""
    print("\n" + "="*60)
    print("ACTION ANALYSIS")
    print("="*60)
    
    if not actions:
        print("✗ No actions found!")
        return
    
    # Convert to numpy array for analysis
    action_array = np.array(actions)
    print(f"Actions shape: {action_array.shape}")
    print(f"Action dimension: {action_array.shape[1] if len(action_array.shape) > 1 else 'scalar'}")
    
    # Basic statistics
    print(f"\nAction Statistics:")
    print(f"  Mean: {np.mean(action_array, axis=0)}")
    print(f"  Std:  {np.std(action_array, axis=0)}")
    print(f"  Min:  {np.min(action_array, axis=0)}")
    print(f"  Max:  {np.max(action_array, axis=0)}")
    
    # Check for anomalies
    check_for_anomalies(action_array, "actions")
    
    # Check if actions are normalized (common range: [-1, 1])
    print(f"\nAction Normalization Check:")
    all_in_range = True
    for i in range(action_array.shape[1]):
        min_val = np.min(action_array[:, i])
        max_val = np.max(action_array[:, i])
        in_range = -1.1 <= min_val <= 1.1 and -1.1 <= max_val <= 1.1
        print(f"  Dim {i}: [{min_val:.4f}, {max_val:.4f}] {'✓' if in_range else '✗'}")
        if not in_range:
            all_in_range = False
    
    if all_in_range:
        print("✓ All actions appear to be normalized to [-1, 1] range")
    else:
        print("⚠️  Some actions may not be properly normalized")
    
    # Check for action diversity
    print(f"\nAction Diversity:")
    unique_actions = len(np.unique(action_array, axis=0))
    total_actions = len(action_array)
    diversity_ratio = unique_actions / total_actions
    print(f"  Unique actions: {unique_actions}/{total_actions} ({diversity_ratio:.2%})")
    
    if diversity_ratio < 0.1:
        print("⚠️  Low action diversity - check for repetitive behavior")
    elif diversity_ratio > 0.9:
        print("⚠️  Very high action diversity - check for noisy actions")
    else:
        print("✓ Good action diversity")

def analyze_episode_structure(data: Any) -> None:
    """Analyze episode structure and lengths."""
    print("\n" + "="*60)
    print("EPISODE STRUCTURE ANALYSIS")
    print("="*60)
    
    if isinstance(data, dict):
        # Look for episode-related keys
        episode_keys = [key for key in data.keys() if 'episode' in key.lower()]
        print(f"Episode-related keys: {episode_keys}")
        
        # Try to infer episode structure from observations and actions
        if 'observations' in data and 'actions' in data:
            obs_len = len(data['observations'])
            act_len = len(data['actions'])
            print(f"Total observations: {obs_len}")
            print(f"Total actions: {act_len}")
            
            if obs_len != act_len:
                print(f"⚠️  Observation-action length mismatch: {obs_len} vs {act_len}")
            else:
                print("✓ Observation-action lengths match")
        
        # Check if episode boundaries are marked
        if 'episode_starts' in data:
            episode_starts = data['episode_starts']
            print(f"Episode starts: {len(episode_starts)} episodes")
            
            # Calculate episode lengths
            episode_lengths = []
            for i in range(len(episode_starts) - 1):
                length = episode_starts[i + 1] - episode_starts[i]
                episode_lengths.append(length)
            
            # Add last episode length
            if episode_starts:
                last_length = obs_len - episode_starts[-1]
                episode_lengths.append(last_length)
            
            if episode_lengths:
                print(f"Episode length statistics:")
                print(f"  Mean: {np.mean(episode_lengths):.2f}")
                print(f"  Std:  {np.std(episode_lengths):.2f}")
                print(f"  Min:  {np.min(episode_lengths)}")
                print(f"  Max:  {np.max(episode_lengths)}")
                
                # Check for very short episodes
                short_episodes = [i for i, length in enumerate(episode_lengths) if length < 10]
                if short_episodes:
                    print(f"⚠️  Short episodes (<10 steps): {len(short_episodes)}")
                else:
                    print("✓ No unusually short episodes")
        
        # Check for terminals/dones
        if 'terminals' in data:
            terminals = data['terminals']
            terminal_count = np.sum(terminals)
            print(f"Terminal states: {terminal_count}")
            
            # Episode count from terminals
            episode_count_from_terminals = terminal_count
            print(f"Episodes (from terminals): {episode_count_from_terminals}")
    
    elif isinstance(data, list):
        print(f"Total episodes: {len(data)}")
        
        # Analyze episode lengths
        episode_lengths = []
        total_observations = 0
        total_actions = 0
        
        for i, episode in enumerate(data):
            if isinstance(episode, dict):
                obs_len = 0
                act_len = 0
                
                # Handle observations
                if 'obs' in episode:
                    obs_data = episode['obs']
                    if isinstance(obs_data, dict) and 'observation' in obs_data:
                        obs_len = len(obs_data['observation'])
                    elif hasattr(obs_data, '__len__'):
                        obs_len = len(obs_data)
                elif 'observations' in episode:
                    obs_len = len(episode['observations'])
                
                # Handle actions
                if 'acts' in episode:
                    act_len = len(episode['acts'])
                elif 'actions' in episode:
                    act_len = len(episode['actions'])
                
                if obs_len > 0 and act_len > 0:
                    episode_lengths.append(obs_len)
                    total_observations += obs_len
                    total_actions += act_len
                    
                    if obs_len != act_len:
                        print(f"⚠️  Episode {i}: Observation-action length mismatch: {obs_len} vs {act_len}")
        
        print(f"Total observations across all episodes: {total_observations}")
        print(f"Total actions across all episodes: {total_actions}")
        
        if episode_lengths:
            print(f"\nEpisode length statistics:")
            print(f"  Mean: {np.mean(episode_lengths):.2f}")
            print(f"  Std:  {np.std(episode_lengths):.2f}")
            print(f"  Min:  {np.min(episode_lengths)}")
            print(f"  Max:  {np.max(episode_lengths)}")
            
            # Check for very short episodes
            short_episodes = [i for i, length in enumerate(episode_lengths) if length < 10]
            if short_episodes:
                print(f"⚠️  Short episodes (<10 steps): {len(short_episodes)} episodes")
                print(f"    Episodes: {short_episodes}")
            else:
                print("✓ No unusually short episodes")
    
    else:
        print(f"Unexpected data structure for episode analysis: {type(data)}")

def generate_recommendations(data: Any) -> List[str]:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    # Extract all observations and actions
    all_observations, all_actions = extract_all_data(data)
    
    # Check if we have basic required data
    if not all_observations:
        recommendations.append("CRITICAL: No observations found in data")
    
    if not all_actions:
        recommendations.append("CRITICAL: No actions found in data")
    
    if all_observations and all_actions:
        obs_len = len(all_observations)
        act_len = len(all_actions)
        
        if obs_len != act_len:
            if isinstance(data, list) and obs_len == act_len + len(data):
                recommendations.append(f"STANDARD: Observations include terminal states (+1 per episode). BC training should use obs[:-1] with actions.")
            else:
                recommendations.append(f"FIX: Observation-action length mismatch ({obs_len} vs {act_len})")
        
        # Check for sufficient data
        if act_len < 1000:
            recommendations.append("CONSIDER: Dataset might be too small for robust BC training")
        elif act_len < 5000:
            recommendations.append("GOOD: Dataset size adequate for BC training")
        else:
            recommendations.append("EXCELLENT: Large dataset size - good for robust BC training")
        
        # Analyze actions for normalization
        if all_actions:
            action_array = np.array(all_actions)
            if np.any(np.abs(action_array) > 2):
                recommendations.append("CONSIDER: Actions might need normalization to [-1, 1] range")
            else:
                recommendations.append("GOOD: Actions are properly normalized")
    
    # Check for episode structure
    if isinstance(data, dict):
        if 'episode_starts' not in data and 'terminals' not in data:
            recommendations.append("CONSIDER: Adding episode boundary information for better training")
    elif isinstance(data, list):
        # List structure already provides episode boundaries
        pass
    
    return recommendations

def extract_all_data(data: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract all observations and actions from the data structure."""
    all_observations = []
    all_actions = []
    
    if isinstance(data, dict):
        # Try common key names
        for obs_key in ['observations', 'obs']:
            if obs_key in data:
                all_observations = data[obs_key]
                break
        for act_key in ['actions', 'acts']:
            if act_key in data:
                all_actions = data[act_key]
                break
    elif isinstance(data, list):
        # Extract from list of episodes
        for episode in data:
            if isinstance(episode, dict):
                # Handle observations
                if 'obs' in episode:
                    obs_data = episode['obs']
                    if isinstance(obs_data, dict):
                        # This is a dict observation (like {'observation': [...], 'achieved_goal': [...], 'desired_goal': [...]})
                        if 'observation' in obs_data:
                            # Add each timestep observation
                            for obs_step in obs_data['observation']:
                                all_observations.append(obs_step)
                        else:
                            # Flatten and concatenate all components
                            for i in range(len(list(obs_data.values())[0])):
                                obs_step = []
                                for key in sorted(obs_data.keys()):
                                    obs_step.extend(obs_data[key][i].flatten())
                                all_observations.append(np.array(obs_step))
                    else:
                        # Direct observation array
                        all_observations.extend(obs_data)
                elif 'observations' in episode:
                    all_observations.extend(episode['observations'])
                
                # Handle actions
                if 'acts' in episode:
                    # Add each timestep action
                    for act_step in episode['acts']:
                        all_actions.append(act_step)
                elif 'actions' in episode:
                    all_actions.extend(episode['actions'])
    
    return all_observations, all_actions

def main():
    """Main analysis function."""
    print("PegTransfer Expert Data Quality Analysis")
    print("="*60)
    
    # Load data
    data_path = "data/expert_data_peg_transfer.pkl"
    data = load_expert_data(data_path)
    
    # Analyze data structure
    analyze_data_structure(data)
    
    # Extract all observations and actions
    all_observations, all_actions = extract_all_data(data)
    
    # Analyze observations
    if all_observations:
        analyze_observations(all_observations)
    else:
        print("\n⚠️  No observations found in data")
    
    # Analyze actions
    if all_actions:
        analyze_actions(all_actions)
    else:
        print("\n⚠️  No actions found in data")
    
    # Analyze episode structure
    analyze_episode_structure(data)
    
    # Generate recommendations
    recommendations = generate_recommendations(data)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✓ No major issues detected. Data appears to be in good condition for BC training.")
    
    # Additional analysis for expert data
    print("\n" + "="*60)
    print("EXPERT DATA COMPATIBILITY ANALYSIS")
    print("="*60)
    
    if all_observations and all_actions:
        obs_count = len(all_observations)
        act_count = len(all_actions)
        
        print(f"Observation-Action Length Analysis:")
        print(f"  Total observations: {obs_count}")
        print(f"  Total actions: {act_count}")
        print(f"  Difference: {obs_count - act_count}")
        
        if obs_count > act_count:
            print(f"  ⚠️  Common in episodic data: observations include terminal states")
            print(f"  ⚠️  For BC training: need to align obs-action pairs")
            print(f"  ⚠️  Solution: Use observations[:-1] and actions, or actions and observations[1:]")
        
        # Check if this is the expected pattern (obs = act + 1 per episode)
        expected_diff = len(data) if isinstance(data, list) else 0
        if obs_count - act_count == expected_diff:
            print(f"  ✓ Pattern matches expected: +1 observation per episode (terminal states)")
        else:
            print(f"  ⚠️  Unexpected pattern: expected diff={expected_diff}, actual diff={obs_count - act_count}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if all_observations and all_actions:
        print(f"Dataset episodes: {len(data) if isinstance(data, list) else 'N/A'}")
        print(f"Total observations: {len(all_observations)}")
        print(f"Total actions: {len(all_actions)}")
        print(f"Observation dimension: {len(all_observations[0]) if all_observations else 'N/A'}")
        print(f"Action dimension: {len(all_actions[0]) if all_actions else 'N/A'}")
    
    print(f"Issues found: {len(recommendations)}")
    
    if len(recommendations) == 0:
        print("✓ Data quality: GOOD")
    elif len(recommendations) <= 2:
        print("⚠️  Data quality: MODERATE - Minor issues")
    else:
        print("✗ Data quality: POOR - Multiple issues need attention")

if __name__ == "__main__":
    main()