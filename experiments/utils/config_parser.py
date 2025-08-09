"""Configuration parser for experiment management."""

import os
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
from copy import deepcopy


class ConfigParser:
    """Parse and manage experiment configurations."""
    
    def __init__(self, base_config_path: str = None):
        """Initialize config parser.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config = {}
        if base_config_path:
            self.base_config = self.load_config(base_config_path)
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif save_path.suffix == '.json':
                json.dump(config, f, indent=2)
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations with later ones overriding earlier ones.
        
        Args:
            configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration
        """
        result = deepcopy(self.base_config)
        
        for config in configs:
            result = self._recursive_merge(result, config)
        
        return result
    
    def _recursive_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    @staticmethod
    def get_config_hash(config: Dict[str, Any]) -> str:
        """Generate a hash for configuration to track experiments.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            8-character hash string
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def parse_command_line_overrides(self, args: list) -> Dict[str, Any]:
        """Parse command line arguments for config overrides.
        
        Args:
            args: List of command line arguments in format key=value
            
        Returns:
            Dictionary of overrides
        """
        overrides = {}
        
        for arg in args:
            if '=' not in arg:
                continue
            
            key, value = arg.split('=', 1)
            keys = key.split('.')
            
            # Try to parse value as number or boolean
            try:
                value = eval(value)
            except:
                pass  # Keep as string
            
            # Build nested dictionary
            current = overrides
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        return overrides
    
    def validate_config(self, config: Dict[str, Any], required_fields: list) -> bool:
        """Validate that configuration contains required fields.
        
        Args:
            config: Configuration dictionary
            required_fields: List of required field paths (e.g., ['algorithm', 'training.epochs'])
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        for field_path in required_fields:
            keys = field_path.split('.')
            current = config
            
            for key in keys:
                if key not in current:
                    raise ValueError(f"Required field missing: {field_path}")
                current = current[key]
        
        return True


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for training scripts.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Unified training script for surgical RL')
    
    # Core arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--task', type=str, required=True,
                       choices=['needle_reach', 'peg_transfer'],
                       help='Task to train on')
    parser.add_argument('--algorithm', type=str,
                       choices=['bc', 'ppo', 'ppo_bc'],
                       help='Algorithm to use (overrides config)')
    
    # Directory arguments
    parser.add_argument('--output-dir', type=str, default='results/experiments',
                       help='Base directory for experiment outputs')
    parser.add_argument('--demo-dir', type=str, default='demonstrations',
                       help='Directory containing expert demonstrations')
    
    # Training arguments
    parser.add_argument('--seed', type=int,
                       help='Random seed (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for training')
    
    # Experiment management
    parser.add_argument('--experiment-name', type=str,
                       help='Name for this experiment run')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation, no training')
    
    # Config overrides
    parser.add_argument('--override', nargs='*', default=[],
                       help='Config overrides in format key=value')
    
    # Logging
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0=silent, 1=normal, 2=debug)')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    
    return parser