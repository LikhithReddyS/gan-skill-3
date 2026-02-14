"""
Configuration utility module for loading YAML config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path to save the YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class Config:
    """
    Configuration class that allows attribute-style access to config values.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def get_config(config_path: str) -> Config:
    """
    Load a YAML configuration file and return a Config object.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with attribute-style access.
    """
    config_dict = load_config(config_path)
    return Config(config_dict)
