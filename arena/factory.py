"""Trainer factory - creates the right trainer based on hero profile."""

from pathlib import Path
from typing import Any, Dict

from arena.trainers.base import BaseTrainer


def create_trainer(hero_config: Dict[str, Any], campaign_path: Path) -> BaseTrainer:
    """
    Create a trainer based on the hero's configuration.
    
    Args:
        hero_config: Hero profile from configs/heroes/*.yaml
        campaign_path: Path to campaign directory
        
    Returns:
        Appropriate trainer instance for the hero
        
    Raises:
        ValueError: If trainer type is unknown
    """
    trainer_config = hero_config.get("trainer", {})
    trainer_type = trainer_config.get("type", "ultimate")
    
    if trainer_type == "flo":
        from arena.trainers.flo import FLOTrainer
        return FLOTrainer(hero_config, campaign_path)
    
    elif trainer_type == "ultimate":
        # UltimateTrainer wrapper - uses existing core/train.py
        from arena.trainers.ultimate import UltimateTrainerWrapper
        return UltimateTrainerWrapper(hero_config, campaign_path)
    
    elif trainer_type == "deepspeed":
        # DeepSpeed ZeRO-3 - for very large models
        from arena.trainers.deepspeed import DeepSpeedTrainer
        return DeepSpeedTrainer(hero_config, campaign_path)
    
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def load_hero_config(hero_id: str) -> Dict[str, Any]:
    """
    Load hero configuration from configs/heroes/{hero_id}.yaml
    
    Args:
        hero_id: Hero identifier (e.g., "titan-qwen3-4b", "dio-qwen3-0.6b")
        
    Returns:
        Hero configuration dictionary
    """
    import yaml
    from core.paths import get_base_dir
    
    base_dir = get_base_dir()
    config_path = base_dir / "configs" / "heroes" / f"{hero_id}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Hero config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_campaign_config(campaign_path: Path) -> Dict[str, Any]:
    """
    Load campaign configuration from campaign_path/campaign.json

    Args:
        campaign_path: Path to campaign directory

    Returns:
        Campaign configuration dictionary
    """
    import json

    config_file = campaign_path / "campaign.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Campaign config not found: {config_file}")
    
    with open(config_file, "r") as f:
        return json.load(f)
