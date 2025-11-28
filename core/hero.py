"""
Active Hero Resolution - Single source of truth for current hero identity.

This module provides the active hero's name, icon, model info, etc.
All UI and system components should use this instead of hardcoding "DIO" or model names.

Usage:
    from core.hero import get_active_hero, get_hero_name, get_hero_icon

    hero = get_active_hero()
    print(hero["name"])  # "FLO"
    print(hero["icon"])  # "emoji or path"
    print(hero["model_name"])  # "Qwen3-4B"
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed, using JSON fallback")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def _get_base_dir() -> Path:
    """Get base directory."""
    try:
        from core.paths import get_base_dir
        return get_base_dir()
    except ImportError:
        return Path(__file__).parent.parent


@lru_cache(maxsize=1)
def get_active_campaign() -> Dict[str, Any]:
    """Get the active campaign info from control/active_campaign.json."""
    base_dir = _get_base_dir()
    campaign_file = base_dir / "control" / "active_campaign.json"

    if not campaign_file.exists():
        logger.warning("No active campaign file found")
        return {}

    try:
        with open(campaign_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load active campaign: {e}")
        return {}


def get_hero_config(hero_id: str) -> Dict[str, Any]:
    """Load hero configuration from configs/heroes/{hero_id}.yaml."""
    base_dir = _get_base_dir()
    hero_file = base_dir / "configs" / "heroes" / f"{hero_id}.yaml"

    if not hero_file.exists():
        logger.warning(f"Hero config not found: {hero_file}")
        return {}

    return _load_yaml(hero_file)


def get_active_hero() -> Dict[str, Any]:
    """
    Get the active hero's full info.

    Returns dict with:
        - name: Hero name (e.g., "FLO")
        - rpg_name: RPG title (e.g., "The Compassionate")
        - description: Hero description
        - icon: Emoji icon for the hero
        - model_name: Base model name (e.g., "Qwen3-4B")
        - model_path: Path to model
        - hero_id: Hero config ID
        - campaign_id: Active campaign ID
    """
    # Clear cache to get fresh data
    get_active_campaign.cache_clear()

    campaign = get_active_campaign()
    hero_id = campaign.get("hero_id", "dio-qwen3-0.6b")  # Default to DIO if no campaign

    config = get_hero_config(hero_id)

    # Extract model info
    model_info = config.get("model", {})
    model_name = model_info.get("hf_name", "Unknown")

    # Parse model display name from path
    if "/" in model_name:
        model_display = model_name.split("/")[-1]
    else:
        model_display = model_name

    # Map hero to icon (can be extended)
    icon_map = {
        "dio-qwen3-0.6b": "🧔🏽",
        "titan-qwen3-4b": "🧙‍♂️",
    }

    return {
        "name": config.get("name", "Hero"),
        "rpg_name": config.get("rpg_name", "The Apprentice"),
        "description": config.get("description", ""),
        "icon": icon_map.get(hero_id, "🦸"),
        "model_name": model_display,
        "model_family": model_info.get("family", "unknown"),
        "model_size": model_info.get("size_b", 0),
        "hero_id": hero_id,
        "campaign_id": campaign.get("campaign_id", ""),
        "campaign_path": campaign.get("campaign_path", ""),
    }


# Convenience functions
def get_hero_name() -> str:
    """Get the active hero's name."""
    return get_active_hero()["name"]


def get_hero_icon() -> str:
    """Get the active hero's icon."""
    return get_active_hero()["icon"]


def get_hero_title() -> str:
    """Get the active hero's RPG title."""
    return get_active_hero()["rpg_name"]


def get_model_name() -> str:
    """Get the active hero's model name."""
    return get_active_hero()["model_name"]


if __name__ == "__main__":
    import pprint
    hero = get_active_hero()
    print("Active Hero:")
    pprint.pprint(hero)
