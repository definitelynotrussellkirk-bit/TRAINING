"""
Skill loading and trainer access.

Two ways to access skills:
1. YAML configs (for Guild UI): load_skill_config(), SkillLoader
2. API trainers (for samples): get_trainer(), SKILL_REGISTRY
"""

from pathlib import Path
from typing import Optional, Type, TypeVar

from guild.config.loader import (
    get_config_dir,
    load_yaml,
    ConfigLoader,
)
from guild.skills.types import SkillConfig, SkillCategory, MetricDefinition
from guild.skills.contract import SkillClient, SkillDefinition, Batch


T = TypeVar('T')


# =============================================================================
# SKILL REGISTRY - Maps skill IDs to API URLs
# =============================================================================

SKILL_REGISTRY: dict[str, dict] = {
    "binary": {
        "api_url": "http://localhost:8090",
        "category": "math",
        "description": "Binary arithmetic with circled notation",
    },
    # DISABLED: Re-enable when ready
    # "syllo": {
    #     "api_url": "http://localhost:8080",
    #     "category": "reasoning",
    #     "description": "Syllable puzzles",
    # },
}


def get_trainer(skill_id: str) -> SkillClient:
    """
    Get a trainer (API client) for a skill.

    Usage:
        trainer = get_trainer("binary")
        batch = trainer.sample(level=5, count=100)

    Args:
        skill_id: Skill identifier (e.g., "binary", "syllo")

    Returns:
        SkillClient instance connected to the skill's API

    Raises:
        KeyError: If skill not in registry
    """
    if skill_id not in SKILL_REGISTRY:
        raise KeyError(
            f"Unknown skill: '{skill_id}'. "
            f"Available: {list(SKILL_REGISTRY.keys())}"
        )

    config = SKILL_REGISTRY[skill_id]
    return SkillClient(skill_id, config["api_url"])


def list_trainers() -> list[str]:
    """List all registered trainer skill IDs."""
    return list(SKILL_REGISTRY.keys())


def get_trainer_info(skill_id: str) -> dict:
    """Get registry info for a skill (without calling API)."""
    if skill_id not in SKILL_REGISTRY:
        raise KeyError(f"Unknown skill: '{skill_id}'")
    return SKILL_REGISTRY[skill_id].copy()


# =============================================================================
# YAML CONFIG LOADING (existing functionality)
# =============================================================================


def load_skill_config(skill_id: str, config_dir: Optional[Path] = None) -> SkillConfig:
    """
    Load a skill configuration from YAML.

    Args:
        skill_id: Skill identifier (e.g., "logic_weaving")
        config_dir: Optional config directory (defaults to GUILD_CONFIG_DIR)

    Returns:
        SkillConfig instance

    Raises:
        FileNotFoundError: If skill config file doesn't exist
        ValueError: If config is invalid
    """
    if config_dir is None:
        config_dir = get_config_dir()

    skill_path = config_dir / "skills" / f"{skill_id}.yaml"

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill config not found: {skill_path}")

    data = load_yaml(skill_path)
    return _dict_to_skill_config(data)


def _dict_to_skill_config(data: dict) -> SkillConfig:
    """Convert a dict to SkillConfig, handling enum conversion."""
    # Required fields
    skill_id = data.get("id")
    if not skill_id:
        raise ValueError("Skill config missing 'id' field")

    name = data.get("name", skill_id)
    description = data.get("description", "")

    # Category enum conversion
    category_str = data.get("category", "reasoning")
    try:
        category = SkillCategory(category_str)
    except ValueError:
        raise ValueError(f"Invalid skill category: {category_str}")

    # Optional fields with defaults
    tags = data.get("tags", [])
    metrics = data.get("metrics", ["accuracy"])
    primary_metric = data.get("primary_metric", "accuracy")

    # Accuracy thresholds - convert string keys to int
    thresholds_raw = data.get("accuracy_thresholds", {})
    accuracy_thresholds = {int(k): float(v) for k, v in thresholds_raw.items()}

    xp_multiplier = float(data.get("xp_multiplier", 1.0))

    # RPG flavor
    rpg_name = data.get("rpg_name")
    rpg_description = data.get("rpg_description")

    return SkillConfig(
        id=skill_id,
        name=name,
        description=description,
        category=category,
        tags=tags,
        metrics=metrics,
        primary_metric=primary_metric,
        accuracy_thresholds=accuracy_thresholds,
        xp_multiplier=xp_multiplier,
        rpg_name=rpg_name,
        rpg_description=rpg_description,
    )


def discover_skills(config_dir: Optional[Path] = None) -> list[str]:
    """
    Discover all skill IDs from config files.

    Returns:
        List of skill IDs (file stems from configs/skills/*.yaml)
    """
    if config_dir is None:
        config_dir = get_config_dir()

    skills_dir = config_dir / "skills"
    if not skills_dir.exists():
        return []

    skill_ids = []
    for path in skills_dir.glob("*.yaml"):
        if not path.name.startswith("_"):
            skill_ids.append(path.stem)

    return sorted(skill_ids)


def load_all_skills(config_dir: Optional[Path] = None) -> dict[str, SkillConfig]:
    """
    Load all skill configurations.

    Returns:
        Dict mapping skill_id to SkillConfig
    """
    if config_dir is None:
        config_dir = get_config_dir()

    skills = {}
    for skill_id in discover_skills(config_dir):
        try:
            skills[skill_id] = load_skill_config(skill_id, config_dir)
        except Exception as e:
            # Log warning but continue loading other skills
            import logging
            logging.warning(f"Failed to load skill '{skill_id}': {e}")

    return skills


class SkillLoader:
    """
    Cached skill configuration loader.

    Provides caching to avoid re-reading YAML files.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, SkillConfig] = {}
        self._discovered: Optional[list[str]] = None

    def load(self, skill_id: str, use_cache: bool = True) -> SkillConfig:
        """Load a skill config with optional caching."""
        if use_cache and skill_id in self._cache:
            return self._cache[skill_id]

        config = load_skill_config(skill_id, self.config_dir)

        if use_cache:
            self._cache[skill_id] = config

        return config

    def load_all(self, use_cache: bool = True) -> dict[str, SkillConfig]:
        """Load all skill configs with optional caching."""
        if use_cache and self._cache:
            # Return cached if we have any
            return self._cache.copy()

        skills = load_all_skills(self.config_dir)

        if use_cache:
            self._cache = skills.copy()

        return skills

    def discover(self, use_cache: bool = True) -> list[str]:
        """Discover skill IDs with optional caching."""
        if use_cache and self._discovered is not None:
            return self._discovered.copy()

        discovered = discover_skills(self.config_dir)

        if use_cache:
            self._discovered = discovered.copy()

        return discovered

    def exists(self, skill_id: str) -> bool:
        """Check if a skill config exists."""
        return skill_id in self.discover()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._discovered = None

    def invalidate(self, skill_id: str):
        """Invalidate cache for a specific skill."""
        self._cache.pop(skill_id, None)
