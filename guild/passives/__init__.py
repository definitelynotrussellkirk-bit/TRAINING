"""
Passives Module Registry - Auto-discovers and loads passive modules.

Passives are general abilities tested independently of active training.
See GUILD_VOCABULARY.md for the RPG terminology.

Usage:
    from guild.passives import get_passive, list_passives, get_all_passives

    # List available passives
    passives = list_passives()  # ['arithmetic', 'logic', 'counting', ...]

    # Get a specific passive
    passive = get_passive('arithmetic')
    problems = passive.generate_problems(count=5)

    # Check an answer
    is_correct = passive.check_answer(expected="42", got="The answer is 42")

To add a new passive:
1. Create a file in guild/passives/ (e.g., my_passive.py)
2. Create a class inheriting from PassiveModule
3. Define id, name, category, description
4. Implement generate_problems() and check_answer()
5. Done! It's auto-discovered.

Categories (from GUILD_VOCABULARY.md):
- logic: Boolean reasoning, deduction
- counting: Enumeration, frequency
- conversion: Format transformation
- string_craft: Text manipulation
- arithmetic: Basic number sense
- sequence: Pattern recognition
- memory: Fact retention, recall
- reasoning: Multi-step logic
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Type

from guild.passives.base import PassiveModule, PassiveConfig, PassiveTier

logger = logging.getLogger(__name__)

# Passives config cache
_passives_config = None

# Registry of discovered passives
_passive_registry: Dict[str, PassiveModule] = {}
_loaded = False


def _discover_passives():
    """Auto-discover all passive modules in this package."""
    global _loaded, _passive_registry

    if _loaded:
        return

    package_dir = Path(__file__).parent

    # Find all Python files in this directory (except base.py and __init__.py)
    for file in package_dir.glob("*.py"):
        if file.name.startswith("_") or file.name == "base.py":
            continue

        module_name = file.stem
        try:
            # Import the module
            module = importlib.import_module(f"guild.passives.{module_name}")

            # Find PassiveModule subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, PassiveModule) and
                    attr is not PassiveModule and
                    hasattr(attr, 'id') and attr.id is not None):

                    # Instantiate and register
                    try:
                        instance = attr()
                        if instance.enabled:
                            _passive_registry[instance.id] = instance
                            logger.debug(f"Registered passive: {instance.id}")
                    except Exception as e:
                        logger.warning(f"Failed to instantiate {attr_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load passive module {module_name}: {e}")

    _loaded = True
    logger.info(f"Discovered {len(_passive_registry)} passives: {list(_passive_registry.keys())}")


def list_passives() -> List[str]:
    """List all available passive IDs."""
    _discover_passives()
    return list(_passive_registry.keys())


def get_passive(passive_id: str) -> Optional[PassiveModule]:
    """Get a passive by ID."""
    _discover_passives()
    return _passive_registry.get(passive_id)


def get_all_passives() -> List[PassiveModule]:
    """Get all registered passives."""
    _discover_passives()
    return list(_passive_registry.values())


def get_passive_configs() -> List[PassiveConfig]:
    """Get configurations for all passives."""
    _discover_passives()
    return [p.get_config() for p in _passive_registry.values()]


def get_passives_by_category(category: str) -> List[PassiveModule]:
    """Get all passives in a category."""
    _discover_passives()
    return [p for p in _passive_registry.values() if p.category == category]


# =============================================================================
# TIER-BASED FUNCTIONS
# =============================================================================


def get_passives_config() -> dict:
    """
    Load passives configuration from configs/passives.yaml.

    Returns cached config on subsequent calls.
    """
    global _passives_config
    if _passives_config is not None:
        return _passives_config

    import yaml
    from core.paths import get_base_dir

    config_path = get_base_dir() / "configs" / "passives.yaml"

    if config_path.exists():
        try:
            with open(config_path) as f:
                _passives_config = yaml.safe_load(f)
                return _passives_config
        except Exception as e:
            logger.warning(f"Failed to load passives config: {e}")

    # Default config
    _passives_config = {
        "core": {"max_count": 5, "lite_problems": 5, "full_problems": 30},
        "extended": {"lite_problems": 5, "full_problems": 30},
        "queue": {"ordering": "tier_priority", "deduplicate": True},
    }
    return _passives_config


def get_core_passives(max_count: Optional[int] = None) -> List[PassiveModule]:
    """
    Get core (sentinel) passives, sorted by priority.

    Args:
        max_count: Maximum number to return. If None, uses config value.
                   Use -1 for no limit.

    Returns:
        List of core passives, sorted by priority (lower first)
    """
    _discover_passives()

    # Get max from config if not specified
    if max_count is None:
        config = get_passives_config()
        max_count = config.get("core", {}).get("max_count", 5)

    # Filter core passives and sort by priority
    core = [p for p in _passive_registry.values() if p.tier == PassiveTier.CORE]
    core.sort(key=lambda p: p.priority)

    # Apply max count limit
    if max_count > 0:
        core = core[:max_count]

    return core


def get_extended_passives() -> List[PassiveModule]:
    """
    Get extended (on-demand) passives, sorted by priority.

    Returns:
        List of extended passives, sorted by priority (lower first)
    """
    _discover_passives()
    extended = [p for p in _passive_registry.values() if p.tier == PassiveTier.EXTENDED]
    extended.sort(key=lambda p: p.priority)
    return extended


def get_passives_by_tier(tier: str) -> List[PassiveModule]:
    """
    Get all passives of a specific tier.

    Args:
        tier: "core" or "extended"

    Returns:
        List of passives in that tier, sorted by priority
    """
    _discover_passives()
    passives = [p for p in _passive_registry.values() if p.tier == tier]
    passives.sort(key=lambda p: p.priority)
    return passives


def list_core_passive_ids() -> List[str]:
    """List IDs of all core passives."""
    return [p.id for p in get_core_passives(max_count=-1)]


def list_extended_passive_ids() -> List[str]:
    """List IDs of all extended passives."""
    return [p.id for p in get_extended_passives()]


def get_tier_summary() -> dict:
    """
    Get summary of passives by tier.

    Returns:
        Dict with tier counts and IDs
    """
    _discover_passives()
    core = get_core_passives(max_count=-1)
    extended = get_extended_passives()

    return {
        "core": {
            "count": len(core),
            "ids": [p.id for p in core],
        },
        "extended": {
            "count": len(extended),
            "ids": [p.id for p in extended],
        },
        "total": len(_passive_registry),
    }


# Export base class for convenience
__all__ = [
    # Base classes
    'PassiveModule',
    'PassiveConfig',
    'PassiveTier',
    # Discovery
    'list_passives',
    'get_passive',
    'get_all_passives',
    'get_passive_configs',
    'get_passives_by_category',
    # Tier system
    'get_passives_config',
    'get_core_passives',
    'get_extended_passives',
    'get_passives_by_tier',
    'list_core_passive_ids',
    'list_extended_passive_ids',
    'get_tier_summary',
]
