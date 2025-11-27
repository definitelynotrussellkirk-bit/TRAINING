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

from guild.passives.base import PassiveModule, PassiveConfig

logger = logging.getLogger(__name__)

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


# Export base class for convenience
__all__ = [
    'PassiveModule',
    'PassiveConfig',
    'list_passives',
    'get_passive',
    'get_all_passives',
    'get_passive_configs',
    'get_passives_by_category',
]
