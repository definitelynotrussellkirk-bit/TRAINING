"""
Guild Heroes Module - Hero profiles and registry.

Heroes represent model architectures with their training defaults,
chat templates, and metadata. Each hero can have multiple campaigns.

Usage:
    from guild.heroes import get_hero, list_heroes, HeroProfile

    # List available heroes
    heroes = list_heroes()
    # ['dio-qwen3-0.6b']

    # Get hero profile
    hero = get_hero("dio-qwen3-0.6b")
    print(hero.name)  # "DIO"
    print(hero.rpg_name)  # "The Skeptic"
    print(hero.model.hf_name)  # "Qwen/Qwen3-0.6B"
    print(hero.model.size_b)  # 0.6
    print(hero.training_defaults.learning_rate)  # 0.0004

RPG Flavor:
    Every great adventure begins with choosing a hero.
    Each hero brings unique strengths and a destiny to fulfill.
    The Guild Hall displays portraits of all registered champions.
"""

from .types import (
    HeroProfile,
    ModelSpec,
    TrainingDefaults,
    QLoRAConfig,
    ChatTemplate,
    VRAMProfile,
    DisplayConfig,
)

from .registry import (
    HeroRegistry,
    HeroNotFoundError,
    HeroConfigError,
    get_registry,
    list_heroes,
    get_hero,
)

__all__ = [
    # Types
    "HeroProfile",
    "ModelSpec",
    "TrainingDefaults",
    "QLoRAConfig",
    "ChatTemplate",
    "VRAMProfile",
    "DisplayConfig",
    # Registry
    "HeroRegistry",
    "HeroNotFoundError",
    "HeroConfigError",
    "get_registry",
    "list_heroes",
    "get_hero",
]
