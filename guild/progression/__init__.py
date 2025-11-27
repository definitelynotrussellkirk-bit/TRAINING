"""
Progression system - XP, levels, status effects, and hero management.

The progression system tracks:
- XP accumulation and level thresholds
- Status effects (buffs/debuffs)
- Hero state persistence

Usage:
    from guild.progression import (
        init_hero_manager, get_hero_manager,
        record_result, get_hero_status,
    )

    # Initialize
    init_hero_manager(Path("status"))

    # Record quest result
    summary = record_result(quest_result)

    # Get hero status
    status = get_hero_status()
"""

# Types
from guild.progression.types import (
    EffectType,
    StatusEffect,
    EffectDefinition,
    EffectRuleConfig,
    EffectRuleState,
    HeroIdentity,
    HeroState,
)

# XP system
from guild.progression.xp import (
    LevelConfig,
    XPModifiers,
    XPCalculator,
    get_calculator as get_xp_calculator,
    reset_calculator as reset_xp_calculator,
    init_calculator as init_xp_calculator,
    calculate_xp,
    xp_for_level,
    level_from_xp,
)

# Effects system
from guild.progression.effects import (
    EffectRegistry,
    EffectEvaluator,
    EffectManager,
    get_effect_manager,
    reset_effect_manager,
    init_effect_manager,
    apply_effect,
    update_effects,
    get_xp_multiplier,
)

# Hero manager
from guild.progression.hero_manager import (
    HeroManager,
    init_hero_manager,
    get_hero_manager,
    reset_hero_manager,
    get_hero,
    record_result,
    get_hero_status,
)

__all__ = [
    # Types
    "EffectType",
    "StatusEffect",
    "EffectDefinition",
    "EffectRuleConfig",
    "EffectRuleState",
    "HeroIdentity",
    "HeroState",
    # XP
    "LevelConfig",
    "XPModifiers",
    "XPCalculator",
    "get_xp_calculator",
    "reset_xp_calculator",
    "init_xp_calculator",
    "calculate_xp",
    "xp_for_level",
    "level_from_xp",
    # Effects
    "EffectRegistry",
    "EffectEvaluator",
    "EffectManager",
    "get_effect_manager",
    "reset_effect_manager",
    "init_effect_manager",
    "apply_effect",
    "update_effects",
    "get_xp_multiplier",
    # Hero
    "HeroManager",
    "init_hero_manager",
    "get_hero_manager",
    "reset_hero_manager",
    "get_hero",
    "record_result",
    "get_hero_status",
]
