"""
Skill definitions, registry, and state management.

Skills are trainable capabilities with:
- Configuration from YAML (category, metrics, thresholds)
- Runtime state (level, XP, accuracy history)
- Progression logic (level-up when threshold met)

Usage:
    from guild.skills import get_skill, get_registry, SkillConfig

    # Get a skill config
    skill = get_skill("logic_weaving")

    # List all skills
    skill_ids = list_skills()

    # Get skills by category
    reasoning_skills = skills_by_category(SkillCategory.REASONING)

    # State management (requires initialization)
    from guild.skills import init_state_manager, get_state_manager

    init_state_manager(Path("status"))
    manager = get_state_manager()
    manager.record_accuracy("logic_weaving", 0.85, step=1000)
"""

# Types
from guild.skills.types import (
    SkillCategory,
    MetricDefinition,
    SkillConfig,
    SkillState,
)

# Loader functions
from guild.skills.loader import (
    load_skill_config,
    discover_skills,
    load_all_skills,
    SkillLoader,
)

# Registry
from guild.skills.registry import (
    SkillRegistry,
    init_registry,
    get_registry,
    reset_registry,
    get_skill,
    list_skills,
    skills_by_category,
    skills_by_tag,
)

# State management
from guild.skills.state_manager import (
    AccuracyRecord,
    ProgressionRecord,
    SkillStateManager,
    init_state_manager,
    get_state_manager,
    reset_state_manager,
)

__all__ = [
    # Types
    "SkillCategory",
    "MetricDefinition",
    "SkillConfig",
    "SkillState",
    # Loader
    "load_skill_config",
    "discover_skills",
    "load_all_skills",
    "SkillLoader",
    # Registry
    "SkillRegistry",
    "init_registry",
    "get_registry",
    "reset_registry",
    "get_skill",
    "list_skills",
    "skills_by_category",
    "skills_by_tag",
    # State
    "AccuracyRecord",
    "ProgressionRecord",
    "SkillStateManager",
    "init_state_manager",
    "get_state_manager",
    "reset_state_manager",
]
