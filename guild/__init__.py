"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.

The Guild framework provides:
- Skills: Trainable capabilities with metrics and progression
- Quests: Task instances with templates and evaluation
- Progression: XP, levels, and status effects
- Facilities: Hardware abstraction for multi-machine setups
- Runs: Unified training/eval/audit execution
- Incidents: Structured error tracking
- Combat: Result calculation (CRIT/HIT/MISS)
- Consistency: World model validation
"""

__version__ = "0.1.0"

# Facility resolution (core functionality)
from guild.facilities.resolver import (
    init_resolver,
    resolve,
    get_facility,
    set_current_facility,
    get_resolver,
    reset_resolver,
)

# Skill management
from guild.skills import (
    # Types
    SkillCategory,
    SkillConfig,
    SkillState,
    # Registry
    init_registry as init_skill_registry,
    get_registry as get_skill_registry,
    reset_registry as reset_skill_registry,
    get_skill,
    list_skills,
    skills_by_category,
    skills_by_tag,
    # State management
    init_state_manager as init_skill_state_manager,
    get_state_manager as get_skill_state_manager,
    reset_state_manager as reset_skill_state_manager,
)

# Quest management
from guild.quests import (
    # Types
    QuestDifficulty,
    CombatResult,
    QuestTemplate,
    QuestInstance,
    QuestResult,
    # Registry
    init_quest_registry,
    get_quest_registry,
    reset_quest_registry,
    get_quest,
    list_quests,
    quests_by_skill,
    # Forge
    get_forge as get_quest_forge,
    reset_forge as reset_quest_forge,
    create_quest,
    create_quest_by_id,
    # Evaluator
    get_judge as get_quest_judge,
    reset_judge as reset_quest_judge,
    evaluate_quest,
)

__all__ = [
    # Facilities
    "init_resolver",
    "resolve",
    "get_facility",
    "set_current_facility",
    "get_resolver",
    "reset_resolver",
    # Skills - types
    "SkillCategory",
    "SkillConfig",
    "SkillState",
    # Skills - registry
    "init_skill_registry",
    "get_skill_registry",
    "reset_skill_registry",
    "get_skill",
    "list_skills",
    "skills_by_category",
    "skills_by_tag",
    # Skills - state
    "init_skill_state_manager",
    "get_skill_state_manager",
    "reset_skill_state_manager",
    # Quests - types
    "QuestDifficulty",
    "CombatResult",
    "QuestTemplate",
    "QuestInstance",
    "QuestResult",
    # Quests - registry
    "init_quest_registry",
    "get_quest_registry",
    "reset_quest_registry",
    "get_quest",
    "list_quests",
    "quests_by_skill",
    # Quests - forge
    "get_quest_forge",
    "reset_quest_forge",
    "create_quest",
    "create_quest_by_id",
    # Quests - evaluator
    "get_quest_judge",
    "reset_quest_judge",
    "evaluate_quest",
]
