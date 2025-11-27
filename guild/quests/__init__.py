"""
Quest system - templates, instances, generation, and evaluation.

Quests are tasks for heroes to complete:
- QuestTemplate: Blueprint loaded from YAML configs
- QuestInstance: Concrete task with prompt and expected answer
- QuestResult: Outcome with combat result and XP

Usage:
    from guild.quests import (
        get_quest, list_quests,
        create_quest, evaluate_quest,
    )

    # Get a quest template
    template = get_quest("syllo_puzzle")

    # Generate an instance
    instance = create_quest(template)

    # Evaluate a response
    result = evaluate_quest(
        hero_id="hero_123",
        response="my answer",
        quest=instance,
        template=template,
    )
"""

# Types
from guild.quests.types import (
    QuestDifficulty,
    CombatResult,
    QuestTemplate,
    QuestInstance,
    QuestResult,
)

# Loader
from guild.quests.loader import (
    load_quest_template,
    discover_quest_templates,
    load_all_quest_templates,
    QuestLoader,
)

# Registry
from guild.quests.registry import (
    QuestRegistry,
    init_quest_registry,
    get_quest_registry,
    reset_quest_registry,
    get_quest,
    list_quests,
    quests_by_skill,
    quests_by_difficulty,
)

# Forge (generation)
from guild.quests.forge import (
    QuestGenerator,
    StaticGenerator,
    CallbackGenerator,
    QuestForge,
    get_forge,
    reset_forge,
    create_quest,
    create_quest_by_id,
    create_quest_for_skill,
    register_generator,
)

# Evaluator
from guild.quests.evaluator import (
    EvaluationContext,
    EvaluationOutcome,
    QuestEvaluator,
    ExactMatchEvaluator,
    ContainsEvaluator,
    CallbackEvaluator,
    QuestJudge,
    get_judge,
    reset_judge,
    evaluate_quest,
    register_evaluator,
)

__all__ = [
    # Types
    "QuestDifficulty",
    "CombatResult",
    "QuestTemplate",
    "QuestInstance",
    "QuestResult",
    # Loader
    "load_quest_template",
    "discover_quest_templates",
    "load_all_quest_templates",
    "QuestLoader",
    # Registry
    "QuestRegistry",
    "init_quest_registry",
    "get_quest_registry",
    "reset_quest_registry",
    "get_quest",
    "list_quests",
    "quests_by_skill",
    "quests_by_difficulty",
    # Forge
    "QuestGenerator",
    "StaticGenerator",
    "CallbackGenerator",
    "QuestForge",
    "get_forge",
    "reset_forge",
    "create_quest",
    "create_quest_by_id",
    "create_quest_for_skill",
    "register_generator",
    # Evaluator
    "EvaluationContext",
    "EvaluationOutcome",
    "QuestEvaluator",
    "ExactMatchEvaluator",
    "ContainsEvaluator",
    "CallbackEvaluator",
    "QuestJudge",
    "get_judge",
    "reset_judge",
    "evaluate_quest",
    "register_evaluator",
]
