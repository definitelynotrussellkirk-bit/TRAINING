"""
Skill definitions, registry, and state management.

Skills are trainable capabilities with:
- Configuration from YAML (category, metrics, thresholds)
- Runtime state (level, XP, accuracy history)
- Progression logic (level-up when threshold met)

Usage:
    from guild.skills import get_skill, get_trainer, SkillConfig

    # Get a skill config (all metadata from YAML)
    skill = get_skill("sy")
    print(skill.icon, skill.color, skill.max_level)

    # Get a trainer (API client) for generating samples
    trainer = get_trainer("sy")
    batch = trainer.sample(level=5, count=100)

    # List all skills
    skill_ids = list_skills()

    # Get skills by category
    reasoning_skills = skills_by_category(SkillCategory.REASONING)

    # State management (requires initialization)
    from guild.skills import init_state_manager, get_state_manager

    init_state_manager(Path("status"))
    manager = get_state_manager()
    manager.record_accuracy("sy", 0.85, step=1000)

SKILL ENGINE (NEW - 2025-11-28):
    The Skill Engine provides a unified interface for all skill operations.

    from guild.skills import get_engine

    engine = get_engine()

    # Get a skill (unified interface for training + eval)
    skill = engine.get("binary")

    # Generate training data
    training = skill.generate_training_batch(level=5, count=100)

    # Run eval and update state
    result, state = engine.run_eval("binary", model_answers, level=5)
"""

# Types
from guild.skills.types import (
    SkillCategory,
    MetricDefinition,
    SkillConfig,
    SkillState,
    SkillDisplay,
    SkillAPI,
    SkillEval,
)

# Loader functions
from guild.skills.loader import (
    load_skill_config,
    discover_skills,
    load_all_skills,
    SkillLoader,
    # Trainer access (uses YAML config for API URL)
    get_trainer,
    list_trainers,
    get_trainer_info,
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

# Verification
from guild.skills.verify import (
    VerificationStatus,
    VerificationIssue,
    SkillVerification,
    verify_skill,
    verify_all_skills,
    print_verification_report,
)

# Primitives (NEW - Skill Engine)
from guild.skills.primitives import (
    PrimitiveId,
    PrimitiveMeta,
    get_primitive,
    list_primitives,
    list_tracks,
    PRIMITIVE_CATALOG,
)

# Eval types (NEW - Skill Engine)
from guild.skills.eval_types import (
    EvalProblem,
    EvalBatch,
    EvalResult,
    EvalResultItem,
    score_batch,
)

# Skill ABC (NEW - Skill Engine)
from guild.skills.skill import Skill

# Composite skill (NEW - Skill Engine)
from guild.skills.composite import CompositeSkill, LocalSkill

# Engine (NEW - Skill Engine)
from guild.skills.engine import (
    SkillEngine,
    get_engine,
    reset_engine,
    init_engine,
)

__all__ = [
    # Types
    "SkillCategory",
    "MetricDefinition",
    "SkillConfig",
    "SkillState",
    "SkillDisplay",
    "SkillAPI",
    "SkillEval",
    # Loader
    "load_skill_config",
    "discover_skills",
    "load_all_skills",
    "SkillLoader",
    # Trainer access
    "get_trainer",
    "list_trainers",
    "get_trainer_info",
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
    # Verification
    "VerificationStatus",
    "VerificationIssue",
    "SkillVerification",
    "verify_skill",
    "verify_all_skills",
    "print_verification_report",
    # Primitives (NEW - Skill Engine)
    "PrimitiveId",
    "PrimitiveMeta",
    "get_primitive",
    "list_primitives",
    "list_tracks",
    "PRIMITIVE_CATALOG",
    # Eval types (NEW - Skill Engine)
    "EvalProblem",
    "EvalBatch",
    "EvalResult",
    "EvalResultItem",
    "score_batch",
    # Skill ABC (NEW - Skill Engine)
    "Skill",
    "CompositeSkill",
    "LocalSkill",
    # Engine (NEW - Skill Engine)
    "SkillEngine",
    "get_engine",
    "reset_engine",
    "init_engine",
]
