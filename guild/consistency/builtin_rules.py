"""
Built-in consistency rules for guild entities.

Provides pre-defined rules for:
- Hero: Level, XP, health, temporal order
- Quest: Difficulty, status, skills
- Run: Type, status, progress
- Incident: Category, status, severity

Usage:
    from guild.consistency.builtin_rules import (
        get_hero_rules,
        get_quest_rules,
        get_run_rules,
        get_incident_rules,
        get_all_rules,
    )

    checker = ConsistencyChecker()
    checker.add_rules(get_all_rules())
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from guild.consistency.rules import (
    ConsistencyRule,
    RuleBuilder,
    RuleViolation,
    RuleCategory,
    RuleSeverity,
)


# =============================================================================
# Hero Rules
# =============================================================================

def _check_hero_timestamps(entity: Any, context: Dict) -> Optional[RuleViolation]:
    """Check that hero created_at < updated_at."""
    created = None
    updated = None

    # Handle both dict and object access
    if hasattr(entity, 'created_at'):
        created = getattr(entity, 'created_at')
        updated = getattr(entity, 'updated_at', None)
    elif isinstance(entity, dict):
        created = entity.get('created_at')
        updated = entity.get('updated_at')

    if created and updated and created > updated:
        return RuleViolation(
            rule_id="hero_temporal_order",
            rule_name="Hero Temporal Order",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.TEMPORAL,
            message=f"created_at must be before updated_at",
            details={"created_at": str(created), "updated_at": str(updated)},
        )
    return None


def _check_hero_name_not_empty(entity: Any, context: Dict) -> Optional[RuleViolation]:
    """Check that hero name is not empty string."""
    name = None
    if hasattr(entity, 'name'):
        name = getattr(entity, 'name')
    elif isinstance(entity, dict):
        name = entity.get('name')

    if name is not None and isinstance(name, str) and name.strip() == "":
        return RuleViolation(
            rule_id="hero_name_not_empty",
            rule_name="Hero Name Not Empty",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Hero name must not be empty",
            details={"name": name},
        )
    return None


def get_hero_rules() -> List[ConsistencyRule]:
    """Get built-in hero consistency rules."""
    return [
        # ID required
        (RuleBuilder("hero_id_required", "hero")
            .name("Hero ID Required")
            .description("Hero must have an ID")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.CRITICAL)
            .field("id")
            .not_null()
            .tag("hero", "required")
            .build()),

        # Level must be positive
        (RuleBuilder("hero_level_positive", "hero")
            .name("Hero Level Positive")
            .description("Hero level must be >= 1")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("level")
            .min_value(1)
            .tag("hero", "progression")
            .build()),

        # XP must be non-negative
        (RuleBuilder("hero_xp_non_negative", "hero")
            .name("Hero XP Non-Negative")
            .description("Hero XP must be >= 0")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("xp")
            .non_negative()
            .tag("hero", "progression")
            .build()),

        # Health check (if present)
        (RuleBuilder("hero_health_range", "hero")
            .name("Hero Health Range")
            .description("Hero health must be in valid range")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.WARNING)
            .field("health")
            .in_range(0, 100)
            .tag("hero", "status")
            .build()),

        # Temporal order
        ConsistencyRule(
            id="hero_temporal_order",
            name="Hero Temporal Order",
            description="created_at must be before updated_at",
            category=RuleCategory.TEMPORAL,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            check_fn=_check_hero_timestamps,
            tags=["hero", "temporal"],
        ),

        # Name not empty (custom check for empty string)
        ConsistencyRule(
            id="hero_name_not_empty",
            name="Hero Name Not Empty",
            description="Hero name must not be empty string",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            check_fn=_check_hero_name_not_empty,
            tags=["hero", "required"],
        ),
    ]


# =============================================================================
# Quest Rules
# =============================================================================

VALID_QUEST_STATUSES = ["pending", "active", "completed", "failed", "expired"]


def _check_quest_has_skills(entity: Any, context: Dict) -> Optional[RuleViolation]:
    """Check that quest has at least one skill."""
    skills = None
    if hasattr(entity, 'skills'):
        skills = getattr(entity, 'skills')
    elif isinstance(entity, dict):
        skills = entity.get('skills')

    if skills is not None and (not isinstance(skills, (list, tuple)) or len(skills) == 0):
        return RuleViolation(
            rule_id="quest_has_skills",
            rule_name="Quest Has Skills",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Quest must have at least one skill",
            details={"skills": skills},
        )
    return None


def get_quest_rules() -> List[ConsistencyRule]:
    """Get built-in quest consistency rules."""
    return [
        # ID required
        (RuleBuilder("quest_id_required", "quest")
            .name("Quest ID Required")
            .description("Quest must have an ID")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.CRITICAL)
            .field("id")
            .not_null()
            .tag("quest", "required")
            .build()),

        # Template ID required
        (RuleBuilder("quest_template_required", "quest")
            .name("Quest Template Required")
            .description("Quest must have a template_id")
            .category(RuleCategory.REFERENCE)
            .severity(RuleSeverity.ERROR)
            .field("template_id")
            .not_null()
            .tag("quest", "required")
            .build()),

        # Difficulty in valid range (1-10)
        (RuleBuilder("quest_difficulty_range", "quest")
            .name("Quest Difficulty Range")
            .description("Quest difficulty must be 1-10")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("difficulty_level")
            .in_range(1, 10)
            .tag("quest", "difficulty")
            .build()),

        # Status valid
        (RuleBuilder("quest_status_valid", "quest")
            .name("Quest Status Valid")
            .description("Quest status must be valid")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("status")
            .in_set(VALID_QUEST_STATUSES)
            .tag("quest", "status")
            .build()),

        # Has skills
        ConsistencyRule(
            id="quest_has_skills",
            name="Quest Has Skills",
            description="Quest must have at least one skill",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="quest",
            check_fn=_check_quest_has_skills,
            tags=["quest", "required"],
        ),
    ]


# =============================================================================
# Run Rules
# =============================================================================

VALID_RUN_TYPES = ["training", "evaluation", "audit", "benchmark"]
VALID_RUN_STATUSES = ["pending", "running", "paused", "completed", "failed", "cancelled"]


def _check_run_progress_range(entity: Any, context: Dict) -> Optional[RuleViolation]:
    """Check that run progress is in 0-100 range."""
    progress = None
    if hasattr(entity, 'progress'):
        progress = getattr(entity, 'progress')
    elif isinstance(entity, dict):
        progress = entity.get('progress')

    if progress is not None:
        if not isinstance(progress, (int, float)) or progress < 0 or progress > 100:
            return RuleViolation(
                rule_id="run_progress_range",
                rule_name="Run Progress Range",
                severity=RuleSeverity.WARNING,
                category=RuleCategory.STATE,
                message=f"Run progress ({progress}) must be in 0-100 range",
                details={"progress": progress},
            )
    return None


def get_run_rules() -> List[ConsistencyRule]:
    """Get built-in run consistency rules."""
    return [
        # ID required
        (RuleBuilder("run_id_required", "run")
            .name("Run ID Required")
            .description("Run must have an ID")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.CRITICAL)
            .field("run_id")
            .not_null()
            .tag("run", "required")
            .build()),

        # Status valid
        (RuleBuilder("run_status_valid", "run")
            .name("Run Status Valid")
            .description("Run status must be valid")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("status")
            .in_set(VALID_RUN_STATUSES)
            .tag("run", "status")
            .build()),

        # Progress range
        ConsistencyRule(
            id="run_progress_range",
            name="Run Progress Range",
            description="Run progress must be 0-100",
            category=RuleCategory.STATE,
            severity=RuleSeverity.WARNING,
            entity_type="run",
            check_fn=_check_run_progress_range,
            tags=["run", "progress"],
        ),
    ]


# =============================================================================
# Incident Rules
# =============================================================================

VALID_INCIDENT_CATEGORIES = [
    "performance", "quality", "system", "data", "model", "training"
]
VALID_INCIDENT_STATUSES = ["open", "investigating", "mitigated", "resolved", "closed"]
VALID_INCIDENT_SEVERITIES = ["low", "medium", "high", "critical"]


def get_incident_rules() -> List[ConsistencyRule]:
    """Get built-in incident consistency rules."""
    return [
        # ID required
        (RuleBuilder("incident_id_required", "incident")
            .name("Incident ID Required")
            .description("Incident must have an ID")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.CRITICAL)
            .field("id")
            .not_null()
            .tag("incident", "required")
            .build()),

        # Category valid
        (RuleBuilder("incident_category_valid", "incident")
            .name("Incident Category Valid")
            .description("Incident category must be valid")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("category")
            .in_set(VALID_INCIDENT_CATEGORIES)
            .tag("incident", "category")
            .build()),

        # Status valid
        (RuleBuilder("incident_status_valid", "incident")
            .name("Incident Status Valid")
            .description("Incident status must be valid")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("status")
            .in_set(VALID_INCIDENT_STATUSES)
            .tag("incident", "status")
            .build()),

        # Severity valid
        (RuleBuilder("incident_severity_valid", "incident")
            .name("Incident Severity Valid")
            .description("Incident severity must be valid")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("severity")
            .in_set(VALID_INCIDENT_SEVERITIES)
            .tag("incident", "severity")
            .build()),

        # Message required
        (RuleBuilder("incident_message_required", "incident")
            .name("Incident Message Required")
            .description("Incident must have a message")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("message")
            .not_null()
            .tag("incident", "required")
            .build()),
    ]


# =============================================================================
# Skill Rules
# =============================================================================

def get_skill_rules() -> List[ConsistencyRule]:
    """Get built-in skill consistency rules."""
    return [
        # ID required
        (RuleBuilder("skill_id_required", "skill")
            .name("Skill ID Required")
            .description("Skill must have an ID")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.CRITICAL)
            .field("id")
            .not_null()
            .tag("skill", "required")
            .build()),

        # Level valid range
        (RuleBuilder("skill_level_range", "skill")
            .name("Skill Level Range")
            .description("Skill level must be 1-10")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("level")
            .in_range(1, 10)
            .tag("skill", "progression")
            .build()),

        # Accuracy in 0-1 range
        (RuleBuilder("skill_accuracy_range", "skill")
            .name("Skill Accuracy Range")
            .description("Skill accuracy must be 0.0-1.0")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("accuracy")
            .in_range(0.0, 1.0)
            .tag("skill", "metrics")
            .build()),
    ]


# =============================================================================
# All Rules
# =============================================================================

def get_all_rules() -> List[ConsistencyRule]:
    """Get all built-in consistency rules."""
    return (
        get_hero_rules() +
        get_quest_rules() +
        get_run_rules() +
        get_incident_rules() +
        get_skill_rules()
    )


def get_rules_by_entity(entity_type: str) -> List[ConsistencyRule]:
    """Get rules for a specific entity type."""
    rule_getters = {
        "hero": get_hero_rules,
        "quest": get_quest_rules,
        "run": get_run_rules,
        "incident": get_incident_rules,
        "skill": get_skill_rules,
    }

    getter = rule_getters.get(entity_type)
    if getter:
        return getter()
    return []
