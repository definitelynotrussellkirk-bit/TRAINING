"""
Consistency checking system.

The consistency module provides:
- ConsistencyRule: Rule definitions with field constraints or custom checks
- RuleBuilder: Fluent API for building rules
- ConsistencyChecker: Engine for checking entities against rules
- Built-in rules for guild entities (hero, run, quest, incident)

Rule Categories:
- STATE: State validity (health > 0, level > 0)
- TRANSITION: State transition rules
- REFERENCE: Cross-reference integrity
- INVARIANT: System-wide invariants
- TEMPORAL: Time-based rules (created < updated)

Severities:
- WARNING: Log and continue
- ERROR: Log, may continue
- CRITICAL: Must stop, data integrity at risk

Example:
    from guild.consistency import (
        ConsistencyChecker,
        RuleBuilder,
        RuleCategory,
        RuleSeverity,
        check_entity,
        check_all,
    )

    # Create a rule using the builder
    rule = (RuleBuilder("hero_level_positive", "hero")
        .name("Hero Level Positive")
        .description("Hero level must be positive")
        .category(RuleCategory.STATE)
        .severity(RuleSeverity.ERROR)
        .field("level")
        .positive()
        .build())

    # Add to checker
    checker = ConsistencyChecker()
    checker.add_rule(rule)

    # Check an entity
    result = checker.check_entity(hero, "hero")
    if not result.passed:
        for v in result.violations:
            print(f"{v.severity}: {v.message}")
"""

# Rule types
from guild.consistency.rules import (
    RuleCategory,
    RuleSeverity,
    RuleViolation,
    ConsistencyRule,
    RuleBuilder,
)

# Checker
from guild.consistency.checker import (
    CheckResult,
    ConsistencyChecker,
    init_consistency_checker,
    get_consistency_checker,
    reset_consistency_checker,
    check_entity,
    check_all,
)

# Built-in rules
from guild.consistency.builtin_rules import (
    get_hero_rules,
    get_quest_rules,
    get_run_rules,
    get_incident_rules,
    get_skill_rules,
    get_all_rules,
    get_rules_by_entity,
)

__all__ = [
    # Rule types
    "RuleCategory",
    "RuleSeverity",
    "RuleViolation",
    "ConsistencyRule",
    "RuleBuilder",
    # Checker
    "CheckResult",
    "ConsistencyChecker",
    "init_consistency_checker",
    "get_consistency_checker",
    "reset_consistency_checker",
    "check_entity",
    "check_all",
    # Built-in rules
    "get_hero_rules",
    "get_quest_rules",
    "get_run_rules",
    "get_incident_rules",
    "get_skill_rules",
    "get_all_rules",
    "get_rules_by_entity",
]
