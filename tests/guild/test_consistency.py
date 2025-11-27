"""Tests for consistency checking system."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from guild.consistency.rules import (
    RuleCategory,
    RuleSeverity,
    RuleViolation,
    ConsistencyRule,
    RuleBuilder,
)
from guild.consistency.checker import (
    CheckResult,
    ConsistencyChecker,
    init_consistency_checker,
    get_consistency_checker,
    reset_consistency_checker,
    check_entity,
    check_all,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_consistency_checker()
    yield
    reset_consistency_checker()


@pytest.fixture
def sample_hero():
    """Create a sample hero dict."""
    return {
        "id": "hero_001",
        "name": "Test Hero",
        "level": 5,
        "xp": 1500,
        "health": 100,
        "created_at": datetime.now() - timedelta(days=10),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def invalid_hero():
    """Create an invalid hero dict."""
    return {
        "id": "hero_002",
        "name": "",
        "level": -1,
        "xp": -500,
        "health": 0,
        "created_at": datetime.now(),  # Created after updated
        "updated_at": datetime.now() - timedelta(days=1),
    }


@pytest.fixture
def sample_quest():
    """Create a sample quest dict."""
    return {
        "id": "quest_001",
        "template_id": "test_template",
        "skills": ["logic", "reasoning"],
        "difficulty": 3,
        "status": "pending",
    }


# =============================================================================
# Rule Types Tests
# =============================================================================

class TestRuleEnums:
    """Tests for rule enumerations."""

    def test_rule_category_values(self):
        assert RuleCategory.STATE.value == "state"
        assert RuleCategory.TRANSITION.value == "transition"
        assert RuleCategory.REFERENCE.value == "reference"
        assert RuleCategory.INVARIANT.value == "invariant"
        assert RuleCategory.TEMPORAL.value == "temporal"

    def test_rule_severity_values(self):
        assert RuleSeverity.WARNING.value == "warning"
        assert RuleSeverity.ERROR.value == "error"
        assert RuleSeverity.CRITICAL.value == "critical"


class TestRuleViolation:
    """Tests for RuleViolation."""

    def test_create_violation(self):
        violation = RuleViolation(
            rule_id="test_rule",
            rule_name="Test Rule",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Something is wrong",
        )

        assert violation.rule_id == "test_rule"
        assert violation.severity == RuleSeverity.ERROR
        assert violation.message == "Something is wrong"

    def test_violation_to_dict(self):
        violation = RuleViolation(
            rule_id="test_rule",
            rule_name="Test Rule",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.STATE,
            message="A warning",
            entity_type="hero",
            entity_id="hero_001",
        )

        d = violation.to_dict()

        assert d["rule_id"] == "test_rule"
        assert d["severity"] == "warning"
        assert d["category"] == "state"
        assert d["entity_type"] == "hero"
        assert "detected_at" in d

    def test_violation_from_dict(self):
        data = {
            "rule_id": "test_rule",
            "rule_name": "Test Rule",
            "severity": "error",
            "category": "state",
            "message": "Error message",
            "details": {"field": "health"},
            "entity_type": "hero",
            "entity_id": "hero_001",
            "detected_at": datetime.now().isoformat(),
        }

        violation = RuleViolation.from_dict(data)

        assert violation.rule_id == "test_rule"
        assert violation.severity == RuleSeverity.ERROR
        assert violation.details["field"] == "health"


# =============================================================================
# ConsistencyRule Tests
# =============================================================================

class TestConsistencyRule:
    """Tests for ConsistencyRule."""

    def test_field_constraint_min(self, sample_hero):
        rule = ConsistencyRule(
            id="hero_level_min",
            name="Hero Level Min",
            description="Level must be >= 1",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="level",
            constraint_type="min",
            constraint_value=1,
        )

        # Valid hero should pass
        violation = rule.check(sample_hero)
        assert violation is None

    def test_field_constraint_min_violation(self, invalid_hero):
        rule = ConsistencyRule(
            id="hero_level_min",
            name="Hero Level Min",
            description="Level must be >= 1",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="level",
            constraint_type="min",
            constraint_value=1,
        )

        violation = rule.check(invalid_hero)
        assert violation is not None
        assert "level" in violation.message
        assert "-1" in violation.message

    def test_field_constraint_max(self):
        hero = {"id": "h1", "level": 100}

        rule = ConsistencyRule(
            id="hero_level_max",
            name="Hero Level Max",
            description="Level must be <= 50",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="level",
            constraint_type="max",
            constraint_value=50,
        )

        violation = rule.check(hero)
        assert violation is not None
        assert "100" in violation.message

    def test_field_constraint_range(self, sample_hero):
        rule = ConsistencyRule(
            id="hero_level_range",
            name="Hero Level Range",
            description="Level must be 1-100",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="level",
            constraint_type="range",
            constraint_value=(1, 100),
        )

        # Level 5 is in range
        violation = rule.check(sample_hero)
        assert violation is None

    def test_field_constraint_not_null(self):
        hero = {"id": "h1", "name": None}

        rule = ConsistencyRule(
            id="hero_name_required",
            name="Hero Name Required",
            description="Name must not be null",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="name",
            constraint_type="not_null",
        )

        violation = rule.check(hero)
        assert violation is not None
        assert "null" in violation.message

    def test_field_constraint_in_set(self):
        quest = {"id": "q1", "status": "invalid_status"}

        rule = ConsistencyRule(
            id="quest_status_valid",
            name="Quest Status Valid",
            description="Status must be valid",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="quest",
            field_name="status",
            constraint_type="in_set",
            constraint_value=["pending", "active", "completed", "failed"],
        )

        violation = rule.check(quest)
        assert violation is not None
        assert "invalid_status" in violation.message

    def test_field_constraint_positive(self, invalid_hero):
        rule = ConsistencyRule(
            id="hero_xp_positive",
            name="Hero XP Positive",
            description="XP must be positive",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="xp",
            constraint_type="positive",
        )

        violation = rule.check(invalid_hero)
        assert violation is not None
        assert "-500" in violation.message

    def test_field_constraint_non_negative(self):
        hero = {"id": "h1", "xp": 0}

        rule = ConsistencyRule(
            id="hero_xp_non_negative",
            name="Hero XP Non-Negative",
            description="XP must be >= 0",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="xp",
            constraint_type="non_negative",
        )

        # 0 is non-negative, should pass
        violation = rule.check(hero)
        assert violation is None

    def test_custom_check_function(self, invalid_hero):
        def check_timestamps(entity, context):
            created = entity.get("created_at")
            updated = entity.get("updated_at")
            if created and updated and created > updated:
                return RuleViolation(
                    rule_id="temporal_order",
                    rule_name="Temporal Order",
                    severity=RuleSeverity.ERROR,
                    category=RuleCategory.TEMPORAL,
                    message="created_at must be before updated_at",
                )
            return None

        rule = ConsistencyRule(
            id="temporal_order",
            name="Temporal Order",
            description="created_at must be before updated_at",
            category=RuleCategory.TEMPORAL,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            check_fn=check_timestamps,
        )

        violation = rule.check(invalid_hero)
        assert violation is not None

    def test_disabled_rule(self, invalid_hero):
        rule = ConsistencyRule(
            id="hero_level_min",
            name="Hero Level Min",
            description="Level must be >= 1",
            category=RuleCategory.STATE,
            severity=RuleSeverity.ERROR,
            entity_type="hero",
            field_name="level",
            constraint_type="min",
            constraint_value=1,
            enabled=False,
        )

        # Disabled rule returns None even for invalid data
        violation = rule.check(invalid_hero)
        assert violation is None


# =============================================================================
# RuleBuilder Tests
# =============================================================================

class TestRuleBuilder:
    """Tests for RuleBuilder."""

    def test_build_min_rule(self):
        rule = (RuleBuilder("hero_level_min", "hero")
            .name("Hero Level Min")
            .description("Level must be >= 1")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("level")
            .min_value(1)
            .build())

        assert rule.id == "hero_level_min"
        assert rule.entity_type == "hero"
        assert rule.field_name == "level"
        assert rule.constraint_type == "min"
        assert rule.constraint_value == 1

    def test_build_range_rule(self):
        rule = (RuleBuilder("health_range", "hero")
            .name("Health Range")
            .field("health")
            .in_range(0, 100)
            .build())

        assert rule.constraint_type == "range"
        assert rule.constraint_value == (0, 100)

    def test_build_in_set_rule(self):
        rule = (RuleBuilder("status_valid", "quest")
            .field("status")
            .in_set(["pending", "active", "done"])
            .build())

        assert rule.constraint_type == "in_set"
        assert "pending" in rule.constraint_value

    def test_build_custom_check_rule(self):
        def my_check(entity, ctx):
            return None

        rule = (RuleBuilder("custom", "hero")
            .check(my_check)
            .build())

        assert rule.check_fn == my_check

    def test_build_with_tags(self):
        rule = (RuleBuilder("tagged_rule", "hero")
            .tag("critical", "hero")
            .build())

        assert "critical" in rule.tags
        assert "hero" in rule.tags

    def test_build_disabled_rule(self):
        rule = (RuleBuilder("disabled", "hero")
            .disabled()
            .build())

        assert rule.enabled is False


# =============================================================================
# CheckResult Tests
# =============================================================================

class TestCheckResult:
    """Tests for CheckResult."""

    def test_empty_result_passes(self):
        result = CheckResult()
        assert result.passed is True
        assert result.has_critical is False
        assert result.has_errors is False

    def test_result_with_violations(self):
        result = CheckResult()
        result.add_violation(RuleViolation(
            rule_id="r1",
            rule_name="R1",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.STATE,
            message="Warning",
        ))
        result.add_violation(RuleViolation(
            rule_id="r2",
            rule_name="R2",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Error",
        ))

        assert result.passed is False
        assert result.has_errors is True
        assert result.has_critical is False
        assert len(result.violations) == 2

    def test_result_by_severity(self):
        result = CheckResult()
        result.add_violation(RuleViolation(
            rule_id="r1",
            rule_name="R1",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.STATE,
            message="Warning",
        ))
        result.add_violation(RuleViolation(
            rule_id="r2",
            rule_name="R2",
            severity=RuleSeverity.CRITICAL,
            category=RuleCategory.STATE,
            message="Critical",
        ))

        warnings = result.by_severity(RuleSeverity.WARNING)
        criticals = result.by_severity(RuleSeverity.CRITICAL)

        assert len(warnings) == 1
        assert len(criticals) == 1
        assert result.has_critical is True

    def test_result_by_category(self):
        result = CheckResult()
        result.add_violation(RuleViolation(
            rule_id="r1",
            rule_name="R1",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="State error",
        ))
        result.add_violation(RuleViolation(
            rule_id="r2",
            rule_name="R2",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.TEMPORAL,
            message="Temporal error",
        ))

        state_violations = result.by_category(RuleCategory.STATE)
        assert len(state_violations) == 1

    def test_result_summary(self):
        result = CheckResult()
        result.rules_checked = 10
        result.entities_checked = 5
        result.add_violation(RuleViolation(
            rule_id="r1",
            rule_name="R1",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Error",
        ))
        result.complete()

        summary = result.summary()

        assert summary["passed"] is False
        assert summary["total_violations"] == 1
        assert summary["rules_checked"] == 10
        assert summary["entities_checked"] == 5
        assert summary["by_severity"]["error"] == 1

    def test_result_to_dict(self):
        result = CheckResult()
        result.rules_checked = 5
        result.entities_checked = 2
        result.complete()

        d = result.to_dict()

        assert "violations" in d
        assert "summary" in d
        assert d["rules_checked"] == 5


# =============================================================================
# ConsistencyChecker Tests
# =============================================================================

class TestConsistencyChecker:
    """Tests for ConsistencyChecker."""

    def test_add_and_get_rule(self):
        checker = ConsistencyChecker()
        rule = (RuleBuilder("test_rule", "hero")
            .name("Test Rule")
            .field("level")
            .min_value(1)
            .build())

        checker.add_rule(rule)

        assert checker.get_rule("test_rule") is not None
        assert "test_rule" in checker.list_rules()

    def test_add_multiple_rules(self):
        checker = ConsistencyChecker()
        rules = [
            (RuleBuilder("r1", "hero").field("level").min_value(1).build()),
            (RuleBuilder("r2", "hero").field("xp").non_negative().build()),
            (RuleBuilder("r3", "quest").field("status").not_null().build()),
        ]

        checker.add_rules(rules)

        assert len(checker.list_rules()) == 3

    def test_remove_rule(self):
        checker = ConsistencyChecker()
        rule = (RuleBuilder("removable", "hero").build())
        checker.add_rule(rule)

        assert checker.remove_rule("removable") is True
        assert checker.remove_rule("nonexistent") is False
        assert "removable" not in checker.list_rules()

    def test_enable_disable_rule(self):
        checker = ConsistencyChecker()
        rule = (RuleBuilder("toggleable", "hero")
            .field("level")
            .min_value(1)
            .build())
        checker.add_rule(rule)

        checker.disable_rule("toggleable")
        assert checker.get_rule("toggleable").enabled is False

        checker.enable_rule("toggleable")
        assert checker.get_rule("toggleable").enabled is True

    def test_rules_for_entity(self):
        checker = ConsistencyChecker()
        checker.add_rules([
            (RuleBuilder("hero_r1", "hero").build()),
            (RuleBuilder("hero_r2", "hero").build()),
            (RuleBuilder("quest_r1", "quest").build()),
        ])

        hero_rules = checker.rules_for_entity("hero")
        quest_rules = checker.rules_for_entity("quest")

        assert len(hero_rules) == 2
        assert len(quest_rules) == 1

    def test_check_entity(self, sample_hero):
        checker = ConsistencyChecker()
        checker.add_rules([
            (RuleBuilder("level_min", "hero")
                .field("level").min_value(1).build()),
            (RuleBuilder("xp_non_neg", "hero")
                .field("xp").non_negative().build()),
        ])

        result = checker.check_entity(sample_hero, "hero")

        assert result.passed is True
        assert result.rules_checked == 2
        assert result.entities_checked == 1

    def test_check_entity_with_violations(self, invalid_hero):
        checker = ConsistencyChecker()
        checker.add_rules([
            (RuleBuilder("level_min", "hero")
                .severity(RuleSeverity.ERROR)
                .field("level").min_value(1).build()),
            (RuleBuilder("xp_positive", "hero")
                .severity(RuleSeverity.ERROR)
                .field("xp").positive().build()),
        ])

        result = checker.check_entity(invalid_hero, "hero")

        assert result.passed is False
        assert len(result.violations) == 2
        assert result.has_errors is True

    def test_check_entities_iterator(self):
        checker = ConsistencyChecker()
        checker.add_rule(
            (RuleBuilder("level_min", "hero")
                .field("level").min_value(1).build())
        )

        heroes = [
            {"id": "h1", "level": 5},
            {"id": "h2", "level": 10},
            {"id": "h3", "level": 0},  # Invalid
        ]

        result = checker.check_entities(iter(heroes), "hero")

        assert result.entities_checked == 3
        assert len(result.violations) == 1

    def test_check_all_with_providers(self):
        checker = ConsistencyChecker()

        # Add rules
        checker.add_rule(
            (RuleBuilder("hero_level", "hero")
                .field("level").min_value(1).build())
        )
        checker.add_rule(
            (RuleBuilder("quest_status", "quest")
                .field("status").not_null().build())
        )

        # Add providers
        heroes = [{"id": "h1", "level": 5}, {"id": "h2", "level": 3}]
        quests = [{"id": "q1", "status": "active"}, {"id": "q2", "status": None}]

        checker.add_provider("hero", lambda: iter(heroes))
        checker.add_provider("quest", lambda: iter(quests))

        result = checker.check_all()

        assert result.entities_checked == 4
        assert len(result.violations) == 1  # quest with null status

    def test_remove_provider(self):
        checker = ConsistencyChecker()
        checker.add_provider("hero", lambda: iter([]))

        checker.remove_provider("hero")
        checker.remove_provider("nonexistent")  # Should not raise

        result = checker.check_all()
        assert result.entities_checked == 0


# =============================================================================
# Global Functions Tests
# =============================================================================

class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_init_and_get_checker(self):
        init_consistency_checker()
        checker = get_consistency_checker()
        assert checker is not None

    def test_check_entity_global(self, sample_hero):
        checker = get_consistency_checker()
        checker.add_rule(
            (RuleBuilder("level_min", "hero")
                .field("level").min_value(1).build())
        )

        result = check_entity(sample_hero, "hero")
        assert result.passed is True

    def test_check_all_global(self):
        checker = get_consistency_checker()
        checker.add_provider("hero", lambda: iter([{"id": "h1", "level": 5}]))
        checker.add_rule(
            (RuleBuilder("level_min", "hero")
                .field("level").min_value(1).build())
        )

        result = check_all()
        assert result.entities_checked == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestConsistencyIntegration:
    """Integration tests for consistency checking."""

    def test_hero_validation_suite(self):
        """Test a complete hero validation suite."""
        checker = ConsistencyChecker()

        # Add comprehensive hero rules
        checker.add_rules([
            (RuleBuilder("hero_id_required", "hero")
                .name("Hero ID Required")
                .severity(RuleSeverity.CRITICAL)
                .field("id").not_null().build()),

            (RuleBuilder("hero_level_positive", "hero")
                .name("Hero Level Positive")
                .severity(RuleSeverity.ERROR)
                .field("level").positive().build()),

            (RuleBuilder("hero_xp_non_negative", "hero")
                .name("Hero XP Non-Negative")
                .severity(RuleSeverity.ERROR)
                .field("xp").non_negative().build()),

            (RuleBuilder("hero_health_range", "hero")
                .name("Hero Health Range")
                .severity(RuleSeverity.WARNING)
                .field("health").in_range(0, 100).build()),
        ])

        # Valid hero
        valid_hero = {
            "id": "hero_001",
            "level": 10,
            "xp": 5000,
            "health": 75,
        }

        result = checker.check_entity(valid_hero, "hero")
        assert result.passed is True
        assert result.rules_checked == 4

        # Invalid hero
        invalid_hero = {
            "id": None,  # Critical violation
            "level": 0,   # Error violation
            "xp": -100,   # Error violation
            "health": 150,  # Warning violation
        }

        result = checker.check_entity(invalid_hero, "hero")
        assert result.passed is False
        assert result.has_critical is True
        assert result.has_errors is True
        assert len(result.violations) == 4

    def test_cross_entity_validation(self):
        """Test validation across multiple entity types."""
        checker = ConsistencyChecker()

        # Hero rules
        checker.add_rule(
            (RuleBuilder("hero_level", "hero")
                .field("level").positive().build())
        )

        # Quest rules
        checker.add_rule(
            (RuleBuilder("quest_difficulty", "quest")
                .field("difficulty").in_range(1, 10).build())
        )

        # Run rules
        checker.add_rule(
            (RuleBuilder("run_status", "run")
                .field("status").in_set(["pending", "running", "completed", "failed"]).build())
        )

        # Check each entity type
        hero_result = checker.check_entity({"id": "h1", "level": 5}, "hero")
        quest_result = checker.check_entity({"id": "q1", "difficulty": 3}, "quest")
        run_result = checker.check_entity({"id": "r1", "status": "running"}, "run")

        assert hero_result.passed is True
        assert quest_result.passed is True
        assert run_result.passed is True

    def test_temporal_validation(self):
        """Test temporal consistency rules."""
        def check_temporal_order(entity, ctx):
            created = entity.get("created_at")
            updated = entity.get("updated_at")
            if created and updated and created > updated:
                return RuleViolation(
                    rule_id="temporal_order",
                    rule_name="Temporal Order",
                    severity=RuleSeverity.ERROR,
                    category=RuleCategory.TEMPORAL,
                    message=f"created_at ({created}) must be before updated_at ({updated})",
                    details={"created_at": str(created), "updated_at": str(updated)},
                )
            return None

        checker = ConsistencyChecker()
        checker.add_rule(
            ConsistencyRule(
                id="temporal_order",
                name="Temporal Order",
                description="created_at must be before updated_at",
                category=RuleCategory.TEMPORAL,
                severity=RuleSeverity.ERROR,
                entity_type="hero",
                check_fn=check_temporal_order,
            )
        )

        # Valid temporal order
        valid_hero = {
            "id": "h1",
            "created_at": datetime.now() - timedelta(days=1),
            "updated_at": datetime.now(),
        }

        result = checker.check_entity(valid_hero, "hero")
        assert result.passed is True

        # Invalid temporal order
        invalid_hero = {
            "id": "h2",
            "created_at": datetime.now(),
            "updated_at": datetime.now() - timedelta(days=1),
        }

        result = checker.check_entity(invalid_hero, "hero")
        assert result.passed is False
        assert result.by_category(RuleCategory.TEMPORAL)[0].rule_id == "temporal_order"

    def test_provider_error_handling(self):
        """Test that provider errors are captured as violations."""
        checker = ConsistencyChecker()

        def failing_provider():
            raise ValueError("Provider failed!")

        checker.add_provider("faulty", failing_provider)

        result = checker.check_all()

        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].rule_id == "provider_error"
        assert "Provider failed!" in result.violations[0].message


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization and persistence."""

    def test_check_result_roundtrip(self):
        """Test CheckResult serialization."""
        result = CheckResult()
        result.rules_checked = 10
        result.entities_checked = 5
        result.add_violation(RuleViolation(
            rule_id="test",
            rule_name="Test",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.STATE,
            message="Error",
            entity_type="hero",
            entity_id="h1",
        ))
        result.complete()

        d = result.to_dict()

        assert d["rules_checked"] == 10
        assert d["entities_checked"] == 5
        assert len(d["violations"]) == 1
        assert d["summary"]["total_violations"] == 1

    def test_save_result(self, tmp_path):
        """Test saving check result to file."""
        checker = ConsistencyChecker()
        checker.add_rule(
            (RuleBuilder("level_min", "hero")
                .field("level").min_value(1).build())
        )

        result = checker.check_entity({"id": "h1", "level": -1}, "hero")

        # Save to file
        output_path = tmp_path / "check_result.json"
        checker.save_result(result, output_path)

        assert output_path.exists()

        # Read and verify
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert data["summary"]["total_violations"] == 1
