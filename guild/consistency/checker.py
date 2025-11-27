"""Consistency checking engine."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator, Callable

from guild.consistency.rules import (
    ConsistencyRule,
    RuleViolation,
    RuleCategory,
    RuleSeverity,
)


logger = logging.getLogger(__name__)


class CheckResult:
    """Result of a consistency check."""

    def __init__(self):
        self.violations: List[RuleViolation] = []
        self.rules_checked: int = 0
        self.entities_checked: int = 0
        self.started_at: datetime = datetime.now()
        self.completed_at: Optional[datetime] = None

    @property
    def passed(self) -> bool:
        """Check passed if no violations."""
        return len(self.violations) == 0

    @property
    def has_critical(self) -> bool:
        """Has critical violations."""
        return any(v.severity == RuleSeverity.CRITICAL for v in self.violations)

    @property
    def has_errors(self) -> bool:
        """Has error violations."""
        return any(v.severity == RuleSeverity.ERROR for v in self.violations)

    @property
    def duration_seconds(self) -> float:
        """Duration of check."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def add_violation(self, violation: RuleViolation):
        """Add a violation."""
        self.violations.append(violation)

    def complete(self):
        """Mark check as complete."""
        self.completed_at = datetime.now()

    def by_severity(self, severity: RuleSeverity) -> List[RuleViolation]:
        """Get violations by severity."""
        return [v for v in self.violations if v.severity == severity]

    def by_category(self, category: RuleCategory) -> List[RuleViolation]:
        """Get violations by category."""
        return [v for v in self.violations if v.category == category]

    def by_entity(self, entity_type: str) -> List[RuleViolation]:
        """Get violations by entity type."""
        return [v for v in self.violations if v.entity_type == entity_type]

    def summary(self) -> Dict[str, Any]:
        """Get check summary."""
        by_severity = {}
        for sev in RuleSeverity:
            count = len(self.by_severity(sev))
            if count > 0:
                by_severity[sev.value] = count

        by_category = {}
        for cat in RuleCategory:
            count = len(self.by_category(cat))
            if count > 0:
                by_category[cat.value] = count

        return {
            "passed": self.passed,
            "total_violations": len(self.violations),
            "rules_checked": self.rules_checked,
            "entities_checked": self.entities_checked,
            "duration_seconds": self.duration_seconds,
            "by_severity": by_severity,
            "by_category": by_category,
            "has_critical": self.has_critical,
            "has_errors": self.has_errors,
        }

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "violations": [v.to_dict() for v in self.violations],
            "rules_checked": self.rules_checked,
            "entities_checked": self.entities_checked,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": self.summary(),
        }


class ConsistencyChecker:
    """
    Main consistency checking engine.

    Manages rules and checks entities against them.

    Example:
        checker = ConsistencyChecker()
        checker.add_rule(rule1)
        checker.add_rule(rule2)

        # Check a single entity
        result = checker.check_entity(hero, "hero")

        # Check all entities from providers
        checker.add_provider("hero", lambda: [hero1, hero2])
        result = checker.check_all()
    """

    def __init__(self):
        self._rules: Dict[str, ConsistencyRule] = {}
        self._providers: Dict[str, Callable[[], Iterator[Any]]] = {}
        self._last_result: Optional[CheckResult] = None

    # --- Rule Management ---

    def add_rule(self, rule: ConsistencyRule):
        """Add a rule."""
        self._rules[rule.id] = rule
        logger.debug(f"Added rule: {rule.id}")

    def add_rules(self, rules: List[ConsistencyRule]):
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[ConsistencyRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self) -> List[str]:
        """List all rule IDs."""
        return list(self._rules.keys())

    def rules_for_entity(self, entity_type: str) -> List[ConsistencyRule]:
        """Get rules that apply to an entity type."""
        return [
            r for r in self._rules.values()
            if r.entity_type == entity_type and r.enabled
        ]

    def enable_rule(self, rule_id: str):
        """Enable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """Disable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False

    # --- Entity Providers ---

    def add_provider(
        self,
        entity_type: str,
        provider: Callable[[], Iterator[Any]],
    ):
        """
        Add an entity provider.

        Providers supply entities to check during check_all().
        """
        self._providers[entity_type] = provider

    def remove_provider(self, entity_type: str):
        """Remove an entity provider."""
        self._providers.pop(entity_type, None)

    # --- Checking ---

    def check_entity(
        self,
        entity: Any,
        entity_type: str,
        entity_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> CheckResult:
        """
        Check a single entity against applicable rules.

        Args:
            entity: Entity to check
            entity_type: Type of entity
            entity_id: Optional ID for reporting
            context: Optional context

        Returns:
            CheckResult with any violations
        """
        result = CheckResult()
        rules = self.rules_for_entity(entity_type)
        ctx = context or {}

        # Get entity ID
        eid = entity_id
        if eid is None and hasattr(entity, 'id'):
            eid = getattr(entity, 'id')
        elif eid is None and isinstance(entity, dict):
            eid = entity.get('id', 'unknown')

        for rule in rules:
            result.rules_checked += 1

            violation = rule.check(entity, ctx)
            if violation:
                violation.entity_type = entity_type
                violation.entity_id = str(eid) if eid else ""
                result.add_violation(violation)
                logger.warning(
                    f"[{rule.severity.value}] {violation.message} "
                    f"(entity: {entity_type}/{eid})"
                )

        result.entities_checked = 1
        result.complete()

        return result

    def check_entities(
        self,
        entities: Iterator[Any],
        entity_type: str,
        context: Optional[Dict] = None,
    ) -> CheckResult:
        """
        Check multiple entities of the same type.

        Args:
            entities: Iterator of entities
            entity_type: Type of entities
            context: Optional context

        Returns:
            Aggregated CheckResult
        """
        result = CheckResult()
        rules = self.rules_for_entity(entity_type)
        ctx = context or {}

        for entity in entities:
            result.entities_checked += 1

            # Get entity ID
            eid = None
            if hasattr(entity, 'id'):
                eid = getattr(entity, 'id')
            elif isinstance(entity, dict):
                eid = entity.get('id')

            for rule in rules:
                result.rules_checked += 1

                violation = rule.check(entity, ctx)
                if violation:
                    violation.entity_type = entity_type
                    violation.entity_id = str(eid) if eid else ""
                    result.add_violation(violation)

        result.complete()
        return result

    def check_all(
        self,
        context: Optional[Dict] = None,
    ) -> CheckResult:
        """
        Check all entities from all providers.

        Returns:
            Aggregated CheckResult
        """
        result = CheckResult()
        ctx = context or {}

        for entity_type, provider in self._providers.items():
            try:
                entities = provider()
                partial_result = self.check_entities(entities, entity_type, ctx)

                result.rules_checked += partial_result.rules_checked
                result.entities_checked += partial_result.entities_checked
                result.violations.extend(partial_result.violations)

            except Exception as e:
                logger.error(f"Error checking {entity_type}: {e}")
                # Create violation for provider error
                result.add_violation(RuleViolation(
                    rule_id="provider_error",
                    rule_name="Provider Error",
                    severity=RuleSeverity.ERROR,
                    category=RuleCategory.INVARIANT,
                    message=f"Error from {entity_type} provider: {str(e)}",
                    entity_type=entity_type,
                ))

        result.complete()
        self._last_result = result

        # Log summary
        summary = result.summary()
        if result.passed:
            logger.info(
                f"Consistency check passed: {summary['entities_checked']} entities, "
                f"{summary['rules_checked']} rules"
            )
        else:
            logger.warning(
                f"Consistency check found {summary['total_violations']} violations: "
                f"critical={summary['by_severity'].get('critical', 0)}, "
                f"error={summary['by_severity'].get('error', 0)}, "
                f"warning={summary['by_severity'].get('warning', 0)}"
            )

        return result

    def get_last_result(self) -> Optional[CheckResult]:
        """Get the last check result."""
        return self._last_result

    # --- Persistence ---

    def save_result(
        self,
        result: CheckResult,
        path: Path,
    ):
        """Save check result to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.debug(f"Saved consistency check result to {path}")


# Global checker
_checker: Optional[ConsistencyChecker] = None


def init_consistency_checker() -> ConsistencyChecker:
    """Initialize the global consistency checker."""
    global _checker
    _checker = ConsistencyChecker()
    return _checker


def get_consistency_checker() -> ConsistencyChecker:
    """Get the global consistency checker."""
    global _checker
    if _checker is None:
        _checker = ConsistencyChecker()
    return _checker


def reset_consistency_checker():
    """Reset the global consistency checker (for testing)."""
    global _checker
    _checker = None


def check_entity(
    entity: Any,
    entity_type: str,
    entity_id: Optional[str] = None,
) -> CheckResult:
    """Check an entity using the global checker."""
    return get_consistency_checker().check_entity(entity, entity_type, entity_id)


def check_all() -> CheckResult:
    """Run full consistency check using the global checker."""
    return get_consistency_checker().check_all()
