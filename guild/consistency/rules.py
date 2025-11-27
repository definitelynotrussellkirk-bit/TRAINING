"""Consistency rule definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable, Dict, List


class RuleCategory(Enum):
    """Categories of consistency rules."""
    STATE = "state"           # State validity (health > 0, level > 0)
    TRANSITION = "transition" # State transition rules
    REFERENCE = "reference"   # Cross-reference integrity
    INVARIANT = "invariant"   # System-wide invariants
    TEMPORAL = "temporal"     # Time-based rules (created < updated)


class RuleSeverity(Enum):
    """Severity of rule violations."""
    WARNING = "warning"   # Log warning, continue
    ERROR = "error"       # Log error, may continue
    CRITICAL = "critical" # Must stop, data integrity at risk


@dataclass
class RuleViolation:
    """A detected rule violation."""

    rule_id: str
    rule_name: str
    severity: RuleSeverity
    category: RuleCategory

    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    entity_type: str = ""
    entity_id: str = ""

    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleViolation":
        return cls(
            rule_id=data["rule_id"],
            rule_name=data["rule_name"],
            severity=RuleSeverity(data["severity"]),
            category=RuleCategory(data["category"]),
            message=data["message"],
            details=data.get("details", {}),
            entity_type=data.get("entity_type", ""),
            entity_id=data.get("entity_id", ""),
            detected_at=datetime.fromisoformat(data["detected_at"]),
        )


@dataclass
class ConsistencyRule:
    """
    Definition of a consistency rule.

    Rules define checks that can be run against entities
    to detect violations.
    """

    id: str
    name: str
    description: str
    category: RuleCategory
    severity: RuleSeverity

    # Entity type this rule applies to (e.g., "hero", "run", "quest")
    entity_type: str

    # Check function: (entity, context) -> Optional[RuleViolation]
    # If returns None, rule passes; otherwise returns violation
    check_fn: Optional[Callable[[Any, Dict], Optional[RuleViolation]]] = None

    # For simple rules, can specify field constraints
    field_name: Optional[str] = None
    constraint_type: Optional[str] = None  # "min", "max", "range", "not_null", "in_set"
    constraint_value: Any = None

    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    def check(
        self,
        entity: Any,
        context: Optional[Dict] = None,
    ) -> Optional[RuleViolation]:
        """
        Check an entity against this rule.

        Args:
            entity: Entity to check
            context: Optional context for checking

        Returns:
            RuleViolation if rule violated, None otherwise
        """
        if not self.enabled:
            return None

        ctx = context or {}

        # Use custom check function if provided
        if self.check_fn:
            return self.check_fn(entity, ctx)

        # Use field constraint check
        if self.field_name and self.constraint_type:
            return self._check_field_constraint(entity)

        return None

    def _check_field_constraint(self, entity: Any) -> Optional[RuleViolation]:
        """Check a simple field constraint."""
        # Get field value
        if hasattr(entity, self.field_name):
            value = getattr(entity, self.field_name)
        elif isinstance(entity, dict):
            value = entity.get(self.field_name)
        else:
            return RuleViolation(
                rule_id=self.id,
                rule_name=self.name,
                severity=self.severity,
                category=self.category,
                message=f"Cannot access field '{self.field_name}'",
                entity_type=self.entity_type,
            )

        # Check constraint
        violation_msg = None

        if self.constraint_type == "not_null":
            if value is None:
                violation_msg = f"{self.field_name} must not be null"

        elif self.constraint_type == "min":
            if value is not None and value < self.constraint_value:
                violation_msg = f"{self.field_name} ({value}) must be >= {self.constraint_value}"

        elif self.constraint_type == "max":
            if value is not None and value > self.constraint_value:
                violation_msg = f"{self.field_name} ({value}) must be <= {self.constraint_value}"

        elif self.constraint_type == "range":
            min_val, max_val = self.constraint_value
            if value is not None and (value < min_val or value > max_val):
                violation_msg = f"{self.field_name} ({value}) must be in range [{min_val}, {max_val}]"

        elif self.constraint_type == "in_set":
            if value not in self.constraint_value:
                violation_msg = f"{self.field_name} ({value}) must be one of {self.constraint_value}"

        elif self.constraint_type == "positive":
            if value is not None and value <= 0:
                violation_msg = f"{self.field_name} ({value}) must be positive"

        elif self.constraint_type == "non_negative":
            if value is not None and value < 0:
                violation_msg = f"{self.field_name} ({value}) must be non-negative"

        if violation_msg:
            return RuleViolation(
                rule_id=self.id,
                rule_name=self.name,
                severity=self.severity,
                category=self.category,
                message=violation_msg,
                details={"field": self.field_name, "value": value},
                entity_type=self.entity_type,
            )

        return None


class RuleBuilder:
    """
    Builder for creating consistency rules.

    Example:
        rule = (RuleBuilder("hero_health_positive", "hero")
            .name("Hero Health Positive")
            .description("Hero health must be positive")
            .category(RuleCategory.STATE)
            .severity(RuleSeverity.ERROR)
            .field("health")
            .min_value(0)
            .build())
    """

    def __init__(self, rule_id: str, entity_type: str):
        self._id = rule_id
        self._entity_type = entity_type
        self._name = rule_id
        self._description = ""
        self._category = RuleCategory.STATE
        self._severity = RuleSeverity.ERROR
        self._field_name: Optional[str] = None
        self._constraint_type: Optional[str] = None
        self._constraint_value: Any = None
        self._check_fn: Optional[Callable] = None
        self._enabled = True
        self._tags: List[str] = []

    def name(self, name: str) -> "RuleBuilder":
        self._name = name
        return self

    def description(self, desc: str) -> "RuleBuilder":
        self._description = desc
        return self

    def category(self, cat: RuleCategory) -> "RuleBuilder":
        self._category = cat
        return self

    def severity(self, sev: RuleSeverity) -> "RuleBuilder":
        self._severity = sev
        return self

    def field(self, field_name: str) -> "RuleBuilder":
        self._field_name = field_name
        return self

    def min_value(self, value: Any) -> "RuleBuilder":
        self._constraint_type = "min"
        self._constraint_value = value
        return self

    def max_value(self, value: Any) -> "RuleBuilder":
        self._constraint_type = "max"
        self._constraint_value = value
        return self

    def in_range(self, min_val: Any, max_val: Any) -> "RuleBuilder":
        self._constraint_type = "range"
        self._constraint_value = (min_val, max_val)
        return self

    def not_null(self) -> "RuleBuilder":
        self._constraint_type = "not_null"
        return self

    def in_set(self, values: List[Any]) -> "RuleBuilder":
        self._constraint_type = "in_set"
        self._constraint_value = values
        return self

    def positive(self) -> "RuleBuilder":
        self._constraint_type = "positive"
        return self

    def non_negative(self) -> "RuleBuilder":
        self._constraint_type = "non_negative"
        return self

    def check(self, fn: Callable[[Any, Dict], Optional[RuleViolation]]) -> "RuleBuilder":
        self._check_fn = fn
        return self

    def tag(self, *tags: str) -> "RuleBuilder":
        self._tags.extend(tags)
        return self

    def disabled(self) -> "RuleBuilder":
        self._enabled = False
        return self

    def build(self) -> ConsistencyRule:
        return ConsistencyRule(
            id=self._id,
            name=self._name,
            description=self._description,
            category=self._category,
            severity=self._severity,
            entity_type=self._entity_type,
            field_name=self._field_name,
            constraint_type=self._constraint_type,
            constraint_value=self._constraint_value,
            check_fn=self._check_fn,
            enabled=self._enabled,
            tags=self._tags,
        )
