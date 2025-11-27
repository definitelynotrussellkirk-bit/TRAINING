"""Incident detection logic."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Type

from guild.incidents.types import Incident, IncidentRule
from guild.incidents.tracker import IncidentTracker, get_incident_tracker


logger = logging.getLogger(__name__)


@dataclass
class DetectionContext:
    """Context for incident detection."""

    step: int
    run_id: Optional[str] = None
    quest_id: Optional[str] = None
    facility_id: Optional[str] = None

    # Metrics snapshot
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Log/output text to scan
    text: str = ""

    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Recent history
    recent_losses: List[float] = field(default_factory=list)
    recent_successes: List[bool] = field(default_factory=list)


@dataclass
class DetectionResult:
    """Result of detection check."""

    triggered: bool
    rule_id: str
    template_vars: Dict[str, Any] = field(default_factory=dict)
    context_additions: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """
    Base class for incident detectors.

    Subclass to implement specific detection logic.
    """

    detector_type: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        """
        Check if this rule triggers on the given context.

        Args:
            rule: The incident rule being checked
            context: Current detection context

        Returns:
            DetectionResult indicating if triggered
        """
        pass


class PatternDetector(BaseDetector):
    """
    Detects incidents based on text patterns.

    Config:
        pattern: Regex pattern to match
        field: Field to search (text, error_message, stack_trace)
        flags: Regex flags (i for case-insensitive)
    """

    detector_type = "pattern"

    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        pattern = self.config.get("pattern", "")
        search_field = self.config.get("field", "text")
        flags_str = self.config.get("flags", "")

        # Get text to search
        text = ""
        if search_field == "text":
            text = context.text
        elif search_field == "error_message":
            text = context.error_message or ""
        elif search_field == "stack_trace":
            text = context.stack_trace or ""
        elif search_field == "error_type":
            text = context.error_type or ""

        if not text or not pattern:
            return DetectionResult(triggered=False, rule_id=rule.id)

        # Build flags
        flags = 0
        if "i" in flags_str:
            flags |= re.IGNORECASE
        if "m" in flags_str:
            flags |= re.MULTILINE

        # Search
        match = re.search(pattern, text, flags)

        if match:
            return DetectionResult(
                triggered=True,
                rule_id=rule.id,
                template_vars={
                    "match": match.group(0),
                    "pattern": pattern,
                    **match.groupdict(),
                },
                context_additions={
                    "matched_pattern": pattern,
                    "matched_text": match.group(0),
                },
            )

        return DetectionResult(triggered=False, rule_id=rule.id)


class ThresholdDetector(BaseDetector):
    """
    Detects incidents based on metric thresholds.

    Config:
        metric: Name of metric to check
        operator: Comparison operator (gt, gte, lt, lte, eq)
        value: Threshold value
    """

    detector_type = "threshold"

    OPERATORS = {
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
    }

    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        metric_name = self.config.get("metric", "")
        operator = self.config.get("operator", "gt")
        threshold = self.config.get("value", 0)

        if not metric_name:
            return DetectionResult(triggered=False, rule_id=rule.id)

        # Get metric value
        metric_value = context.metrics.get(metric_name)
        if metric_value is None:
            return DetectionResult(triggered=False, rule_id=rule.id)

        # Compare
        op_func = self.OPERATORS.get(operator)
        if op_func is None:
            logger.warning(f"Unknown operator: {operator}")
            return DetectionResult(triggered=False, rule_id=rule.id)

        triggered = op_func(metric_value, threshold)

        if triggered:
            return DetectionResult(
                triggered=True,
                rule_id=rule.id,
                template_vars={
                    "metric": metric_name,
                    "value": metric_value,
                    "threshold": threshold,
                    "operator": operator,
                },
                context_additions={
                    "metric": metric_name,
                    "metric_value": metric_value,
                    "threshold": threshold,
                },
            )

        return DetectionResult(triggered=False, rule_id=rule.id)


class ConsecutiveDetector(BaseDetector):
    """
    Detects incidents based on consecutive results.

    Config:
        check: What to check (failures, successes)
        count: Number of consecutive occurrences to trigger
    """

    detector_type = "consecutive"

    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        check_type = self.config.get("check", "failures")
        count = self.config.get("count", 3)

        if not context.recent_successes:
            return DetectionResult(triggered=False, rule_id=rule.id)

        recent = context.recent_successes[-count:]
        if len(recent) < count:
            return DetectionResult(triggered=False, rule_id=rule.id)

        if check_type == "failures":
            triggered = all(not s for s in recent)
        elif check_type == "successes":
            triggered = all(s for s in recent)
        else:
            return DetectionResult(triggered=False, rule_id=rule.id)

        if triggered:
            return DetectionResult(
                triggered=True,
                rule_id=rule.id,
                template_vars={
                    "count": count,
                    "check_type": check_type,
                },
                context_additions={
                    "consecutive_count": count,
                    "consecutive_type": check_type,
                },
            )

        return DetectionResult(triggered=False, rule_id=rule.id)


class TrendDetector(BaseDetector):
    """
    Detects incidents based on metric trends.

    Config:
        metric: Name of metric (uses recent_losses if "loss")
        direction: Trend direction (increasing, decreasing)
        window: Number of values to analyze
        min_change: Minimum absolute change to trigger
    """

    detector_type = "trend"

    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        metric_name = self.config.get("metric", "loss")
        direction = self.config.get("direction", "increasing")
        window = self.config.get("window", 5)
        min_change = self.config.get("min_change", 0.0)

        # Get values
        if metric_name == "loss":
            values = context.recent_losses[-window:]
        else:
            return DetectionResult(triggered=False, rule_id=rule.id)

        if len(values) < window:
            return DetectionResult(triggered=False, rule_id=rule.id)

        # Check trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        change = second_avg - first_avg

        triggered = False
        if direction == "increasing" and change >= min_change:
            triggered = True
        elif direction == "decreasing" and change <= -min_change:
            triggered = True

        if triggered:
            return DetectionResult(
                triggered=True,
                rule_id=rule.id,
                template_vars={
                    "metric": metric_name,
                    "direction": direction,
                    "change": change,
                    "first_avg": first_avg,
                    "second_avg": second_avg,
                },
                context_additions={
                    "trend_metric": metric_name,
                    "trend_change": change,
                },
            )

        return DetectionResult(triggered=False, rule_id=rule.id)


class CompositeDetector(BaseDetector):
    """
    Combines multiple detectors with AND/OR logic.

    Config:
        operator: "and" or "or"
        detectors: List of detector configs
    """

    detector_type = "composite"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._sub_detectors: List[BaseDetector] = []

        for det_config in config.get("detectors", []):
            det_type = det_config.get("type", "pattern")
            detector_class = DETECTOR_REGISTRY.get(det_type)
            if detector_class:
                self._sub_detectors.append(
                    detector_class(det_config.get("config", {}))
                )

    def check(self, rule: IncidentRule, context: DetectionContext) -> DetectionResult:
        operator = self.config.get("operator", "and")

        results = []
        for detector in self._sub_detectors:
            result = detector.check(rule, context)
            results.append(result)

        if not results:
            return DetectionResult(triggered=False, rule_id=rule.id)

        if operator == "and":
            triggered = all(r.triggered for r in results)
        else:  # or
            triggered = any(r.triggered for r in results)

        if triggered:
            # Merge template vars and context from triggered sub-detectors
            template_vars = {}
            context_additions = {}
            for r in results:
                if r.triggered:
                    template_vars.update(r.template_vars)
                    context_additions.update(r.context_additions)

            return DetectionResult(
                triggered=True,
                rule_id=rule.id,
                template_vars=template_vars,
                context_additions=context_additions,
            )

        return DetectionResult(triggered=False, rule_id=rule.id)


# Detector registry
DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    "pattern": PatternDetector,
    "threshold": ThresholdDetector,
    "consecutive": ConsecutiveDetector,
    "trend": TrendDetector,
    "composite": CompositeDetector,
}


def register_detector(detector_class: Type[BaseDetector]):
    """Register a custom detector class."""
    DETECTOR_REGISTRY[detector_class.detector_type] = detector_class


def get_detector(detector_type: str, config: Dict[str, Any]) -> Optional[BaseDetector]:
    """Get a detector instance by type."""
    detector_class = DETECTOR_REGISTRY.get(detector_type)
    if detector_class is None:
        logger.warning(f"Unknown detector type: {detector_type}")
        return None
    return detector_class(config)


class IncidentDetector:
    """
    Main detector that checks all rules against context.

    Example:
        detector = IncidentDetector(tracker)
        detector.add_rules([rule1, rule2])

        # Check on each step
        incidents = detector.check(context)
    """

    def __init__(
        self,
        tracker: Optional[IncidentTracker] = None,
        rules: Optional[List[IncidentRule]] = None,
    ):
        self.tracker = tracker or get_incident_tracker()
        self._rules: List[IncidentRule] = []
        self._detectors: Dict[str, BaseDetector] = {}
        self._cooldowns: Dict[str, datetime] = {}
        self._cooldown_seconds = 300  # 5 minute cooldown per rule

        if rules:
            self.add_rules(rules)

    def add_rule(self, rule: IncidentRule):
        """Add a rule to check."""
        self._rules.append(rule)

        # Create detector
        detector = get_detector(rule.detector_type, rule.detector_config)
        if detector:
            self._detectors[rule.id] = detector

    def add_rules(self, rules: List[IncidentRule]):
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)

    def clear_rules(self):
        """Clear all rules."""
        self._rules.clear()
        self._detectors.clear()
        self._cooldowns.clear()

    def set_cooldown(self, seconds: int):
        """Set cooldown period between incidents from same rule."""
        self._cooldown_seconds = seconds

    def _is_on_cooldown(self, rule_id: str) -> bool:
        """Check if a rule is on cooldown."""
        last_triggered = self._cooldowns.get(rule_id)
        if last_triggered is None:
            return False

        elapsed = (datetime.now() - last_triggered).total_seconds()
        return elapsed < self._cooldown_seconds

    def check(self, context: DetectionContext) -> List[Incident]:
        """
        Check all rules against the context.

        Args:
            context: Detection context

        Returns:
            List of created incidents
        """
        incidents = []

        for rule in self._rules:
            # Skip if on cooldown
            if self._is_on_cooldown(rule.id):
                continue

            detector = self._detectors.get(rule.id)
            if detector is None:
                continue

            try:
                result = detector.check(rule, context)

                if result.triggered:
                    # Create incident
                    incident = self.tracker.create_from_rule(
                        rule=rule,
                        detected_at_step=context.step,
                        template_vars=result.template_vars,
                        run_id=context.run_id,
                        quest_id=context.quest_id,
                        facility_id=context.facility_id,
                        context=result.context_additions,
                    )
                    incidents.append(incident)

                    # Set cooldown
                    self._cooldowns[rule.id] = datetime.now()

            except Exception as e:
                logger.error(f"Error checking rule {rule.id}: {e}")

        return incidents

    def check_text(
        self,
        text: str,
        step: int,
        run_id: Optional[str] = None,
    ) -> List[Incident]:
        """Convenience method to check text for patterns."""
        context = DetectionContext(
            step=step,
            run_id=run_id,
            text=text,
        )
        return self.check(context)

    def check_error(
        self,
        error_type: str,
        error_message: str,
        step: int,
        run_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> List[Incident]:
        """Convenience method to check an error."""
        context = DetectionContext(
            step=step,
            run_id=run_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            text=f"{error_type}: {error_message}\n{stack_trace or ''}",
        )
        return self.check(context)

    def check_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        run_id: Optional[str] = None,
    ) -> List[Incident]:
        """Convenience method to check metrics."""
        context = DetectionContext(
            step=step,
            run_id=run_id,
            metrics=metrics,
        )
        return self.check(context)


# Global detector
_detector: Optional[IncidentDetector] = None


def init_incident_detector(
    tracker: Optional[IncidentTracker] = None,
    rules: Optional[List[IncidentRule]] = None,
) -> IncidentDetector:
    """Initialize the global incident detector."""
    global _detector
    _detector = IncidentDetector(tracker, rules)
    return _detector


def get_incident_detector() -> IncidentDetector:
    """Get the global incident detector."""
    global _detector
    if _detector is None:
        _detector = IncidentDetector()
    return _detector


def reset_incident_detector():
    """Reset the global incident detector (for testing)."""
    global _detector
    _detector = None
