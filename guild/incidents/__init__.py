"""
Incidents system - error tracking and management.

The incidents module provides:
- Incident/IncidentRule: Data types for incidents and detection rules
- IncidentRuleRegistry: Central access to rule configurations
- IncidentTracker: Track, create, and manage incidents
- IncidentDetector: Automatic detection from patterns, thresholds, etc.

Example:
    from guild.incidents import (
        init_incident_rule_registry,
        init_incident_tracker,
        init_incident_detector,
        create_incident,
        IncidentCategory,
        Severity,
    )

    # Initialize
    init_incident_rule_registry("/path/to/configs")
    init_incident_tracker("/path/to/status")

    # Create manually
    incident = create_incident(
        category=IncidentCategory.TRAINING,
        severity=Severity.HIGH,
        title="OOM Crash",
        description="Out of memory during training",
        detected_at_step=1000,
    )

    # Or use automatic detection
    detector = init_incident_detector()
    detector.add_rules(rules)
    incidents = detector.check(context)
"""

# Types
from guild.incidents.types import (
    IncidentCategory,
    IncidentStatus,
    Incident,
    IncidentRule,
)

# Re-export Severity from guild.types for convenience
from guild.types import Severity

# Loader
from guild.incidents.loader import (
    load_incident_rule,
    discover_incident_rules,
    load_all_incident_rules,
    IncidentRuleLoader,
)

# Registry
from guild.incidents.registry import (
    IncidentRuleRegistry,
    init_incident_rule_registry,
    get_incident_rule_registry,
    reset_incident_rule_registry,
    get_incident_rule,
    list_incident_rules,
    incident_rules_by_category,
    incident_rules_by_severity,
)

# Tracker
from guild.incidents.tracker import (
    IncidentTracker,
    init_incident_tracker,
    get_incident_tracker,
    reset_incident_tracker,
    create_incident,
    get_incident,
    resolve_incident,
    list_open_incidents,
    get_incident_stats,
)

# Detector
from guild.incidents.detector import (
    DetectionContext,
    DetectionResult,
    BaseDetector,
    PatternDetector,
    ThresholdDetector,
    ConsecutiveDetector,
    TrendDetector,
    CompositeDetector,
    DETECTOR_REGISTRY,
    register_detector,
    get_detector,
    IncidentDetector,
    init_incident_detector,
    get_incident_detector,
    reset_incident_detector,
)

__all__ = [
    # Types
    "IncidentCategory",
    "IncidentStatus",
    "Incident",
    "IncidentRule",
    "Severity",
    # Loader
    "load_incident_rule",
    "discover_incident_rules",
    "load_all_incident_rules",
    "IncidentRuleLoader",
    # Registry
    "IncidentRuleRegistry",
    "init_incident_rule_registry",
    "get_incident_rule_registry",
    "reset_incident_rule_registry",
    "get_incident_rule",
    "list_incident_rules",
    "incident_rules_by_category",
    "incident_rules_by_severity",
    # Tracker
    "IncidentTracker",
    "init_incident_tracker",
    "get_incident_tracker",
    "reset_incident_tracker",
    "create_incident",
    "get_incident",
    "resolve_incident",
    "list_open_incidents",
    "get_incident_stats",
    # Detector
    "DetectionContext",
    "DetectionResult",
    "BaseDetector",
    "PatternDetector",
    "ThresholdDetector",
    "ConsecutiveDetector",
    "TrendDetector",
    "CompositeDetector",
    "DETECTOR_REGISTRY",
    "register_detector",
    "get_detector",
    "IncidentDetector",
    "init_incident_detector",
    "get_incident_detector",
    "reset_incident_detector",
]
