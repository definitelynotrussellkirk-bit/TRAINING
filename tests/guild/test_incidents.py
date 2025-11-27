"""Tests for incident loading, registry, tracking, and detection."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
from datetime import datetime, timedelta

from guild.types import Severity
from guild.incidents.types import (
    IncidentCategory,
    IncidentStatus,
    Incident,
    IncidentRule,
)
from guild.incidents.loader import (
    load_incident_rule,
    discover_incident_rules,
    load_all_incident_rules,
    IncidentRuleLoader,
    _dict_to_incident_rule,
)
from guild.incidents.registry import (
    IncidentRuleRegistry,
    init_incident_rule_registry,
    get_incident_rule_registry,
    reset_incident_rule_registry,
    get_incident_rule,
    list_incident_rules,
)
from guild.incidents.tracker import (
    IncidentTracker,
    init_incident_tracker,
    get_incident_tracker,
    reset_incident_tracker,
    create_incident as create_incident_func,
)
from guild.incidents.detector import (
    DetectionContext,
    DetectionResult,
    PatternDetector,
    ThresholdDetector,
    ConsecutiveDetector,
    TrendDetector,
    CompositeDetector,
    IncidentDetector,
    init_incident_detector,
    get_incident_detector,
    reset_incident_detector,
    get_detector,
    DETECTOR_REGISTRY,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test incident rules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        incidents_dir = config_dir / "incidents"
        incidents_dir.mkdir(parents=True)

        # Create test incident rules
        oom_rule = """
id: test_oom
name: Test OOM Rule
category: infra
severity: critical

detector_type: pattern
detector_config:
  pattern: "CUDA out of memory|OOM"
  field: text
  flags: "i"

title_template: "OOM: {match}"
description_template: "Out of memory error detected"
"""
        (incidents_dir / "test_oom.yaml").write_text(oom_rule)

        loss_rule = """
id: test_loss_spike
name: Test Loss Spike
category: training
severity: high

detector_type: threshold
detector_config:
  metric: loss
  operator: gt
  value: 5.0

title_template: "Loss spike: {value}"
description_template: "Loss exceeded threshold"
"""
        (incidents_dir / "test_loss_spike.yaml").write_text(loss_rule)

        failures_rule = """
id: test_failures
name: Test Consecutive Failures
category: training
severity: medium

detector_type: consecutive
detector_config:
  check: failures
  count: 3

title_template: "{count} consecutive failures"
description_template: "Multiple failures in a row"
"""
        (incidents_dir / "test_failures.yaml").write_text(failures_rule)

        yield config_dir


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_incident_rule_registry()
    reset_incident_tracker()
    reset_incident_detector()
    yield
    reset_incident_rule_registry()
    reset_incident_tracker()
    reset_incident_detector()


@pytest.fixture
def sample_rule():
    """Create a sample IncidentRule."""
    return IncidentRule(
        id="sample_rule",
        name="Sample Rule",
        category=IncidentCategory.TRAINING,
        severity=Severity.HIGH,
        detector_type="pattern",
        detector_config={"pattern": "error", "field": "text"},
        title_template="Error: {match}",
        description_template="An error occurred",
    )


# =============================================================================
# Type Tests
# =============================================================================

class TestIncidentTypes:
    """Tests for incident type definitions."""

    def test_incident_category_enum(self):
        assert IncidentCategory.DATA.value == "data"
        assert IncidentCategory.TRAINING.value == "training"
        assert IncidentCategory.INFRA.value == "infra"
        assert IncidentCategory.LOGIC.value == "logic"

    def test_incident_status_enum(self):
        assert IncidentStatus.OPEN.value == "open"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.WONTFIX.value == "wontfix"

    def test_incident_creation(self):
        incident = Incident(
            id="inc_001",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test Incident",
            description="A test incident",
            detected_at_step=100,
        )

        assert incident.id == "inc_001"
        assert incident.category == IncidentCategory.TRAINING
        assert incident.severity == Severity.HIGH
        assert incident.status == IncidentStatus.OPEN
        assert incident.detected_at_step == 100

    def test_incident_serialization(self):
        incident = Incident(
            id="inc_001",
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="OOM Crash",
            description="Out of memory",
            detected_at_step=500,
            run_id="run_001",
            context={"gpu_memory": "24GB"},
        )

        data = incident.to_dict()
        assert data["id"] == "inc_001"
        assert data["category"] == "infra"
        assert data["severity"] == "critical"
        assert data["run_id"] == "run_001"

        restored = Incident.from_dict(data)
        assert restored.id == "inc_001"
        assert restored.category == IncidentCategory.INFRA
        assert restored.severity == Severity.CRITICAL

    def test_incident_rule_creation(self):
        rule = IncidentRule(
            id="rule_001",
            name="Test Rule",
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            detector_type="pattern",
            detector_config={"pattern": "error"},
        )

        assert rule.id == "rule_001"
        assert rule.detector_type == "pattern"


# =============================================================================
# Loader Tests
# =============================================================================

class TestIncidentRuleLoader:
    """Tests for incident rule loading from YAML."""

    def test_load_incident_rule(self, temp_config_dir):
        rule = load_incident_rule("test_oom", temp_config_dir)

        assert rule.id == "test_oom"
        assert rule.name == "Test OOM Rule"
        assert rule.category == IncidentCategory.INFRA
        assert rule.severity == Severity.CRITICAL
        assert rule.detector_type == "pattern"

    def test_load_rule_not_found(self, temp_config_dir):
        with pytest.raises(FileNotFoundError):
            load_incident_rule("nonexistent", temp_config_dir)

    def test_discover_incident_rules(self, temp_config_dir):
        rule_ids = discover_incident_rules(temp_config_dir)

        assert "test_oom" in rule_ids
        assert "test_loss_spike" in rule_ids
        assert "test_failures" in rule_ids
        assert len(rule_ids) == 3

    def test_load_all_incident_rules(self, temp_config_dir):
        rules = load_all_incident_rules(temp_config_dir)

        assert len(rules) == 3
        assert "test_oom" in rules
        assert rules["test_oom"].severity == Severity.CRITICAL

    def test_rule_loader_caching(self, temp_config_dir):
        loader = IncidentRuleLoader(temp_config_dir)

        rule1 = loader.load("test_oom")
        rule2 = loader.load("test_oom")
        assert rule1 is rule2

        loader.clear_cache()
        rule3 = loader.load("test_oom")
        assert rule1 is not rule3

    def test_dict_to_incident_rule_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            _dict_to_incident_rule({"name": "Test"})

    def test_dict_to_incident_rule_invalid_category(self):
        with pytest.raises(ValueError, match="Invalid incident category"):
            _dict_to_incident_rule({
                "id": "test",
                "category": "invalid",
            })


# =============================================================================
# Registry Tests
# =============================================================================

class TestIncidentRuleRegistry:
    """Tests for incident rule registry."""

    def test_registry_get(self, temp_config_dir):
        registry = IncidentRuleRegistry(temp_config_dir)

        rule = registry.get("test_oom")
        assert rule.id == "test_oom"

    def test_registry_get_unknown(self, temp_config_dir):
        registry = IncidentRuleRegistry(temp_config_dir)

        with pytest.raises(KeyError, match="Unknown incident rule"):
            registry.get("nonexistent")

    def test_registry_by_category(self, temp_config_dir):
        registry = IncidentRuleRegistry(temp_config_dir)

        infra_rules = registry.by_category(IncidentCategory.INFRA)
        assert len(infra_rules) == 1
        assert infra_rules[0].id == "test_oom"

        training_rules = registry.by_category(IncidentCategory.TRAINING)
        assert len(training_rules) == 2

    def test_registry_by_severity(self, temp_config_dir):
        registry = IncidentRuleRegistry(temp_config_dir)

        critical = registry.by_severity(Severity.CRITICAL)
        assert len(critical) == 1

        high = registry.by_severity(Severity.HIGH)
        assert len(high) == 1

    def test_registry_search(self, temp_config_dir):
        registry = IncidentRuleRegistry(temp_config_dir)

        results = registry.search(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
        )
        assert len(results) == 1
        assert results[0].id == "test_loss_spike"


class TestGlobalRuleRegistry:
    """Tests for global registry functions."""

    def test_init_and_get_registry(self, temp_config_dir):
        init_incident_rule_registry(temp_config_dir)

        registry = get_incident_rule_registry()
        assert registry is not None
        assert "test_oom" in registry

    def test_get_incident_rule_convenience(self, temp_config_dir):
        init_incident_rule_registry(temp_config_dir)

        rule = get_incident_rule("test_oom")
        assert rule.id == "test_oom"


# =============================================================================
# Tracker Tests
# =============================================================================

class TestIncidentTracker:
    """Tests for incident tracking."""

    def test_create_incident(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        incident = tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test Incident",
            description="A test incident",
            detected_at_step=100,
        )

        assert incident.id.startswith("inc_")
        assert incident.status == IncidentStatus.OPEN

    def test_create_incident_with_explicit_id(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        incident = tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="OOM",
            description="Out of memory",
            detected_at_step=500,
            incident_id="custom_id",
        )

        assert incident.id == "custom_id"

    def test_create_incident_duplicate_id(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.LOW,
            title="Test",
            description="Test",
            detected_at_step=100,
            incident_id="dup_id",
        )

        with pytest.raises(ValueError, match="already exists"):
            tracker.create_incident(
                category=IncidentCategory.TRAINING,
                severity=Severity.LOW,
                title="Test 2",
                description="Test 2",
                detected_at_step=200,
                incident_id="dup_id",
            )

    def test_create_from_rule(self, temp_state_dir, sample_rule):
        tracker = IncidentTracker(temp_state_dir)

        incident = tracker.create_from_rule(
            rule=sample_rule,
            detected_at_step=100,
            template_vars={"match": "test error"},
            run_id="run_001",
        )

        assert incident.category == sample_rule.category
        assert incident.severity == sample_rule.severity
        assert incident.run_id == "run_001"
        assert "rule_id" in incident.context

    def test_get_incident(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        created = tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            title="Test",
            description="Test",
            detected_at_step=100,
            incident_id="get_test",
        )

        retrieved = tracker.get_incident("get_test")
        assert retrieved is not None
        assert retrieved.id == "get_test"

    def test_start_investigation(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test",
            description="Test",
            detected_at_step=100,
            incident_id="investigate_test",
        )

        incident = tracker.start_investigation("investigate_test")
        assert incident.status == IncidentStatus.INVESTIGATING

    def test_resolve(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test",
            description="Test",
            detected_at_step=100,
            incident_id="resolve_test",
        )

        incident = tracker.resolve("resolve_test", "Fixed by reducing batch size")

        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolution == "Fixed by reducing batch size"
        assert incident.resolved_at is not None

    def test_wontfix(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.LOGIC,
            severity=Severity.LOW,
            title="Minor Issue",
            description="Not worth fixing",
            detected_at_step=100,
            incident_id="wontfix_test",
        )

        incident = tracker.wontfix("wontfix_test", "Not reproducible")
        assert incident.status == IncidentStatus.WONTFIX

    def test_reopen(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test",
            description="Test",
            detected_at_step=100,
            incident_id="reopen_test",
        )
        tracker.resolve("reopen_test", "Fixed")

        incident = tracker.reopen("reopen_test")
        assert incident.status == IncidentStatus.OPEN
        assert incident.resolution is None

    def test_list_open(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Open 1",
            description="Open",
            detected_at_step=100,
            incident_id="open1",
        )
        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Open 2",
            description="Open",
            detected_at_step=200,
            incident_id="open2",
        )
        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.LOW,
            title="Resolved",
            description="Resolved",
            detected_at_step=300,
            incident_id="resolved1",
        )
        tracker.resolve("resolved1", "Fixed")

        open_incidents = tracker.list_open()
        ids = [i.id for i in open_incidents]

        assert "open1" in ids
        assert "open2" in ids
        assert "resolved1" not in ids

    def test_list_by_category(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Training Issue",
            description="Training",
            detected_at_step=100,
        )
        tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="Infra Issue",
            description="Infra",
            detected_at_step=200,
        )

        training = tracker.list_by_category(IncidentCategory.TRAINING)
        infra = tracker.list_by_category(IncidentCategory.INFRA)

        assert len(training) == 1
        assert len(infra) == 1

    def test_list_critical(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="Critical 1",
            description="Critical",
            detected_at_step=100,
            incident_id="crit1",
        )
        tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="Critical 2",
            description="Critical",
            detected_at_step=200,
            incident_id="crit2",
        )
        tracker.resolve("crit2", "Fixed")

        critical = tracker.list_critical()
        assert len(critical) == 1
        assert critical[0].id == "crit1"

    def test_search(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Training Error",
            description="Error in training",
            detected_at_step=100,
            run_id="run_001",
        )
        tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="OOM Crash",
            description="Out of memory",
            detected_at_step=200,
            run_id="run_001",
        )

        results = tracker.search(run_id="run_001")
        assert len(results) == 2

        results = tracker.search(severity=Severity.CRITICAL)
        assert len(results) == 1

        results = tracker.search(title_contains="error")
        assert len(results) == 1

    def test_get_stats(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir)

        tracker.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Open",
            description="Open",
            detected_at_step=100,
        )
        tracker.create_incident(
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Critical",
            detected_at_step=200,
        )

        stats = tracker.get_stats()

        assert stats["total"] == 2
        assert stats["open_count"] == 2
        assert stats["critical_count"] == 1
        assert stats["by_category"]["training"] == 1
        assert stats["by_category"]["infra"] == 1

    def test_persistence(self, temp_state_dir):
        # First tracker
        tracker1 = IncidentTracker(temp_state_dir)
        tracker1.create_incident(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Persistent",
            description="Should persist",
            detected_at_step=100,
            incident_id="persist_test",
        )

        # New tracker loads from disk
        tracker2 = IncidentTracker(temp_state_dir)
        incident = tracker2.get_incident("persist_test")

        assert incident is not None
        assert incident.title == "Persistent"

    def test_history_limit(self, temp_state_dir):
        tracker = IncidentTracker(temp_state_dir, history_limit=2)

        # Create and resolve several incidents
        for i in range(5):
            tracker.create_incident(
                category=IncidentCategory.TRAINING,
                severity=Severity.LOW,
                title=f"Incident {i}",
                description="Test",
                detected_at_step=i * 100,
                incident_id=f"inc_{i}",
            )
            tracker.resolve(f"inc_{i}", "Fixed")

        resolved = tracker.list_by_status(IncidentStatus.RESOLVED)
        assert len(resolved) == 2


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_init_and_get_tracker(self, temp_state_dir):
        init_incident_tracker(temp_state_dir)

        tracker = get_incident_tracker()
        assert tracker is not None

    def test_get_tracker_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_incident_tracker()


# =============================================================================
# Detector Tests
# =============================================================================

class TestPatternDetector:
    """Tests for pattern-based detection."""

    def test_pattern_match(self):
        config = {"pattern": "error", "field": "text", "flags": "i"}
        detector = PatternDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="pattern",
        )

        context = DetectionContext(step=100, text="An ERROR occurred")
        result = detector.check(rule, context)

        assert result.triggered is True
        assert "match" in result.template_vars

    def test_pattern_no_match(self):
        config = {"pattern": "error", "field": "text"}
        detector = PatternDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="pattern",
        )

        context = DetectionContext(step=100, text="All is well")
        result = detector.check(rule, context)

        assert result.triggered is False

    def test_pattern_error_field(self):
        config = {"pattern": "OOM", "field": "error_message"}
        detector = PatternDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.INFRA,
            severity=Severity.CRITICAL,
            detector_type="pattern",
        )

        context = DetectionContext(step=100, error_message="CUDA OOM error")
        result = detector.check(rule, context)

        assert result.triggered is True


class TestThresholdDetector:
    """Tests for threshold-based detection."""

    def test_threshold_gt(self):
        config = {"metric": "loss", "operator": "gt", "value": 5.0}
        detector = ThresholdDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="threshold",
        )

        context = DetectionContext(step=100, metrics={"loss": 10.0})
        result = detector.check(rule, context)

        assert result.triggered is True
        assert result.template_vars["value"] == 10.0

    def test_threshold_not_triggered(self):
        config = {"metric": "loss", "operator": "gt", "value": 5.0}
        detector = ThresholdDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="threshold",
        )

        context = DetectionContext(step=100, metrics={"loss": 2.0})
        result = detector.check(rule, context)

        assert result.triggered is False

    def test_threshold_lt(self):
        config = {"metric": "accuracy", "operator": "lt", "value": 0.5}
        detector = ThresholdDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            detector_type="threshold",
        )

        context = DetectionContext(step=100, metrics={"accuracy": 0.3})
        result = detector.check(rule, context)

        assert result.triggered is True


class TestConsecutiveDetector:
    """Tests for consecutive result detection."""

    def test_consecutive_failures(self):
        config = {"check": "failures", "count": 3}
        detector = ConsecutiveDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            detector_type="consecutive",
        )

        context = DetectionContext(
            step=100,
            recent_successes=[False, False, False],
        )
        result = detector.check(rule, context)

        assert result.triggered is True

    def test_consecutive_not_enough(self):
        config = {"check": "failures", "count": 3}
        detector = ConsecutiveDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            detector_type="consecutive",
        )

        context = DetectionContext(
            step=100,
            recent_successes=[False, False],
        )
        result = detector.check(rule, context)

        assert result.triggered is False

    def test_consecutive_mixed(self):
        config = {"check": "failures", "count": 3}
        detector = ConsecutiveDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.MEDIUM,
            detector_type="consecutive",
        )

        context = DetectionContext(
            step=100,
            recent_successes=[False, True, False],
        )
        result = detector.check(rule, context)

        assert result.triggered is False


class TestTrendDetector:
    """Tests for trend detection."""

    def test_trend_increasing(self):
        config = {
            "metric": "loss",
            "direction": "increasing",
            "window": 6,
            "min_change": 0.1,
        }
        detector = TrendDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="trend",
        )

        context = DetectionContext(
            step=100,
            recent_losses=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        result = detector.check(rule, context)

        assert result.triggered is True

    def test_trend_not_enough_change(self):
        config = {
            "metric": "loss",
            "direction": "increasing",
            "window": 6,
            "min_change": 0.5,
        }
        detector = TrendDetector(config)
        rule = IncidentRule(
            id="test",
            name="Test",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="trend",
        )

        context = DetectionContext(
            step=100,
            recent_losses=[0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
        )
        result = detector.check(rule, context)

        assert result.triggered is False


class TestIncidentDetector:
    """Tests for the main incident detector."""

    def test_detector_add_rule(self, temp_state_dir, sample_rule):
        tracker = IncidentTracker(temp_state_dir)
        detector = IncidentDetector(tracker)

        detector.add_rule(sample_rule)

        assert sample_rule in detector._rules
        assert sample_rule.id in detector._detectors

    def test_detector_check(self, temp_state_dir, sample_rule):
        tracker = IncidentTracker(temp_state_dir)
        detector = IncidentDetector(tracker, [sample_rule])

        context = DetectionContext(step=100, text="test error occurred")
        incidents = detector.check(context)

        assert len(incidents) == 1
        assert incidents[0].category == sample_rule.category

    def test_detector_cooldown(self, temp_state_dir, sample_rule):
        tracker = IncidentTracker(temp_state_dir)
        detector = IncidentDetector(tracker, [sample_rule])
        detector.set_cooldown(300)  # 5 minutes

        context = DetectionContext(step=100, text="test error")

        # First check triggers
        incidents1 = detector.check(context)
        assert len(incidents1) == 1

        # Second check (within cooldown) doesn't trigger
        incidents2 = detector.check(context)
        assert len(incidents2) == 0

    def test_detector_check_text_convenience(self, temp_state_dir, sample_rule):
        tracker = IncidentTracker(temp_state_dir)
        detector = IncidentDetector(tracker, [sample_rule])

        incidents = detector.check_text("error in training", step=100)

        assert len(incidents) == 1

    def test_detector_check_metrics_convenience(self, temp_state_dir):
        rule = IncidentRule(
            id="loss_rule",
            name="Loss Rule",
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            detector_type="threshold",
            detector_config={"metric": "loss", "operator": "gt", "value": 5.0},
        )

        tracker = IncidentTracker(temp_state_dir)
        detector = IncidentDetector(tracker, [rule])

        incidents = detector.check_metrics({"loss": 10.0}, step=100)

        assert len(incidents) == 1

    def test_get_detector_factory(self):
        pattern_det = get_detector("pattern", {"pattern": "test"})
        assert isinstance(pattern_det, PatternDetector)

        threshold_det = get_detector("threshold", {"metric": "loss"})
        assert isinstance(threshold_det, ThresholdDetector)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIncidentsIntegration:
    """Integration tests for the incidents module."""

    def test_full_workflow(self, temp_config_dir, temp_state_dir):
        """Test complete workflow: load rules, detect, track, resolve."""
        # Initialize
        init_incident_rule_registry(temp_config_dir)
        init_incident_tracker(temp_state_dir)

        tracker = get_incident_tracker()
        registry = get_incident_rule_registry()

        # Get all rules
        rules = list(registry)
        assert len(rules) == 3

        # Create detector with rules
        detector = IncidentDetector(tracker, rules)

        # Simulate OOM error
        incidents = detector.check_text(
            "CUDA out of memory - trying to allocate 4GB",
            step=1000,
            run_id="train_001",
        )
        assert len(incidents) == 1
        assert incidents[0].category == IncidentCategory.INFRA

        # Check tracker stats
        stats = tracker.get_stats()
        assert stats["open_count"] == 1
        assert stats["critical_count"] == 1

        # Resolve incident
        tracker.resolve(incidents[0].id, "Reduced batch size to 16")

        stats = tracker.get_stats()
        assert stats["open_count"] == 0

    def test_multiple_detections(self, temp_config_dir, temp_state_dir):
        """Test detecting multiple incident types."""
        init_incident_rule_registry(temp_config_dir)
        init_incident_tracker(temp_state_dir)

        tracker = get_incident_tracker()
        registry = get_incident_rule_registry()

        detector = IncidentDetector(tracker, list(registry))
        detector.set_cooldown(0)  # Disable cooldown for testing

        # Test loss spike
        context = DetectionContext(
            step=100,
            metrics={"loss": 10.0},
        )
        incidents = detector.check(context)
        assert len(incidents) == 1
        assert incidents[0].severity == Severity.HIGH

        # Test consecutive failures
        context = DetectionContext(
            step=200,
            recent_successes=[False, False, False, False, False],
        )
        incidents = detector.check(context)
        assert len(incidents) == 1

        # Check total
        assert len(tracker.list_open()) == 2


# =============================================================================
# Test with Real Configs
# =============================================================================

class TestRealConfigs:
    """Tests with real config files."""

    def test_load_real_incident_rules(self):
        """Test loading real incident rule configs."""
        real_config_dir = project_root / "configs"

        if not (real_config_dir / "incidents").exists():
            pytest.skip("No real incident configs found")

        rules = discover_incident_rules(real_config_dir)

        if not rules:
            pytest.skip("No incident rule configs in configs/incidents/")

        # Load first rule
        rule = load_incident_rule(rules[0], real_config_dir)
        assert rule.id is not None
        assert rule.detector_type is not None
