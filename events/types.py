"""
Event Types for Global Announcement Channel

All components emit events through the broadcaster using these types.
Tavern UI subscribes via SSE to receive real-time updates.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import json


class EventType(str, Enum):
    """All event types in the system."""

    # Queue Events
    QUEUE_EMPTY = "queue.empty"
    QUEUE_LOW = "queue.low"
    QUEUE_ITEM_ADDED = "queue.item_added"
    QUEUE_ITEM_STARTED = "queue.item_started"
    QUEUE_ITEM_COMPLETED = "queue.item_completed"
    QUEUE_ITEM_FAILED = "queue.item_failed"

    # Data Generation Events
    DATA_NEED = "data.need"              # Daemon requests data
    DATA_GENERATING = "data.generating"   # DataManager starts generating
    DATA_GENERATED = "data.generated"     # Generation complete
    DATA_QUEUED = "data.queued"          # Data added to queue
    DATA_QUALITY_PASS = "data.quality_pass"
    DATA_QUALITY_FAIL = "data.quality_fail"

    # Training Events
    TRAINING_STARTED = "training.started"
    TRAINING_STEP = "training.step"
    TRAINING_CHECKPOINT = "training.checkpoint"
    TRAINING_PAUSED = "training.paused"
    TRAINING_RESUMED = "training.resumed"
    TRAINING_STOPPED = "training.stopped"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_ERROR = "training.error"

    # Hero Events (DIO)
    HERO_LEVEL_UP = "hero.level_up"
    HERO_SKILL_UP = "hero.skill_up"
    HERO_STAT_CHANGE = "hero.stat_change"

    # System Events
    DAEMON_STARTED = "daemon.started"
    DAEMON_STOPPED = "daemon.stopped"
    DAEMON_HEARTBEAT = "daemon.heartbeat"
    SERVICE_UP = "service.up"
    SERVICE_DOWN = "service.down"

    # Vault Events
    CHECKPOINT_SAVED = "vault.checkpoint_saved"
    CHECKPOINT_PROMOTED = "vault.checkpoint_promoted"
    TRANSFER_STARTED = "vault.transfer_started"
    TRANSFER_COMPLETED = "vault.transfer_completed"

    # Analytics Events (Model Archaeology)
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    ANALYSIS_DRIFT_DETECTED = "analysis.drift_detected"


class Severity(str, Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class Event:
    """
    A single event in the global channel.

    All components emit events using this schema.
    """
    type: EventType
    message: str
    severity: Severity = Severity.INFO
    source: str = "system"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            # Generate unique ID from timestamp + type
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            self.id = f"{ts}_{self.type.value.replace('.', '_')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "message": self.message,
            "severity": self.severity.value if isinstance(self.severity, Severity) else self.severity,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"id: {self.id}\nevent: {self.type.value}\ndata: {self.to_json()}\n\n"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Create Event from dictionary."""
        return cls(
            type=EventType(d["type"]) if isinstance(d["type"], str) else d["type"],
            message=d["message"],
            severity=Severity(d.get("severity", "info")),
            source=d.get("source", "system"),
            data=d.get("data", {}),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            id=d.get("id"),
        )


# Pre-built event factories for common scenarios
def queue_empty_event(queue_depth: int = 0, min_depth: int = 2) -> Event:
    """Queue is empty, need data."""
    return Event(
        type=EventType.QUEUE_EMPTY,
        message=f"Queue empty! Need {min_depth} items, have {queue_depth}",
        severity=Severity.WARNING,
        source="daemon",
        data={"queue_depth": queue_depth, "min_depth": min_depth},
    )


def data_need_event(skill: str, count: int, reason: str = "queue_low") -> Event:
    """Daemon requesting data generation."""
    return Event(
        type=EventType.DATA_NEED,
        message=f"Requesting {count:,} {skill.upper()} samples",
        severity=Severity.INFO,
        source="daemon",
        data={"skill": skill, "count": count, "reason": reason},
    )


def data_generating_event(skill: str, count: int, level: Optional[int] = None) -> Event:
    """DataManager starting generation."""
    level_str = f" L{level}" if level else ""
    return Event(
        type=EventType.DATA_GENERATING,
        message=f"Generating {count:,} {skill.upper()}{level_str} samples...",
        severity=Severity.INFO,
        source="data_manager",
        data={"skill": skill, "count": count, "level": level},
    )


def data_generated_event(skill: str, count: int, duration: float) -> Event:
    """Generation complete."""
    return Event(
        type=EventType.DATA_GENERATED,
        message=f"Generated {count:,} {skill.upper()} samples in {duration:.1f}s",
        severity=Severity.SUCCESS,
        source="data_manager",
        data={"skill": skill, "count": count, "duration_sec": duration},
    )


def data_queued_event(filename: str, count: int, priority: str = "normal") -> Event:
    """Data queued for training."""
    return Event(
        type=EventType.DATA_QUEUED,
        message=f"Queued {count:,} samples ({priority}): {filename}",
        severity=Severity.SUCCESS,
        source="data_manager",
        data={"filename": filename, "count": count, "priority": priority},
    )


def quality_pass_event(tests_passed: int, total_tests: int) -> Event:
    """Quality tests passed."""
    return Event(
        type=EventType.DATA_QUALITY_PASS,
        message=f"Quality check PASSED ({tests_passed}/{total_tests} tests)",
        severity=Severity.SUCCESS,
        source="data_manager",
        data={"passed": tests_passed, "total": total_tests},
    )


def quality_fail_event(tests_passed: int, total_tests: int, failures: list) -> Event:
    """Quality tests failed."""
    return Event(
        type=EventType.DATA_QUALITY_FAIL,
        message=f"Quality check FAILED ({tests_passed}/{total_tests} tests)",
        severity=Severity.ERROR,
        source="data_manager",
        data={"passed": tests_passed, "total": total_tests, "failures": failures},
    )


def training_started_event(file: str, step: int) -> Event:
    """Training started on file."""
    return Event(
        type=EventType.TRAINING_STARTED,
        message=f"Training started: {file} (step {step:,})",
        severity=Severity.INFO,
        source="daemon",
        data={"file": file, "step": step},
    )


def training_completed_event(file: str, steps: int, duration: float) -> Event:
    """Training completed."""
    return Event(
        type=EventType.TRAINING_COMPLETED,
        message=f"Completed: {file} ({steps:,} steps in {duration:.0f}s)",
        severity=Severity.SUCCESS,
        source="daemon",
        data={"file": file, "steps": steps, "duration_sec": duration},
    )


def checkpoint_saved_event(step: int, loss: float, path: str) -> Event:
    """Checkpoint saved."""
    return Event(
        type=EventType.CHECKPOINT_SAVED,
        message=f"Checkpoint saved: step {step:,} (loss: {loss:.4f})",
        severity=Severity.SUCCESS,
        source="trainer",
        data={"step": step, "loss": loss, "path": path},
    )


def level_up_event(old_level: int, new_level: int) -> Event:
    """Hero leveled up!"""
    return Event(
        type=EventType.HERO_LEVEL_UP,
        message=f"LEVEL UP! {old_level} -> {new_level}",
        severity=Severity.SUCCESS,
        source="hero",
        data={"old_level": old_level, "new_level": new_level},
    )


def daemon_heartbeat_event(status: str, queue_depth: int, step: int) -> Event:
    """Periodic daemon heartbeat."""
    return Event(
        type=EventType.DAEMON_HEARTBEAT,
        message=f"Daemon alive: {status} (queue: {queue_depth}, step: {step:,})",
        severity=Severity.DEBUG,
        source="daemon",
        data={"status": status, "queue_depth": queue_depth, "step": step},
    )


# =============================================================================
# ANALYTICS EVENTS (Model Archaeology)
# =============================================================================

def analysis_started_event(job_type: str, checkpoint_step: int, hero_id: str = "") -> Event:
    """Analysis job started."""
    return Event(
        type=EventType.ANALYSIS_STARTED,
        message=f"Analyzing checkpoint {checkpoint_step:,} ({job_type})",
        severity=Severity.INFO,
        source="archaeologist",
        data={
            "job_type": job_type,
            "checkpoint_step": checkpoint_step,
            "hero_id": hero_id,
        },
    )


def analysis_completed_event(
    job_type: str,
    checkpoint_step: int,
    duration_sec: float = 0,
    num_layers: int = 0,
    most_changed_layer: Optional[str] = None,
) -> Event:
    """Analysis job completed."""
    msg = f"Analysis complete: checkpoint {checkpoint_step:,}"
    if most_changed_layer:
        msg += f" (most drift: {most_changed_layer})"

    return Event(
        type=EventType.ANALYSIS_COMPLETED,
        message=msg,
        severity=Severity.SUCCESS,
        source="archaeologist",
        data={
            "job_type": job_type,
            "checkpoint_step": checkpoint_step,
            "duration_sec": duration_sec,
            "num_layers": num_layers,
            "most_changed_layer": most_changed_layer,
        },
    )


def analysis_failed_event(
    job_type: str,
    checkpoint_step: int,
    error: str,
) -> Event:
    """Analysis job failed."""
    return Event(
        type=EventType.ANALYSIS_FAILED,
        message=f"Analysis failed: checkpoint {checkpoint_step:,} - {error}",
        severity=Severity.ERROR,
        source="archaeologist",
        data={
            "job_type": job_type,
            "checkpoint_step": checkpoint_step,
            "error": error,
        },
    )


def drift_detected_event(
    checkpoint_step: int,
    layer_name: str,
    drift_l2: float,
    threshold: float = 0.1,
) -> Event:
    """Significant drift detected in a layer."""
    return Event(
        type=EventType.ANALYSIS_DRIFT_DETECTED,
        message=f"Drift alert: {layer_name} changed by {drift_l2:.4f} (threshold: {threshold})",
        severity=Severity.WARNING,
        source="archaeologist",
        data={
            "checkpoint_step": checkpoint_step,
            "layer_name": layer_name,
            "drift_l2": drift_l2,
            "threshold": threshold,
        },
    )
