"""
Dispatch Types - Data structures for the Quest Dispatcher.

The Dispatcher coordinates quest flow:
- Checks if hero needs work
- Requests quests from skill trainers
- Validates quest quality
- Posts to the quest board
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence


class DispatchDecision(Enum):
    """Outcome of a dispatch check."""
    DISPATCH = "dispatch"           # Hero needs quests, go get them
    WAIT = "wait"                   # Cooldown or queue is full
    UNAVAILABLE = "unavailable"     # Trainer offline
    DISABLED = "disabled"           # Auto-dispatch disabled


class QuestVerdict(Enum):
    """Quality gate verdict on quest batch."""
    APPROVED = "approved"           # Safe to post to quest board
    CONDITIONAL = "conditional"     # Minor issues, review before posting
    REJECTED = "rejected"           # Too many issues, do not post


@dataclass
class DispatchStatus:
    """
    Current status of the Quest Dispatcher.

    Tracks dispatch cycles, quest throughput, and trainer availability.
    """
    # Dispatch state
    enabled: bool = True
    last_dispatch_time: Optional[datetime] = None
    dispatch_count: int = 0

    # Quest throughput
    total_quests_requested: int = 0
    total_quests_approved: int = 0
    total_quests_rejected: int = 0

    # Trainer status
    active_skill: str = "binary"
    trainer_online: bool = False

    # Quality metrics
    approval_rate: float = 1.0
    avg_quality_score: float = 1.0
    quality_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "enabled": self.enabled,
            "last_dispatch_time": self.last_dispatch_time.isoformat() if self.last_dispatch_time else None,
            "dispatch_count": self.dispatch_count,
            "total_quests_requested": self.total_quests_requested,
            "total_quests_approved": self.total_quests_approved,
            "total_quests_rejected": self.total_quests_rejected,
            "active_skill": self.active_skill,
            "trainer_online": self.trainer_online,
            "approval_rate": self.approval_rate,
            "avg_quality_score": self.avg_quality_score,
        }


@dataclass
class QualityReport:
    """
    Report from the Quality Gate inspection.

    Contains detailed results of each check and overall verdict.
    """
    verdict: QuestVerdict
    total_checks: int
    passed_checks: int
    failed_checks: int

    # Per-check results
    check_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Human-readable recommendation
    recommendation: str = ""

    @property
    def pass_rate(self) -> float:
        """Fraction of checks that passed."""
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "verdict": self.verdict.value,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "pass_rate": self.pass_rate,
            "check_results": self.check_results,
            "recommendation": self.recommendation,
        }


@dataclass
class ProgressionStatus:
    """
    Hero's progression status for a skill.

    Tracks current level, accuracy history, and progression eligibility.
    """
    skill_id: str
    skill_name: str

    # Current state
    current_level: int
    total_levels: int
    level_name: str

    # Accuracy tracking
    accuracy_threshold: Optional[float]  # None = mastered
    evals_at_level: int
    avg_accuracy: Optional[float]

    # Progression eligibility
    ready_to_advance: bool
    reason: str

    # History
    progressions_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "current_level": self.current_level,
            "total_levels": self.total_levels,
            "level_name": self.level_name,
            "accuracy_threshold": self.accuracy_threshold,
            "evals_at_level": self.evals_at_level,
            "avg_accuracy": self.avg_accuracy,
            "ready_to_advance": self.ready_to_advance,
            "reason": self.reason,
            "progressions_completed": self.progressions_completed,
        }


@dataclass
class DispatchResult:
    """
    Result of a dispatch cycle.

    Contains outcome, quests dispatched, and quality report.
    """
    success: bool
    decision: DispatchDecision
    reason: str

    # Quest details (if dispatched)
    skill_id: Optional[str] = None
    level: Optional[int] = None
    quests_requested: int = 0
    quests_approved: int = 0

    # Quality report (if quality check ran)
    quality_report: Optional[QualityReport] = None

    # Output file (if queued)
    output_file: Optional[str] = None

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "success": self.success,
            "decision": self.decision.value,
            "reason": self.reason,
            "skill_id": self.skill_id,
            "level": self.level,
            "quests_requested": self.quests_requested,
            "quests_approved": self.quests_approved,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "output_file": self.output_file,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }
