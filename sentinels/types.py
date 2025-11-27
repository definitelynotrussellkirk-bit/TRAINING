"""
Sentinel Types - Core types for the guardian and protection system.

The Sentinels are the realm's protectors, watching for danger and
keeping the training systems healthy:

    SentinelState   - Current state of a sentinel
    ThreatLevel     - Severity of detected issues
    HealthReport    - System health assessment
    AlertRecord     - Record of detected threats

RPG Flavor:
    The Sentinels patrol the training grounds, ever vigilant against
    crashes, stalls, and corruption. Each has a specialty:
    Guardian watches the daemon, Scout detects stuck training,
    Healer recovers from crashes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SentinelState(Enum):
    """Current state of a sentinel."""
    DORMANT = "dormant"        # Not active
    PATROLLING = "patrolling"  # Actively monitoring
    ALERTED = "alerted"        # Detected issue
    RESPONDING = "responding"  # Taking action
    RESTING = "resting"        # Temporarily paused


class ThreatLevel(Enum):
    """Severity of detected threats."""
    CLEAR = "clear"            # No issues
    MINOR = "minor"            # Small concern
    MODERATE = "moderate"      # Needs attention
    SEVERE = "severe"          # Urgent action needed
    CRITICAL = "critical"      # System at risk


class HealthDomain(Enum):
    """Areas of system health."""
    DAEMON = "daemon"          # Training daemon
    GPU = "gpu"                # GPU resources
    DISK = "disk"              # Disk space
    QUEUE = "queue"            # Training queue
    MODEL = "model"            # Model integrity
    CONFIG = "config"          # Configuration
    NETWORK = "network"        # Network/API


@dataclass
class HealthReport:
    """
    Health assessment from a sentinel patrol.
    """
    domain: HealthDomain
    status: ThreatLevel
    message: str

    # Details
    checks_passed: int = 0
    checks_failed: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    # Timing
    checked_at: Optional[datetime] = None

    @property
    def is_healthy(self) -> bool:
        return self.status in (ThreatLevel.CLEAR, ThreatLevel.MINOR)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "status": self.status.value,
            "message": self.message,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "details": self.details,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "is_healthy": self.is_healthy,
        }


@dataclass
class PatrolReport:
    """
    Full patrol report from all sentinels.
    """
    overall_status: ThreatLevel
    domain_reports: List[HealthReport] = field(default_factory=list)
    patrol_time: Optional[datetime] = None
    patrol_duration_ms: float = 0.0

    @property
    def is_all_clear(self) -> bool:
        return self.overall_status == ThreatLevel.CLEAR

    @property
    def threats_detected(self) -> List[HealthReport]:
        return [r for r in self.domain_reports if not r.is_healthy]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "is_all_clear": self.is_all_clear,
            "domains": [r.to_dict() for r in self.domain_reports],
            "threats": [r.to_dict() for r in self.threats_detected],
            "patrol_time": self.patrol_time.isoformat() if self.patrol_time else None,
            "patrol_duration_ms": self.patrol_duration_ms,
        }


@dataclass
class AlertRecord:
    """
    Record of a detected threat.
    """
    alert_id: str
    threat_level: ThreatLevel
    domain: HealthDomain
    title: str
    description: str

    # Response
    auto_response: bool = False
    response_action: Optional[str] = None
    resolved: bool = False

    # Timing
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "threat_level": self.threat_level.value,
            "domain": self.domain.value,
            "title": self.title,
            "description": self.description,
            "auto_response": self.auto_response,
            "response_action": self.response_action,
            "resolved": self.resolved,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
