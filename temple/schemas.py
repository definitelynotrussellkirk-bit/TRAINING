"""
Temple Schemas - Result types for diagnostic rituals.

These dataclasses define the structure of ritual check results,
used by both CLI and HTTP API responses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

ResultStatus = Literal["ok", "warn", "fail", "skip"]


@dataclass
class RitualCheckResult:
    """Result of a single diagnostic check within a ritual."""

    id: str                     # e.g., "realm_state_read"
    name: str                   # Human-friendly name
    description: str            # What this check does
    status: ResultStatus
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def duration_ms(self) -> Optional[float]:
        """Get check duration in milliseconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class RitualResult:
    """Result of running a complete diagnostic ritual."""

    ritual_id: str              # "quick", "api", "full"
    name: str                   # "Ritual of Quick"
    description: str
    status: ResultStatus        # Aggregate over checks
    checks: List[RitualCheckResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    @property
    def ok(self) -> bool:
        """True if ritual passed (status is 'ok')."""
        return self.status == "ok"

    def duration_ms(self) -> Optional[float]:
        """Get total ritual duration in milliseconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    def count_by_status(self) -> Dict[str, int]:
        """Count checks by status."""
        counts: Dict[str, int] = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts
