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
    remediation: Optional[str] = None  # How to fix if failed
    category: Optional[str] = None     # Group: "network", "storage", "training", etc.

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


# =============================================================================
# BLESSING - Temple verdict on effort validity
# =============================================================================
#
# The Blessing transforms Effort into Experience. When a Campaign returns
# to the Temple, the Cleric examines the training through Rituals and
# computes a quality_factor. Only blessed Effort becomes Experience.
#
# experience_gain = effort × quality_factor
#
# This is the "proof-of-effort" concept: not all training counts equally.
# Cursed training (bad data, broken evals, env issues) produces Effort
# but not Experience.
#


@dataclass
class Blessing:
    """
    Temple verdict on effort validity.

    When a Campaign completes training (or a phase of training), it returns
    to the Temple for judgment. The Cleric runs Rituals and computes a
    Blessing that determines how much of the Effort becomes Experience.

    Attributes:
        granted: Whether the blessing was granted
        quality_factor: Multiplier for effort→experience (0.0 to 1.0)
        orders_consulted: Which ritual orders were consulted
        verdict: Overall verdict (blessed, cursed, partial)
        reason: Human-readable explanation
        campaign_id: Campaign that was judged
        effort_examined: Amount of effort being judged
        experience_awarded: Effort × quality_factor
        timestamp: When blessing was computed
    """
    granted: bool
    quality_factor: float  # 0.0 - 1.0
    orders_consulted: List[str] = field(default_factory=list)
    verdict: str = "unknown"  # blessed, cursed, partial
    reason: str = ""
    campaign_id: Optional[str] = None
    effort_examined: float = 0.0
    experience_awarded: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def grant(
        cls,
        quality: float,
        orders: List[str],
        reason: str,
        campaign_id: Optional[str] = None,
        effort: float = 0.0,
    ) -> "Blessing":
        """
        Grant a blessing with the given quality factor.

        Args:
            quality: Quality factor (0.0 to 1.0)
            orders: Which orders were consulted
            reason: Why this quality was assigned
            campaign_id: Campaign being blessed
            effort: Amount of effort being judged

        Returns:
            Blessing with granted=True
        """
        return cls(
            granted=True,
            quality_factor=quality,
            orders_consulted=orders,
            verdict="blessed" if quality >= 0.8 else "partial",
            reason=reason,
            campaign_id=campaign_id,
            effort_examined=effort,
            experience_awarded=effort * quality,
        )

    @classmethod
    def deny(
        cls,
        orders: List[str],
        reason: str,
        campaign_id: Optional[str] = None,
        effort: float = 0.0,
    ) -> "Blessing":
        """
        Deny blessing (curse the effort).

        Args:
            orders: Which orders were consulted
            reason: Why the blessing was denied
            campaign_id: Campaign being cursed
            effort: Amount of effort that won't count

        Returns:
            Blessing with granted=False, quality_factor=0
        """
        return cls(
            granted=False,
            quality_factor=0.0,
            orders_consulted=orders,
            verdict="cursed",
            reason=reason,
            campaign_id=campaign_id,
            effort_examined=effort,
            experience_awarded=0.0,
        )

    @classmethod
    def from_ceremony(
        cls,
        results: Dict[str, "RitualResult"],
        campaign_id: Optional[str] = None,
        effort: float = 0.0,
    ) -> "Blessing":
        """
        Compute blessing from ceremony results.

        The quality factor is computed based on ritual outcomes:
        - All pass (ok): quality = 1.0
        - Some warnings: quality = 0.8
        - Critical failures: quality = 0.0 (denied)

        Args:
            results: Dict mapping ritual_id to RitualResult
            campaign_id: Campaign being blessed
            effort: Amount of effort being judged

        Returns:
            Computed Blessing
        """
        orders = list(results.keys())

        # Count outcomes
        n_ok = sum(1 for r in results.values() if r.status == "ok")
        n_warn = sum(1 for r in results.values() if r.status == "warn")
        n_fail = sum(1 for r in results.values() if r.status == "fail")
        n_total = len(results)

        # Critical orders that must pass
        critical_orders = {"forge", "oracle", "champion"}
        critical_failed = [
            rid for rid, r in results.items()
            if rid in critical_orders and r.status == "fail"
        ]

        # Compute quality factor
        if critical_failed:
            # Critical failure = cursed
            return cls.deny(
                orders=orders,
                reason=f"Critical orders failed: {', '.join(critical_failed)}",
                campaign_id=campaign_id,
                effort=effort,
            )

        if n_fail > 0:
            # Non-critical failures = partial blessing
            quality = max(0.3, 1.0 - (n_fail * 0.3))
            return cls.grant(
                quality=quality,
                orders=orders,
                reason=f"{n_fail} order(s) failed (non-critical)",
                campaign_id=campaign_id,
                effort=effort,
            )

        if n_warn > 0:
            # Warnings = slightly reduced quality
            quality = max(0.7, 1.0 - (n_warn * 0.1))
            return cls.grant(
                quality=quality,
                orders=orders,
                reason=f"{n_warn} order(s) warned",
                campaign_id=campaign_id,
                effort=effort,
            )

        # All passed = full blessing
        return cls.grant(
            quality=1.0,
            orders=orders,
            reason=f"All {n_total} orders passed",
            campaign_id=campaign_id,
            effort=effort,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "granted": self.granted,
            "quality_factor": self.quality_factor,
            "orders_consulted": self.orders_consulted,
            "verdict": self.verdict,
            "reason": self.reason,
            "campaign_id": self.campaign_id,
            "effort_examined": self.effort_examined,
            "experience_awarded": self.experience_awarded,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blessing":
        """Create from dictionary."""
        return cls(
            granted=data.get("granted", False),
            quality_factor=data.get("quality_factor", 0.0),
            orders_consulted=data.get("orders_consulted", []),
            verdict=data.get("verdict", "unknown"),
            reason=data.get("reason", ""),
            campaign_id=data.get("campaign_id"),
            effort_examined=data.get("effort_examined", 0.0),
            experience_awarded=data.get("experience_awarded", 0.0),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )
