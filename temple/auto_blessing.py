"""
Auto-Blessing from Eval Results
===============================

Automatically computes Temple Blessings from evaluation results.

Instead of running manual Rituals, this module computes quality_factor
directly from the EvaluationLedger, providing automatic feedback on
training quality.

The Blessing formula:

    quality = (accuracy_weight × avg_accuracy) + (coverage_weight × skill_coverage) + (trend_weight × improvement_trend)

Where:
- avg_accuracy: Average accuracy across all recent evals
- skill_coverage: What fraction of skills have been evaluated
- improvement_trend: Whether accuracy is improving over time

Usage:
    from temple.auto_blessing import compute_blessing_from_evals

    # Get blessing for a checkpoint
    blessing = compute_blessing_from_evals(
        checkpoint_step=1000,
        campaign_id="campaign-001",
    )

    # Get blessing for a time window
    blessing = compute_blessing_for_session(
        start_step=500,
        end_step=1000,
        campaign_id="campaign-001",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging

from temple.schemas import Blessing

logger = logging.getLogger(__name__)


# =============================================================================
# QUALITY FACTOR COMPUTATION
# =============================================================================

@dataclass
class EvalSnapshot:
    """Snapshot of eval state for blessing computation."""
    total_evals: int = 0
    avg_accuracy: float = 0.0
    skill_coverage: float = 0.0  # 0-1: fraction of skills evaluated
    skills_evaluated: List[str] = None
    improvement_trend: float = 0.0  # -1 to 1: declining to improving
    recent_accuracies: List[float] = None
    eval_ids: List[str] = None

    def __post_init__(self):
        if self.skills_evaluated is None:
            self.skills_evaluated = []
        if self.recent_accuracies is None:
            self.recent_accuracies = []
        if self.eval_ids is None:
            self.eval_ids = []


def get_eval_snapshot(
    checkpoint_step: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    campaign_id: Optional[str] = None,
    lookback_evals: int = 10,
) -> EvalSnapshot:
    """
    Get snapshot of eval state for blessing computation.

    Args:
        checkpoint_step: Specific checkpoint to check (gets evals at or before)
        start_step: Start of step range (alternative to checkpoint_step)
        end_step: End of step range
        campaign_id: Campaign to filter by
        lookback_evals: How many recent evals to consider for trend

    Returns:
        EvalSnapshot with aggregated metrics
    """
    from core.evaluation_ledger import get_eval_ledger

    ledger = get_eval_ledger()

    # Get all relevant evals
    all_evals = ledger.list_all(limit=500)

    # Filter by checkpoint range
    if checkpoint_step is not None:
        evals = [e for e in all_evals if e.checkpoint_step <= checkpoint_step]
    elif start_step is not None and end_step is not None:
        evals = [e for e in all_evals if start_step <= e.checkpoint_step <= end_step]
    else:
        evals = all_evals

    # Filter by campaign
    if campaign_id:
        evals = [e for e in evals if e.campaign_id == campaign_id]

    if not evals:
        return EvalSnapshot()

    # Compute metrics
    accuracies = [e.accuracy for e in evals]
    avg_accuracy = sum(accuracies) / len(accuracies)

    skills_evaluated = list(set(e.skill for e in evals))
    # Assume 2 skills for now (sy, bin) - could be dynamic
    known_skills = ["sy", "bin"]
    skill_coverage = len(skills_evaluated) / max(len(known_skills), 1)

    # Compute trend from recent evals
    recent = sorted(evals, key=lambda e: e.timestamp)[-lookback_evals:]
    if len(recent) >= 3:
        recent_acc = [e.accuracy for e in recent]
        # Simple trend: compare first half to second half
        mid = len(recent_acc) // 2
        first_half = sum(recent_acc[:mid]) / max(mid, 1)
        second_half = sum(recent_acc[mid:]) / max(len(recent_acc) - mid, 1)
        trend = (second_half - first_half) * 2  # Scale to roughly -1 to 1
        trend = max(-1.0, min(1.0, trend))
    else:
        trend = 0.0

    return EvalSnapshot(
        total_evals=len(evals),
        avg_accuracy=avg_accuracy,
        skill_coverage=skill_coverage,
        skills_evaluated=skills_evaluated,
        improvement_trend=trend,
        recent_accuracies=[e.accuracy for e in recent],
        eval_ids=[e.key for e in evals],
    )


def compute_quality_factor(snapshot: EvalSnapshot) -> tuple[float, str]:
    """
    Compute quality_factor from eval snapshot.

    Formula:
        quality = (0.6 × accuracy) + (0.2 × coverage) + (0.2 × (trend + 1) / 2)

    Returns:
        (quality_factor, reason)
    """
    if snapshot.total_evals == 0:
        return 0.0, "No evaluations found"

    # Weights
    accuracy_weight = 0.6
    coverage_weight = 0.2
    trend_weight = 0.2

    # Normalize trend from [-1, 1] to [0, 1]
    normalized_trend = (snapshot.improvement_trend + 1) / 2

    quality = (
        accuracy_weight * snapshot.avg_accuracy
        + coverage_weight * snapshot.skill_coverage
        + trend_weight * normalized_trend
    )

    # Clamp to [0, 1]
    quality = max(0.0, min(1.0, quality))

    # Generate reason
    parts = []
    parts.append(f"accuracy={snapshot.avg_accuracy:.0%}")
    parts.append(f"coverage={snapshot.skill_coverage:.0%}")
    if snapshot.improvement_trend > 0.1:
        parts.append("improving")
    elif snapshot.improvement_trend < -0.1:
        parts.append("declining")
    else:
        parts.append("stable")

    reason = ", ".join(parts)

    return quality, reason


# =============================================================================
# BLESSING COMPUTATION
# =============================================================================

def compute_blessing_from_evals(
    checkpoint_step: int,
    campaign_id: Optional[str] = None,
    effort: float = 0.0,
) -> Blessing:
    """
    Compute a Blessing for a specific checkpoint based on evals.

    Args:
        checkpoint_step: The checkpoint to bless
        campaign_id: Campaign ID for filtering
        effort: Amount of effort being blessed (for XP calculation)

    Returns:
        Blessing with quality computed from evals
    """
    snapshot = get_eval_snapshot(
        checkpoint_step=checkpoint_step,
        campaign_id=campaign_id,
    )

    quality, reason = compute_quality_factor(snapshot)

    # Determine verdict
    if quality >= 0.8:
        verdict = "blessed"
    elif quality >= 0.4:
        verdict = "partial"
    else:
        verdict = "cursed"

    return Blessing(
        granted=quality > 0.0,
        quality_factor=quality,
        orders_consulted=["eval_ledger"],
        verdict=verdict,
        reason=f"Auto-blessing from {snapshot.total_evals} evals: {reason}",
        campaign_id=campaign_id,
        effort_examined=effort,
        experience_awarded=effort * quality,
    )


def compute_blessing_for_session(
    start_step: int,
    end_step: int,
    campaign_id: Optional[str] = None,
    effort: Optional[float] = None,
) -> Blessing:
    """
    Compute a Blessing for a training session (step range).

    Args:
        start_step: First step of session
        end_step: Last step of session
        campaign_id: Campaign ID
        effort: Effort spent (if None, estimated from step count)

    Returns:
        Blessing for the session
    """
    snapshot = get_eval_snapshot(
        start_step=start_step,
        end_step=end_step,
        campaign_id=campaign_id,
    )

    quality, reason = compute_quality_factor(snapshot)

    # Estimate effort if not provided
    if effort is None:
        effort = float(end_step - start_step)

    # Determine verdict
    if quality >= 0.8:
        verdict = "blessed"
    elif quality >= 0.4:
        verdict = "partial"
    else:
        verdict = "cursed"

    return Blessing(
        granted=quality > 0.0,
        quality_factor=quality,
        orders_consulted=["eval_ledger"],
        verdict=verdict,
        reason=f"Session blessing ({start_step}-{end_step}): {reason}",
        campaign_id=campaign_id,
        effort_examined=effort,
        experience_awarded=effort * quality,
    )


# =============================================================================
# AUTO-BLESSING HOOKS
# =============================================================================

def auto_bless_checkpoint(
    checkpoint_step: int,
    campaign_id: Optional[str] = None,
) -> Optional[Blessing]:
    """
    Automatically compute and record blessing for a checkpoint.

    Called after checkpoint save to update blessing status.
    Returns None if no evals are available yet.

    Args:
        checkpoint_step: The checkpoint that was saved
        campaign_id: Campaign ID

    Returns:
        Blessing if evals exist, None otherwise
    """
    snapshot = get_eval_snapshot(
        checkpoint_step=checkpoint_step,
        campaign_id=campaign_id,
    )

    if snapshot.total_evals == 0:
        logger.debug(f"No evals for checkpoint {checkpoint_step}, skipping blessing")
        return None

    blessing = compute_blessing_from_evals(
        checkpoint_step=checkpoint_step,
        campaign_id=campaign_id,
    )

    # Log the blessing
    logger.info(
        f"[Temple] Checkpoint {checkpoint_step} {blessing.verdict}: "
        f"quality={blessing.quality_factor:.2f} ({blessing.reason})"
    )

    # Optionally store blessing in ledger
    _store_blessing(checkpoint_step, blessing)

    return blessing


def _store_blessing(checkpoint_step: int, blessing: Blessing) -> None:
    """Store blessing in checkpoint ledger (if available)."""
    try:
        from core.checkpoint_ledger import get_ledger

        ledger = get_ledger()
        record = ledger.get(checkpoint_step)
        if record:
            # Add blessing to metadata (if we had that field)
            # For now, just log it
            logger.debug(f"Blessing for step {checkpoint_step} computed")
    except Exception as e:
        logger.debug(f"Could not store blessing: {e}")


# =============================================================================
# SKILL-SPECIFIC BLESSING
# =============================================================================

def compute_skill_blessing(
    skill_id: str,
    checkpoint_step: int,
    campaign_id: Optional[str] = None,
) -> tuple[float, str]:
    """
    Compute blessing quality for a specific skill.

    Returns:
        (quality_factor, reason) for the skill
    """
    from core.evaluation_ledger import get_eval_ledger

    ledger = get_eval_ledger()
    evals = ledger.get_by_skill(skill_id)

    # Filter to checkpoint
    evals = [e for e in evals if e.checkpoint_step <= checkpoint_step]

    if campaign_id:
        evals = [e for e in evals if e.campaign_id == campaign_id]

    if not evals:
        return 0.0, f"No evals for skill {skill_id}"

    # Get best and recent accuracy
    accuracies = [e.accuracy for e in evals]
    avg_accuracy = sum(accuracies) / len(accuracies)
    best_accuracy = max(accuracies)

    # Quality based on average with best as bonus
    quality = 0.7 * avg_accuracy + 0.3 * best_accuracy

    return quality, f"{skill_id}: avg={avg_accuracy:.0%}, best={best_accuracy:.0%}"


def get_all_skill_blessings(
    checkpoint_step: int,
    campaign_id: Optional[str] = None,
) -> Dict[str, tuple[float, str]]:
    """
    Get blessing quality for all skills at a checkpoint.

    Returns:
        Dict mapping skill_id to (quality, reason)
    """
    from core.evaluation_ledger import get_eval_ledger

    ledger = get_eval_ledger()
    summary = ledger.summary()

    results = {}
    for skill_id in summary.get("by_skill", {}).keys():
        quality, reason = compute_skill_blessing(
            skill_id, checkpoint_step, campaign_id
        )
        results[skill_id] = (quality, reason)

    return results


# =============================================================================
# BLESSING HISTORY
# =============================================================================

@dataclass
class BlessingHistoryEntry:
    """Entry in blessing history."""
    checkpoint_step: int
    quality_factor: float
    verdict: str
    timestamp: str
    skills_evaluated: List[str]


def get_blessing_history(
    campaign_id: Optional[str] = None,
    limit: int = 20,
) -> List[BlessingHistoryEntry]:
    """
    Get history of blessings for a campaign.

    Reconstructs blessing history from evaluation ledger.
    """
    from core.checkpoint_ledger import get_ledger

    ledger = get_ledger()
    checkpoints = ledger.list_all(limit=limit)

    history = []
    for cp in checkpoints:
        blessing = compute_blessing_from_evals(
            checkpoint_step=cp.step,
            campaign_id=campaign_id,
        )

        snapshot = get_eval_snapshot(
            checkpoint_step=cp.step,
            campaign_id=campaign_id,
        )

        history.append(BlessingHistoryEntry(
            checkpoint_step=cp.step,
            quality_factor=blessing.quality_factor,
            verdict=blessing.verdict,
            timestamp=cp.timestamp,
            skills_evaluated=snapshot.skills_evaluated,
        ))

    return history
