"""
Checkpoint Health Scoring - Grade Your Checkpoints
====================================================

When you have 50 checkpoints, which ones are "good"? This module
provides A/B/C/D/F grades based on multiple quality signals.

Grade Breakdown:
    A (90-100): Excellent - Loss improving, gradients healthy, good val metrics
    B (80-89):  Good - Solid checkpoint, minor issues
    C (70-79):  Average - Usable but has concerns
    D (60-69):  Poor - Significant issues, consider skipping
    F (0-59):   Failing - Critical problems, do not use

Factors Analyzed:
    1. Loss Trend - Is loss improving vs previous checkpoints?
    2. Loss Value - Absolute loss compared to campaign floor
    3. Gradient Health - Were gradients healthy at save time?
    4. Memory Headroom - Was training close to OOM?
    5. Validation Metrics - If eval was run, how did it perform?
    6. Training Stability - Loss variance, no NaN/Inf

Usage:
    from temple.diagnostics import CheckpointHealthScorer

    scorer = CheckpointHealthScorer()

    # Score a single checkpoint
    grade = scorer.score_checkpoint(checkpoint_record)
    print(f"Grade: {grade.letter} ({grade.score}/100)")
    print(f"Issues: {grade.issues}")

    # Score all checkpoints in ledger
    from core.checkpoint_ledger import get_ledger
    ledger = get_ledger()
    grades = scorer.score_all(ledger)

    # Get top checkpoints
    best = scorer.get_top_checkpoints(ledger, n=5)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)


@dataclass
class CheckpointGrade:
    """Grade for a single checkpoint."""

    step: int
    checkpoint_name: str
    score: float  # 0-100
    letter: str  # A, B, C, D, F

    # Component scores (0-100 each)
    loss_trend_score: float = 100.0
    loss_value_score: float = 100.0
    gradient_score: float = 100.0
    memory_score: float = 100.0
    validation_score: float = 100.0
    stability_score: float = 100.0

    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    timestamp: Optional[str] = None
    train_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None

    @classmethod
    def from_score(
        cls,
        score: float,
        step: int,
        checkpoint_name: str,
        **kwargs,
    ) -> "CheckpointGrade":
        """Create grade from score."""
        letter = cls._score_to_letter(score)
        return cls(
            step=step,
            checkpoint_name=checkpoint_name,
            score=round(score, 1),
            letter=letter,
            **kwargs,
        )

    @staticmethod
    def _score_to_letter(score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "checkpoint_name": self.checkpoint_name,
            "score": self.score,
            "letter": self.letter,
            "components": {
                "loss_trend": self.loss_trend_score,
                "loss_value": self.loss_value_score,
                "gradient": self.gradient_score,
                "memory": self.memory_score,
                "validation": self.validation_score,
                "stability": self.stability_score,
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "train_loss": self.train_loss,
            "eval_accuracy": self.eval_accuracy,
        }

    def to_report_line(self) -> str:
        """Single line for reports."""
        emoji = {
            "A": "🌟",
            "B": "✅",
            "C": "⚠️",
            "D": "⚡",
            "F": "❌",
        }.get(self.letter, "❓")

        loss_str = f"loss={self.train_loss:.4f}" if self.train_loss else ""
        acc_str = f"acc={self.eval_accuracy:.1%}" if self.eval_accuracy else ""
        metrics = ", ".join(filter(None, [loss_str, acc_str]))

        return f"{emoji} {self.letter} ({self.score:5.1f}) step {self.step:,} - {metrics}"


class CheckpointHealthScorer:
    """
    Scores checkpoints based on multiple quality signals.

    Weights (configurable):
        - Loss trend: 25% - Is loss improving?
        - Loss value: 20% - How low is the loss?
        - Stability: 20% - Was training stable?
        - Validation: 20% - Eval performance
        - Gradient: 10% - Gradient health
        - Memory: 5% - Memory headroom
    """

    DEFAULT_WEIGHTS = {
        "loss_trend": 0.25,
        "loss_value": 0.20,
        "stability": 0.20,
        "validation": 0.20,
        "gradient": 0.10,
        "memory": 0.05,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        loss_floor: float = 0.5,  # Expected minimum achievable loss
        loss_ceiling: float = 5.0,  # Starting/bad loss
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.loss_floor = loss_floor
        self.loss_ceiling = loss_ceiling

        # Ensure weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score_checkpoint(
        self,
        record: Dict[str, Any],
        prev_record: Optional[Dict[str, Any]] = None,
        campaign_stats: Optional[Dict[str, Any]] = None,
    ) -> CheckpointGrade:
        """
        Score a single checkpoint record.

        Args:
            record: Checkpoint record from ledger
            prev_record: Previous checkpoint for trend analysis
            campaign_stats: Overall campaign statistics (min/max/avg loss)

        Returns:
            CheckpointGrade with score and breakdown
        """
        issues: List[str] = []
        warnings: List[str] = []

        step = record.get("step", 0)
        checkpoint_name = record.get("name", f"checkpoint-{step}")
        stats = record.get("stats", {})

        # Extract metrics
        train_loss = stats.get("train_loss")
        eval_accuracy = stats.get("eval_accuracy")
        grad_norm = stats.get("grad_norm")
        memory_usage = stats.get("memory_usage")
        timestamp = record.get("timestamp")

        # 1. Loss trend score (25%)
        loss_trend_score = self._score_loss_trend(
            train_loss, prev_record, issues, warnings
        )

        # 2. Loss value score (20%)
        loss_value_score = self._score_loss_value(
            train_loss, campaign_stats, issues, warnings
        )

        # 3. Stability score (20%)
        stability_score = self._score_stability(
            stats, issues, warnings
        )

        # 4. Validation score (20%)
        validation_score = self._score_validation(
            stats, issues, warnings
        )

        # 5. Gradient score (10%)
        gradient_score = self._score_gradient(
            stats, issues, warnings
        )

        # 6. Memory score (5%)
        memory_score = self._score_memory(
            stats, issues, warnings
        )

        # Calculate weighted total
        total_score = (
            loss_trend_score * self.weights["loss_trend"]
            + loss_value_score * self.weights["loss_value"]
            + stability_score * self.weights["stability"]
            + validation_score * self.weights["validation"]
            + gradient_score * self.weights["gradient"]
            + memory_score * self.weights["memory"]
        )

        # Critical issues can cap the score
        if any("critical" in issue.lower() for issue in issues):
            total_score = min(total_score, 59)  # Cap at F
        elif any("nan" in issue.lower() or "inf" in issue.lower() for issue in issues):
            total_score = min(total_score, 59)  # Cap at F

        return CheckpointGrade.from_score(
            score=total_score,
            step=step,
            checkpoint_name=checkpoint_name,
            loss_trend_score=loss_trend_score,
            loss_value_score=loss_value_score,
            gradient_score=gradient_score,
            memory_score=memory_score,
            validation_score=validation_score,
            stability_score=stability_score,
            issues=issues,
            warnings=warnings,
            timestamp=timestamp,
            train_loss=train_loss,
            eval_accuracy=eval_accuracy,
        )

    def _score_loss_trend(
        self,
        train_loss: Optional[float],
        prev_record: Optional[Dict[str, Any]],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on loss improvement vs previous checkpoint."""
        if train_loss is None:
            warnings.append("No training loss recorded")
            return 70.0  # Neutral score

        if prev_record is None:
            # First checkpoint, give benefit of doubt
            return 85.0

        prev_loss = prev_record.get("stats", {}).get("train_loss")
        if prev_loss is None:
            return 85.0

        # Calculate improvement
        if prev_loss > 0:
            improvement = (prev_loss - train_loss) / prev_loss
        else:
            improvement = 0

        if math.isnan(train_loss) or math.isinf(train_loss):
            issues.append("Loss is NaN/Inf")
            return 0.0

        if improvement > 0.1:
            # >10% improvement - excellent
            return 100.0
        elif improvement > 0.05:
            # 5-10% improvement - very good
            return 95.0
        elif improvement > 0.01:
            # 1-5% improvement - good
            return 85.0
        elif improvement > -0.01:
            # Flat (-1% to +1%) - okay
            return 75.0
        elif improvement > -0.05:
            # Slight regression (-1% to -5%)
            warnings.append(f"Loss increased {-improvement:.1%} vs previous")
            return 60.0
        elif improvement > -0.1:
            # Notable regression (-5% to -10%)
            issues.append(f"Loss regressed {-improvement:.1%} vs previous")
            return 40.0
        else:
            # Severe regression (>10%)
            issues.append(f"Loss spiked {-improvement:.1%} vs previous")
            return 20.0

    def _score_loss_value(
        self,
        train_loss: Optional[float],
        campaign_stats: Optional[Dict[str, Any]],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on absolute loss value."""
        if train_loss is None:
            return 70.0

        if math.isnan(train_loss) or math.isinf(train_loss):
            return 0.0

        # Get floor from campaign stats or use default
        if campaign_stats:
            floor = campaign_stats.get("min_loss", self.loss_floor)
            ceiling = campaign_stats.get("max_loss", self.loss_ceiling)
        else:
            floor = self.loss_floor
            ceiling = self.loss_ceiling

        # Normalize loss to 0-1 scale (lower is better)
        if ceiling > floor:
            normalized = (train_loss - floor) / (ceiling - floor)
            normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        else:
            normalized = 0.5

        # Convert to score (lower loss = higher score)
        score = 100 * (1 - normalized)

        # Additional penalties
        if train_loss > ceiling:
            issues.append(f"Loss {train_loss:.4f} exceeds expected ceiling {ceiling:.4f}")
            score = min(score, 40)

        return score

    def _score_stability(
        self,
        stats: Dict[str, Any],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on training stability indicators."""
        score = 100.0

        # Check for NaN in any metric
        for key, value in stats.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                issues.append(f"NaN/Inf detected in {key}")
                return 0.0

        # Check loss variance if available
        loss_variance = stats.get("loss_variance")
        if loss_variance is not None:
            if loss_variance > 1.0:
                issues.append(f"High loss variance: {loss_variance:.3f}")
                score -= 30
            elif loss_variance > 0.5:
                warnings.append(f"Moderate loss variance: {loss_variance:.3f}")
                score -= 15
            elif loss_variance > 0.2:
                score -= 5

        # Check if loss was oscillating
        loss_oscillation = stats.get("loss_oscillation", False)
        if loss_oscillation:
            warnings.append("Loss was oscillating at checkpoint time")
            score -= 20

        # Check for spikes
        had_spike = stats.get("had_loss_spike", False)
        if had_spike:
            warnings.append("Loss spike detected near checkpoint")
            score -= 15

        return max(0, score)

    def _score_validation(
        self,
        stats: Dict[str, Any],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on validation/eval metrics."""
        # Check for eval accuracy
        eval_accuracy = stats.get("eval_accuracy")
        if eval_accuracy is not None:
            if eval_accuracy >= 0.9:
                return 100.0
            elif eval_accuracy >= 0.8:
                return 90.0
            elif eval_accuracy >= 0.7:
                return 80.0
            elif eval_accuracy >= 0.6:
                warnings.append(f"Low eval accuracy: {eval_accuracy:.1%}")
                return 65.0
            elif eval_accuracy >= 0.5:
                warnings.append(f"Poor eval accuracy: {eval_accuracy:.1%}")
                return 50.0
            else:
                issues.append(f"Very low eval accuracy: {eval_accuracy:.1%}")
                return 30.0

        # Check for validation loss
        val_loss = stats.get("val_loss") or stats.get("eval_loss")
        train_loss = stats.get("train_loss")

        if val_loss is not None and train_loss is not None:
            # Check for overfitting (val_loss >> train_loss)
            if train_loss > 0:
                overfit_ratio = val_loss / train_loss
                if overfit_ratio > 2.0:
                    issues.append(f"Severe overfitting: val/train = {overfit_ratio:.2f}")
                    return 40.0
                elif overfit_ratio > 1.5:
                    warnings.append(f"Overfitting detected: val/train = {overfit_ratio:.2f}")
                    return 60.0
                elif overfit_ratio > 1.2:
                    return 75.0
                else:
                    return 90.0

        # No validation data - neutral score
        return 75.0

    def _score_gradient(
        self,
        stats: Dict[str, Any],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on gradient health."""
        grad_norm = stats.get("grad_norm")

        if grad_norm is None:
            return 80.0  # No data, slight penalty

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            issues.append("Gradient norm is NaN/Inf")
            return 0.0

        # Score based on gradient magnitude
        if grad_norm > 1000:
            issues.append(f"Gradient explosion: norm = {grad_norm:.2e}")
            return 10.0
        elif grad_norm > 100:
            warnings.append(f"High gradient norm: {grad_norm:.2f}")
            return 50.0
        elif grad_norm > 10:
            return 80.0
        elif grad_norm > 0.001:
            return 100.0  # Healthy range
        else:
            warnings.append(f"Very small gradients: {grad_norm:.2e}")
            return 60.0  # Vanishing gradients

    def _score_memory(
        self,
        stats: Dict[str, Any],
        issues: List[str],
        warnings: List[str],
    ) -> float:
        """Score based on memory usage."""
        memory_usage = stats.get("memory_usage")

        if memory_usage is None:
            return 85.0  # No data

        # Expect memory_usage as ratio 0-1
        if memory_usage > 1:
            # Might be in GB or MB, normalize
            memory_usage = memory_usage / 100 if memory_usage > 100 else memory_usage / 24

        if memory_usage > 0.95:
            issues.append(f"Critical memory usage: {memory_usage:.0%}")
            return 30.0
        elif memory_usage > 0.90:
            warnings.append(f"High memory usage: {memory_usage:.0%}")
            return 60.0
        elif memory_usage > 0.80:
            return 80.0
        else:
            return 100.0

    def score_all(
        self,
        ledger: Any,
        campaign_id: Optional[str] = None,
    ) -> List[CheckpointGrade]:
        """
        Score all checkpoints in a ledger.

        Args:
            ledger: CheckpointLedger instance
            campaign_id: Optional campaign to filter by

        Returns:
            List of CheckpointGrade sorted by step
        """
        # Get entries
        if hasattr(ledger, "get_entries"):
            entries = ledger.get_entries()
        elif hasattr(ledger, "entries"):
            entries = ledger.entries
        else:
            entries = {}

        # Filter by campaign if specified
        if campaign_id:
            entries = {
                k: v for k, v in entries.items()
                if v.get("campaign_id") == campaign_id
            }

        # Sort by step
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("step", 0)
        )

        # Calculate campaign stats
        losses = []
        for _, record in sorted_entries:
            loss = record.get("stats", {}).get("train_loss")
            if loss is not None and not math.isnan(loss):
                losses.append(loss)

        campaign_stats = None
        if losses:
            campaign_stats = {
                "min_loss": min(losses),
                "max_loss": max(losses),
                "avg_loss": sum(losses) / len(losses),
            }

        # Score each checkpoint
        grades = []
        prev_record = None

        for name, record in sorted_entries:
            record["name"] = name
            grade = self.score_checkpoint(
                record,
                prev_record=prev_record,
                campaign_stats=campaign_stats,
            )
            grades.append(grade)
            prev_record = record

        return grades

    def get_top_checkpoints(
        self,
        ledger: Any,
        n: int = 5,
        campaign_id: Optional[str] = None,
        min_grade: str = "C",
    ) -> List[CheckpointGrade]:
        """
        Get top N checkpoints by score.

        Args:
            ledger: CheckpointLedger instance
            n: Number of checkpoints to return
            campaign_id: Optional campaign filter
            min_grade: Minimum acceptable grade

        Returns:
            Top N CheckpointGrades sorted by score (descending)
        """
        all_grades = self.score_all(ledger, campaign_id)

        # Filter by minimum grade
        min_score = {"A": 90, "B": 80, "C": 70, "D": 60, "F": 0}[min_grade]
        filtered = [g for g in all_grades if g.score >= min_score]

        # Sort by score descending
        filtered.sort(key=lambda g: g.score, reverse=True)

        return filtered[:n]

    def get_grade_distribution(
        self,
        ledger: Any,
        campaign_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get count of checkpoints per grade."""
        grades = self.score_all(ledger, campaign_id)

        distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for grade in grades:
            distribution[grade.letter] += 1

        return distribution

    def generate_report(
        self,
        ledger: Any,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Generate a text report of checkpoint grades."""
        grades = self.score_all(ledger, campaign_id)

        if not grades:
            return "No checkpoints to grade."

        lines = [
            "=" * 60,
            "CHECKPOINT HEALTH REPORT",
            "=" * 60,
            "",
        ]

        # Distribution summary
        dist = self.get_grade_distribution(ledger, campaign_id)
        lines.append("Grade Distribution:")
        for letter in ["A", "B", "C", "D", "F"]:
            count = dist[letter]
            bar = "█" * count
            lines.append(f"  {letter}: {bar} ({count})")
        lines.append("")

        # Top 5
        lines.append("Top 5 Checkpoints:")
        top = self.get_top_checkpoints(ledger, n=5, campaign_id=campaign_id)
        for grade in top:
            lines.append(f"  {grade.to_report_line()}")
        lines.append("")

        # Bottom 5 (if there are F grades)
        f_grades = [g for g in grades if g.letter == "F"]
        if f_grades:
            lines.append("Failing Checkpoints (avoid these):")
            for grade in f_grades[:5]:
                lines.append(f"  {grade.to_report_line()}")
                for issue in grade.issues[:2]:
                    lines.append(f"    - {issue}")
            lines.append("")

        # Issues summary
        all_issues = []
        for grade in grades:
            for issue in grade.issues:
                all_issues.append(f"step {grade.step}: {issue}")

        if all_issues:
            lines.append("All Issues Found:")
            for issue in all_issues[:10]:
                lines.append(f"  - {issue}")
            if len(all_issues) > 10:
                lines.append(f"  ... and {len(all_issues) - 10} more")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def score_checkpoint_from_ledger(
    step: int,
    ledger_path: Optional[str] = None,
) -> Optional[CheckpointGrade]:
    """
    Quick helper to score a single checkpoint by step number.

    Usage:
        from temple.diagnostics.checkpoint_health import score_checkpoint_from_ledger
        grade = score_checkpoint_from_ledger(150000)
        print(f"Checkpoint at step 150000: {grade.letter}")
    """
    if ledger_path is None:
        ledger_path = "status/checkpoint_ledger.json"

    path = Path(ledger_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        entries = data.get("entries", {})

        # Find entry for this step
        target = None
        prev = None

        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("step", 0)
        )

        for name, entry in sorted_entries:
            if entry.get("step") == step:
                entry["name"] = name
                target = entry
                break
            prev = entry

        if target is None:
            return None

        scorer = CheckpointHealthScorer()
        return scorer.score_checkpoint(target, prev_record=prev)

    except Exception as e:
        logger.error(f"Error scoring checkpoint: {e}")
        return None
