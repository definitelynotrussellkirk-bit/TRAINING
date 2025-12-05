"""
LR Autopsy - Learning Rate Health Analysis
==========================================

The LR Autopsy analyzes loss behavior to diagnose learning rate issues:
- Too high: oscillation, divergence, NaN
- Too low: plateau, slow convergence
- Just right: steady decrease with acceptable variance

The key insight: Loss behavior tells you everything about LR health.

Usage:
    from temple.diagnostics import LRAutopsy

    autopsy = LRAutopsy()

    # Record loss and LR
    autopsy.record(loss=loss, lr=lr, step=step)

    # Get diagnosis
    diagnoses = autopsy.diagnose()
    for d in diagnoses:
        print(f"LR issue: {d.summary}")
        print(f"Suggested: {d.remediation}")
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from temple.diagnostics.severity import (
    Diagnosis,
    DiagnosisCategory,
    DiagnosticSeverity,
    DiagnosticThresholds,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class LossSnapshot:
    """Snapshot of loss and learning rate."""
    step: int
    loss: float
    lr: float


class LRAutopsy:
    """
    Analyzes learning rate health from loss behavior.

    Uses statistical analysis of loss trends to diagnose LR issues
    and suggest corrections.
    """

    def __init__(self, thresholds: Optional[DiagnosticThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.history: deque = deque(maxlen=2000)
        self.lr_change_points: List[Tuple[int, float, float]] = []  # (step, old_lr, new_lr)

    def record(self, loss: float, lr: float, step: int):
        """
        Record loss and learning rate.

        Args:
            loss: Current loss value
            lr: Current learning rate
            step: Current training step
        """
        if math.isnan(loss) or math.isinf(loss):
            return  # Don't record invalid losses

        snapshot = LossSnapshot(step=step, loss=loss, lr=lr)
        self.history.append(snapshot)

        # Track LR changes
        if len(self.history) >= 2:
            prev = self.history[-2]
            if abs(prev.lr - lr) > 1e-10:
                self.lr_change_points.append((step, prev.lr, lr))

    def diagnose(self) -> List[Diagnosis]:
        """
        Analyze loss history and return LR diagnoses.

        Returns:
            List of Diagnosis objects for detected LR issues
        """
        diagnoses = []

        if len(self.history) < 50:
            return diagnoses

        # Check for oscillation (LR too high)
        oscillation_diagnosis = self._check_oscillation()
        if oscillation_diagnosis:
            diagnoses.append(oscillation_diagnosis)

        # Check for divergence (LR way too high)
        divergence_diagnosis = self._check_divergence()
        if divergence_diagnosis:
            diagnoses.append(divergence_diagnosis)

        # Check for plateau (LR too low or converged)
        plateau_diagnosis = self._check_plateau()
        if plateau_diagnosis:
            diagnoses.append(plateau_diagnosis)

        # Check for slow convergence
        slow_diagnosis = self._check_slow_convergence()
        if slow_diagnosis:
            diagnoses.append(slow_diagnosis)

        return diagnoses

    def _check_oscillation(self) -> Optional[Diagnosis]:
        """Check if loss is oscillating (LR too high)."""
        if len(self.history) < 100:
            return None

        recent = list(self.history)[-100:]
        losses = [s.loss for s in recent]
        lr = recent[-1].lr

        # Calculate statistics
        mean = sum(losses) / len(losses)
        variance = sum((l - mean) ** 2 for l in losses) / len(losses)
        std = math.sqrt(variance)

        # Coefficient of variation
        cv = std / mean if mean > 0 else 0

        if cv > self.thresholds.lr_oscillation_threshold:
            # Also check for sign changes in loss delta (oscillation pattern)
            deltas = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
            sign_changes = sum(1 for i in range(len(deltas)-1)
                             if deltas[i] * deltas[i+1] < 0)
            sign_change_ratio = sign_changes / len(deltas)

            if sign_change_ratio > 0.4:  # More than 40% sign changes = oscillating
                return Diagnosis(
                    id="lr_oscillation",
                    category=DiagnosisCategory.LEARNING_RATE,
                    severity=DiagnosticSeverity.WARN,
                    summary=f"Loss oscillating (CV={cv:.2f})",
                    details=(
                        f"Loss has high variance relative to mean (CV={cv:.2f}, std={std:.4f}, mean={mean:.4f}). "
                        f"Direction changes {sign_change_ratio:.0%} of the time. "
                        f"Current LR: {lr:.2e}. This typically indicates LR is too high."
                    ),
                    remediation=(
                        f"Reduce learning rate:\n"
                        f"1. Try LR = {lr * 0.5:.2e} (50% reduction)\n"
                        f"2. If still oscillating, try LR = {lr * 0.3:.2e} (70% reduction)\n"
                        f"3. Consider using learning rate warmup\n"
                        f"4. Enable gradient clipping: max_grad_norm=1.0"
                    ),
                    evidence={
                        "cv": cv,
                        "std": std,
                        "mean": mean,
                        "current_lr": lr,
                        "suggested_lr": lr * 0.5,
                        "sign_change_ratio": sign_change_ratio,
                    },
                    step=recent[-1].step,
                )

        return None

    def _check_divergence(self) -> Optional[Diagnosis]:
        """Check if loss is diverging (getting worse)."""
        if len(self.history) < 100:
            return None

        recent = list(self.history)[-50:]
        older = list(self.history)[-100:-50]

        recent_avg = sum(s.loss for s in recent) / len(recent)
        older_avg = sum(s.loss for s in older) / len(older)
        lr = recent[-1].lr

        ratio = recent_avg / older_avg if older_avg > 0 else 1.0

        if ratio > self.thresholds.lr_divergence_threshold:
            return Diagnosis(
                id="lr_divergence",
                category=DiagnosisCategory.LEARNING_RATE,
                severity=DiagnosticSeverity.ERROR,
                summary=f"Loss diverging ({ratio:.1f}x increase)",
                details=(
                    f"Loss increased from {older_avg:.4f} to {recent_avg:.4f} ({ratio:.1f}x). "
                    f"Current LR: {lr:.2e}. Training is moving away from optimum."
                ),
                remediation=(
                    f"Learning rate is too high:\n"
                    f"1. REDUCE LR significantly: {lr:.2e} → {lr * 0.1:.2e}\n"
                    f"2. Consider resuming from earlier checkpoint\n"
                    f"3. Enable gradient clipping if not already\n"
                    f"4. Check data for anomalies (bad batch?)"
                ),
                evidence={
                    "recent_avg": recent_avg,
                    "older_avg": older_avg,
                    "ratio": ratio,
                    "current_lr": lr,
                    "suggested_lr": lr * 0.1,
                },
                step=recent[-1].step,
            )

        return None

    def _check_plateau(self) -> Optional[Diagnosis]:
        """Check if loss has plateaued."""
        if len(self.history) < 200:
            return None

        recent = list(self.history)[-100:]
        older = list(self.history)[-200:-100]

        recent_avg = sum(s.loss for s in recent) / len(recent)
        older_avg = sum(s.loss for s in older) / len(older)
        lr = recent[-1].lr

        change = abs(recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        if change < self.thresholds.lr_plateau_threshold:
            # Check variance - low variance + no change = true plateau
            variance = sum((s.loss - recent_avg) ** 2 for s in recent) / len(recent)
            std = math.sqrt(variance)
            cv = std / recent_avg if recent_avg > 0 else 0

            if cv < 0.1:  # Very stable = converged or stuck
                return Diagnosis(
                    id="lr_plateau",
                    category=DiagnosisCategory.CONVERGENCE,
                    severity=DiagnosticSeverity.WARN,
                    summary=f"Loss plateaued at {recent_avg:.4f}",
                    details=(
                        f"Loss has not changed significantly over 200 steps "
                        f"(change: {change:.2%}, CV: {cv:.2%}). "
                        f"Current LR: {lr:.2e}. Training may have converged or gotten stuck."
                    ),
                    remediation=(
                        f"Options for plateau:\n"
                        f"1. If good accuracy, training may be complete!\n"
                        f"2. Try increasing LR: {lr:.2e} → {lr * 3:.2e} to escape local minimum\n"
                        f"3. Add learning rate warmup restart (cosine annealing with restarts)\n"
                        f"4. Check if data variety is sufficient"
                    ),
                    evidence={
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "change_percent": change,
                        "cv": cv,
                        "current_lr": lr,
                        "suggested_lr": lr * 3,
                    },
                    step=recent[-1].step,
                )

        return None

    def _check_slow_convergence(self) -> Optional[Diagnosis]:
        """Check if convergence is slower than expected."""
        if len(self.history) < 500:
            return None

        # Calculate convergence rate over different windows
        windows = [100, 200, 500]
        rates = []

        for window in windows:
            if len(self.history) >= window:
                recent = list(self.history)[-window//2:]
                older = list(self.history)[-window:-window//2]

                recent_avg = sum(s.loss for s in recent) / len(recent)
                older_avg = sum(s.loss for s in older) / len(older)

                # Convergence rate = decrease per step
                steps = window // 2
                rate = (older_avg - recent_avg) / steps if steps > 0 else 0
                rates.append(rate)

        # If all rates are very small but positive
        lr = self.history[-1].lr
        if all(r > 0 and r < 1e-6 for r in rates):
            return Diagnosis(
                id="lr_slow_convergence",
                category=DiagnosisCategory.LEARNING_RATE,
                severity=DiagnosticSeverity.INFO,
                summary="Slow convergence detected",
                details=(
                    f"Loss is decreasing very slowly. "
                    f"Convergence rates: {[f'{r:.2e}' for r in rates]}. "
                    f"Current LR: {lr:.2e}. Training will take a long time at this pace."
                ),
                remediation=(
                    f"Options to speed up training:\n"
                    f"1. Increase LR: {lr:.2e} → {lr * 2:.2e}\n"
                    f"2. This is fine if you have the compute budget\n"
                    f"3. Consider using OneCycleLR for faster convergence"
                ),
                evidence={
                    "convergence_rates": rates,
                    "current_lr": lr,
                },
                step=self.history[-1].step,
            )

        return None

    def get_health_score(self) -> float:
        """
        Get learning rate health score (0-1).

        1.0 = Steady convergence, no issues
        0.0 = Diverging or severe oscillation
        """
        if len(self.history) < 50:
            return 1.0

        diagnoses = self.diagnose()

        if not diagnoses:
            return 1.0

        # Worst diagnosis determines score
        worst_severity = max(d.severity for d in diagnoses)

        return {
            DiagnosticSeverity.INFO: 0.9,
            DiagnosticSeverity.WARN: 0.6,
            DiagnosticSeverity.ERROR: 0.3,
            DiagnosticSeverity.CRITICAL: 0.0,
        }.get(worst_severity, 1.0)

    def get_suggested_lr(self) -> Optional[float]:
        """Get suggested learning rate based on current behavior."""
        diagnoses = self.diagnose()

        for d in diagnoses:
            if "suggested_lr" in d.evidence:
                return d.evidence["suggested_lr"]

        return None

    def get_convergence_rate(self, window: int = 100) -> float:
        """Get current convergence rate (loss decrease per step)."""
        if len(self.history) < window:
            return 0.0

        recent = list(self.history)[-window//2:]
        older = list(self.history)[-window:-window//2]

        recent_avg = sum(s.loss for s in recent) / len(recent)
        older_avg = sum(s.loss for s in older) / len(older)

        steps = window // 2
        return (older_avg - recent_avg) / steps if steps > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get LR health summary."""
        if len(self.history) < 10:
            return {"status": "insufficient_data"}

        recent = list(self.history)[-50:]
        losses = [s.loss for s in recent]
        lr = recent[-1].lr

        mean = sum(losses) / len(losses)
        variance = sum((l - mean) ** 2 for l in losses) / len(losses)
        std = math.sqrt(variance)
        cv = std / mean if mean > 0 else 0

        return {
            "current_lr": lr,
            "recent_loss_mean": round(mean, 6),
            "recent_loss_std": round(std, 6),
            "coefficient_of_variation": round(cv, 4),
            "convergence_rate": round(self.get_convergence_rate(), 8),
            "health_score": round(self.get_health_score(), 2),
            "suggested_lr": self.get_suggested_lr(),
        }
