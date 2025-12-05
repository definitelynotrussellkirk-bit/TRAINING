"""
Gradient Health Profiler - Per-Layer Gradient Flow Tracking
============================================================

Monitors gradient health across all layers to detect:
- Vanishing gradients (training stalls)
- Exploding gradients (NaN imminent)
- Diverging gradients (growing over time)
- Dead neurons (ReLU dying)

The key insight: Problems START in specific layers and PROPAGATE.
Finding the source layer is crucial for fixing the issue.

Usage:
    from temple.diagnostics import GradientHealthProfiler

    profiler = GradientHealthProfiler()

    # After each backward pass
    profiler.record(model, step=step)

    # Get diagnoses
    diagnoses = profiler.diagnose()
    for d in diagnoses:
        print(f"{d.layer}: {d.summary}")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from temple.diagnostics.severity import (
    Diagnosis,
    DiagnosisCategory,
    DiagnosticSeverity,
    DiagnosticThresholds,
    DEFAULT_THRESHOLDS,
)

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LayerGradientStats:
    """Statistics for gradients in a single layer."""
    layer_name: str
    norm_history: deque = field(default_factory=lambda: deque(maxlen=500))
    zero_grad_count: int = 0
    total_recordings: int = 0

    @property
    def avg_norm(self) -> float:
        if not self.norm_history:
            return 0.0
        return sum(self.norm_history) / len(self.norm_history)

    @property
    def recent_avg_norm(self) -> float:
        """Average of last 10 recordings."""
        if len(self.norm_history) < 10:
            return self.avg_norm
        recent = list(self.norm_history)[-10:]
        return sum(recent) / len(recent)

    @property
    def older_avg_norm(self) -> float:
        """Average of recordings 50-40 steps ago."""
        if len(self.norm_history) < 50:
            return self.avg_norm
        older = list(self.norm_history)[-50:-40]
        return sum(older) / len(older) if older else self.avg_norm

    @property
    def growth_rate(self) -> float:
        """How much gradients are growing (ratio of recent to older)."""
        older = self.older_avg_norm
        if older < 1e-10:
            return 0.0
        return self.recent_avg_norm / older

    @property
    def zero_grad_ratio(self) -> float:
        """Fraction of recordings with zero gradient."""
        if self.total_recordings == 0:
            return 0.0
        return self.zero_grad_count / self.total_recordings


class GradientHealthProfiler:
    """
    Profiles gradient health across all model layers.

    Tracks per-layer gradient norms over time to detect issues
    before they cause training failure.
    """

    def __init__(self, thresholds: Optional[DiagnosticThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.layer_stats: Dict[str, LayerGradientStats] = {}
        self.global_norm_history: deque = deque(maxlen=500)
        self.step_count: int = 0

    def record(
        self,
        model: "nn.Module",
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Record gradient norms for all layers.

        Args:
            model: The model after backward pass
            step: Current training step

        Returns:
            Dict mapping layer name to gradient norm
        """
        import torch

        self.step_count = step if step is not None else self.step_count + 1
        layer_norms = {}
        global_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.norm().item()
            layer_norms[name] = grad_norm
            global_norm += grad_norm ** 2

            # Update layer stats
            if name not in self.layer_stats:
                self.layer_stats[name] = LayerGradientStats(layer_name=name)

            stats = self.layer_stats[name]
            stats.norm_history.append(grad_norm)
            stats.total_recordings += 1

            if grad_norm < 1e-10:
                stats.zero_grad_count += 1

        global_norm = math.sqrt(global_norm)
        self.global_norm_history.append(global_norm)

        return layer_norms

    def diagnose(self) -> List[Diagnosis]:
        """
        Analyze gradient history and return diagnoses.

        Returns:
            List of Diagnosis objects for detected issues
        """
        diagnoses = []

        # Check each layer
        for name, stats in self.layer_stats.items():
            # Check for vanishing gradients
            if stats.recent_avg_norm < self.thresholds.gradient_vanishing_threshold:
                if stats.zero_grad_ratio > 0.5:
                    # More than 50% zero gradients = dead layer
                    diagnoses.append(self._diagnose_dead_layer(name, stats))
                else:
                    diagnoses.append(self._diagnose_vanishing(name, stats))

            # Check for exploding gradients
            elif stats.recent_avg_norm > self.thresholds.gradient_exploding_threshold:
                diagnoses.append(self._diagnose_exploding(name, stats))

            # Check for diverging gradients (growing over time)
            elif stats.growth_rate > self.thresholds.gradient_diverging_ratio:
                diagnoses.append(self._diagnose_diverging(name, stats))

        # Check global gradient health
        global_diagnosis = self._diagnose_global()
        if global_diagnosis:
            diagnoses.append(global_diagnosis)

        # Sort by severity (critical first)
        diagnoses.sort(key=lambda d: d.severity, reverse=True)

        return diagnoses

    def _diagnose_vanishing(self, layer: str, stats: LayerGradientStats) -> Diagnosis:
        """Diagnose vanishing gradient in a layer."""
        return Diagnosis(
            id=f"gradient_vanishing_{layer}",
            category=DiagnosisCategory.GRADIENT,
            severity=DiagnosticSeverity.WARN,
            summary=f"Vanishing gradient in {self._short_name(layer)}",
            details=(
                f"Layer `{layer}` has average gradient norm {stats.recent_avg_norm:.2e}, "
                f"which is below threshold {self.thresholds.gradient_vanishing_threshold:.2e}. "
                f"This layer is barely learning."
            ),
            remediation=(
                f"Options to fix vanishing gradients in `{self._short_name(layer)}`:\n"
                "1. Add skip connections / residual connections\n"
                "2. Use different activation (GELU instead of ReLU)\n"
                "3. Use better initialization (Xavier or Kaiming)\n"
                "4. Reduce model depth or width"
            ),
            evidence={
                "layer": layer,
                "avg_norm": stats.recent_avg_norm,
                "threshold": self.thresholds.gradient_vanishing_threshold,
                "zero_ratio": stats.zero_grad_ratio,
            },
            step=self.step_count,
            layer=layer,
        )

    def _diagnose_dead_layer(self, layer: str, stats: LayerGradientStats) -> Diagnosis:
        """Diagnose dead neurons/layer."""
        return Diagnosis(
            id=f"gradient_dead_{layer}",
            category=DiagnosisCategory.GRADIENT,
            severity=DiagnosticSeverity.ERROR,
            summary=f"Dead layer: {self._short_name(layer)}",
            details=(
                f"Layer `{layer}` has {stats.zero_grad_ratio:.0%} zero gradients. "
                "This layer is not learning at all - likely dead ReLU neurons."
            ),
            remediation=(
                f"Options to revive `{self._short_name(layer)}`:\n"
                "1. Use LeakyReLU or GELU instead of ReLU\n"
                "2. Add BatchNorm/LayerNorm before activation\n"
                "3. Use smaller initial learning rate\n"
                "4. Check for bug in forward pass (maybe layer is disconnected)"
            ),
            evidence={
                "layer": layer,
                "zero_ratio": stats.zero_grad_ratio,
                "avg_norm": stats.recent_avg_norm,
            },
            step=self.step_count,
            layer=layer,
        )

    def _diagnose_exploding(self, layer: str, stats: LayerGradientStats) -> Diagnosis:
        """Diagnose exploding gradient in a layer."""
        return Diagnosis(
            id=f"gradient_exploding_{layer}",
            category=DiagnosisCategory.GRADIENT,
            severity=DiagnosticSeverity.ERROR,
            summary=f"Exploding gradient in {self._short_name(layer)}",
            details=(
                f"Layer `{layer}` has average gradient norm {stats.recent_avg_norm:.2e}, "
                f"which exceeds threshold {self.thresholds.gradient_exploding_threshold:.2e}. "
                "This will likely cause NaN loss soon."
            ),
            remediation=(
                "Immediate actions:\n"
                "1. Enable gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`\n"
                "2. Reduce learning rate by 2-10x\n"
                "3. Add weight decay: `weight_decay=0.01`\n"
                f"4. Check layer `{self._short_name(layer)}` for initialization issues"
            ),
            evidence={
                "layer": layer,
                "avg_norm": stats.recent_avg_norm,
                "threshold": self.thresholds.gradient_exploding_threshold,
            },
            step=self.step_count,
            layer=layer,
            predicted_failure_in=self._predict_nan_steps(stats),
        )

    def _diagnose_diverging(self, layer: str, stats: LayerGradientStats) -> Diagnosis:
        """Diagnose diverging (growing) gradient in a layer."""
        return Diagnosis(
            id=f"gradient_diverging_{layer}",
            category=DiagnosisCategory.GRADIENT,
            severity=DiagnosticSeverity.WARN,
            summary=f"Diverging gradient in {self._short_name(layer)}",
            details=(
                f"Layer `{layer}` gradients are growing: {stats.growth_rate:.1f}x increase "
                f"over last 50 steps (recent: {stats.recent_avg_norm:.2e}, older: {stats.older_avg_norm:.2e}). "
                "If this continues, gradients will explode."
            ),
            remediation=(
                f"Gradients in `{self._short_name(layer)}` are diverging:\n"
                "1. Enable gradient clipping NOW: `max_grad_norm=1.0`\n"
                "2. Consider reducing learning rate\n"
                "3. Check for accumulating errors in this layer"
            ),
            evidence={
                "layer": layer,
                "growth_rate": stats.growth_rate,
                "recent_avg": stats.recent_avg_norm,
                "older_avg": stats.older_avg_norm,
            },
            step=self.step_count,
            layer=layer,
            predicted_failure_in=self._predict_divergence_steps(stats),
        )

    def _diagnose_global(self) -> Optional[Diagnosis]:
        """Check overall gradient health."""
        if len(self.global_norm_history) < 20:
            return None

        recent = list(self.global_norm_history)[-10:]
        older = list(self.global_norm_history)[-20:-10]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg

        # Check for global explosion
        if recent_avg > self.thresholds.gradient_exploding_threshold:
            return Diagnosis(
                id="gradient_global_explosion",
                category=DiagnosisCategory.GRADIENT,
                severity=DiagnosticSeverity.CRITICAL,
                summary="Global gradient explosion",
                details=(
                    f"Global gradient norm is {recent_avg:.2e}, "
                    f"exceeding threshold {self.thresholds.gradient_exploding_threshold:.2e}. "
                    "Training is unstable and likely to fail."
                ),
                remediation=(
                    "Immediate actions required:\n"
                    "1. ENABLE gradient clipping: `max_grad_norm=1.0`\n"
                    "2. REDUCE learning rate by 10x\n"
                    "3. Consider checkpointing and resuming with lower LR"
                ),
                evidence={
                    "global_norm": recent_avg,
                    "threshold": self.thresholds.gradient_exploding_threshold,
                },
                step=self.step_count,
            )

        # Check for global growth
        if older_avg > 1e-10 and recent_avg / older_avg > self.thresholds.gradient_diverging_ratio:
            return Diagnosis(
                id="gradient_global_diverging",
                category=DiagnosisCategory.GRADIENT,
                severity=DiagnosticSeverity.WARN,
                summary="Global gradient divergence",
                details=(
                    f"Global gradient norm is growing: {recent_avg / older_avg:.1f}x increase "
                    f"over last 20 steps. Current: {recent_avg:.2e}"
                ),
                remediation=(
                    "Training becoming unstable:\n"
                    "1. Add gradient clipping: `max_grad_norm=1.0`\n"
                    "2. Monitor closely for NaN"
                ),
                evidence={
                    "recent_avg": recent_avg,
                    "older_avg": older_avg,
                    "growth_rate": recent_avg / older_avg,
                },
                step=self.step_count,
            )

        return None

    def _predict_nan_steps(self, stats: LayerGradientStats) -> Optional[int]:
        """Predict how many steps until NaN based on gradient growth."""
        if stats.growth_rate <= 1.0:
            return None

        # Exponential growth model
        # If growth_rate = r per 50 steps, and we explode at 1e38,
        # steps = 50 * log_r(1e38 / current)
        current = stats.recent_avg_norm
        if current < 1e-10:
            return None

        threshold = 1e38  # Float overflow
        import math
        try:
            steps = 50 * math.log(threshold / current) / math.log(stats.growth_rate)
            return int(steps) if steps > 0 else 1
        except:
            return None

    def _predict_divergence_steps(self, stats: LayerGradientStats) -> Optional[int]:
        """Predict steps until explosion based on current divergence rate."""
        if stats.growth_rate <= 1.0:
            return None

        # How many 50-step windows until we hit explosion threshold?
        current = stats.recent_avg_norm
        threshold = self.thresholds.gradient_exploding_threshold

        if current >= threshold:
            return 0

        import math
        try:
            windows = math.log(threshold / current) / math.log(stats.growth_rate)
            return int(windows * 50)  # Convert to steps
        except:
            return None

    def _short_name(self, layer: str) -> str:
        """Shorten layer name for display."""
        parts = layer.split(".")
        if len(parts) <= 2:
            return layer
        return f"...{'.'.join(parts[-2:])}"

    def get_health_score(self) -> float:
        """
        Get overall gradient health score (0-1).

        1.0 = All layers healthy
        0.0 = Critical issues detected
        """
        if not self.layer_stats:
            return 1.0

        issues = 0
        total = len(self.layer_stats)

        for stats in self.layer_stats.values():
            if stats.recent_avg_norm < self.thresholds.gradient_vanishing_threshold:
                issues += 0.5  # Vanishing is bad but not critical
            elif stats.recent_avg_norm > self.thresholds.gradient_exploding_threshold:
                issues += 1.0  # Exploding is critical
            elif stats.growth_rate > self.thresholds.gradient_diverging_ratio:
                issues += 0.3  # Diverging is concerning

        if total == 0:
            return 1.0

        return max(0.0, 1.0 - (issues / total))

    def get_layer_report(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed per-layer gradient report."""
        report = {}
        for name, stats in self.layer_stats.items():
            report[name] = {
                "avg_norm": stats.avg_norm,
                "recent_avg_norm": stats.recent_avg_norm,
                "growth_rate": stats.growth_rate,
                "zero_grad_ratio": stats.zero_grad_ratio,
                "total_recordings": stats.total_recordings,
                "status": self._get_layer_status(stats),
            }
        return report

    def _get_layer_status(self, stats: LayerGradientStats) -> str:
        """Get status string for a layer."""
        if stats.recent_avg_norm < self.thresholds.gradient_vanishing_threshold:
            if stats.zero_grad_ratio > 0.5:
                return "dead"
            return "vanishing"
        elif stats.recent_avg_norm > self.thresholds.gradient_exploding_threshold:
            return "exploding"
        elif stats.growth_rate > self.thresholds.gradient_diverging_ratio:
            return "diverging"
        return "healthy"
