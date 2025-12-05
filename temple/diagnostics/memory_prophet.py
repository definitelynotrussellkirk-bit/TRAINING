"""
Memory Prophet - OOM Prediction and Leak Detection
===================================================

The Memory Prophet watches GPU memory usage and predicts:
- When OOM will occur (before it happens!)
- Memory leaks (gradual accumulation)
- Memory spikes (sudden increases)

The key insight: Memory problems are PREDICTABLE if you track the trend.

Usage:
    from temple.diagnostics import MemoryProphet

    prophet = MemoryProphet()

    # In training loop
    prophet.record(step=step)

    # Check for predictions
    diagnoses = prophet.diagnose()
    for d in diagnoses:
        if d.predicted_failure_in:
            print(f"WARNING: OOM in {d.predicted_failure_in} steps!")
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from temple.diagnostics.severity import (
    Diagnosis,
    DiagnosisCategory,
    DiagnosticSeverity,
    DiagnosticThresholds,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state."""
    step: int
    timestamp: str
    allocated_gb: float
    reserved_gb: float
    total_gb: float

    @property
    def usage_percent(self) -> float:
        if self.total_gb == 0:
            return 0.0
        return self.allocated_gb / self.total_gb

    @property
    def free_gb(self) -> float:
        return self.total_gb - self.allocated_gb


class MemoryProphet:
    """
    Predicts memory problems before they cause OOM.

    Tracks memory usage over time and uses trend analysis
    to predict when OOM will occur.
    """

    def __init__(
        self,
        thresholds: Optional[DiagnosticThresholds] = None,
        device: int = 0,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.device = device
        self.history: deque = deque(maxlen=1000)
        self.peak_allocated: float = 0.0
        self.baseline_allocated: Optional[float] = None
        self._cuda_available: Optional[bool] = None

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        if self._cuda_available is None:
            try:
                import torch
                self._cuda_available = torch.cuda.is_available()
            except ImportError:
                self._cuda_available = False
        return self._cuda_available

    def record(self, step: int = 0) -> Optional[MemorySnapshot]:
        """
        Record current memory state.

        Args:
            step: Current training step

        Returns:
            MemorySnapshot if CUDA available, None otherwise
        """
        if not self._check_cuda():
            return None

        import torch

        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9

            snapshot = MemorySnapshot(
                step=step,
                timestamp=datetime.now().isoformat(),
                allocated_gb=allocated,
                reserved_gb=reserved,
                total_gb=total,
            )

            self.history.append(snapshot)

            # Track peak
            if allocated > self.peak_allocated:
                self.peak_allocated = allocated

            # Set baseline from first few recordings
            if self.baseline_allocated is None and len(self.history) >= 10:
                first_ten = list(self.history)[:10]
                self.baseline_allocated = sum(s.allocated_gb for s in first_ten) / 10

            return snapshot

        except Exception as e:
            logger.debug(f"Failed to record memory: {e}")
            return None

    def diagnose(self) -> List[Diagnosis]:
        """
        Analyze memory history and return diagnoses.

        Returns:
            List of Diagnosis objects for detected issues
        """
        diagnoses = []

        if len(self.history) < 5:
            return diagnoses

        latest = self.history[-1]

        # Check current usage level
        usage_diagnosis = self._check_usage_level(latest)
        if usage_diagnosis:
            diagnoses.append(usage_diagnosis)

        # Check for memory leak
        leak_diagnosis = self._check_memory_leak()
        if leak_diagnosis:
            diagnoses.append(leak_diagnosis)

        # Check for memory spike
        spike_diagnosis = self._check_memory_spike()
        if spike_diagnosis:
            diagnoses.append(spike_diagnosis)

        # Predict OOM
        oom_prediction = self._predict_oom()
        if oom_prediction:
            diagnoses.append(oom_prediction)

        return diagnoses

    def _check_usage_level(self, snapshot: MemorySnapshot) -> Optional[Diagnosis]:
        """Check if current memory usage is concerning."""
        usage = snapshot.usage_percent

        if usage >= self.thresholds.memory_critical_percent:
            return Diagnosis(
                id="memory_critical",
                category=DiagnosisCategory.MEMORY,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"Critical memory usage: {usage:.0%}",
                details=(
                    f"GPU memory is at {snapshot.allocated_gb:.1f}GB / {snapshot.total_gb:.1f}GB ({usage:.0%}). "
                    f"Only {snapshot.free_gb:.2f}GB free. OOM is imminent."
                ),
                remediation=(
                    "Immediate actions to avoid OOM:\n"
                    "1. Reduce batch size (most effective)\n"
                    "2. Enable gradient checkpointing\n"
                    "3. Use mixed precision (fp16/bf16)\n"
                    "4. Reduce sequence length\n"
                    "5. Consider gradient accumulation with smaller batches"
                ),
                evidence={
                    "allocated_gb": snapshot.allocated_gb,
                    "total_gb": snapshot.total_gb,
                    "usage_percent": usage,
                    "free_gb": snapshot.free_gb,
                },
                step=snapshot.step,
            )

        elif usage >= self.thresholds.memory_warning_percent:
            return Diagnosis(
                id="memory_warning",
                category=DiagnosisCategory.MEMORY,
                severity=DiagnosticSeverity.WARN,
                summary=f"High memory usage: {usage:.0%}",
                details=(
                    f"GPU memory is at {snapshot.allocated_gb:.1f}GB / {snapshot.total_gb:.1f}GB ({usage:.0%}). "
                    f"Only {snapshot.free_gb:.2f}GB free. May OOM on larger batches."
                ),
                remediation=(
                    "Consider memory optimization:\n"
                    "1. Monitor for memory growth (could indicate leak)\n"
                    "2. Consider smaller batch size for safety margin\n"
                    "3. Enable gradient checkpointing if not already"
                ),
                evidence={
                    "allocated_gb": snapshot.allocated_gb,
                    "total_gb": snapshot.total_gb,
                    "usage_percent": usage,
                },
                step=snapshot.step,
            )

        return None

    def _check_memory_leak(self) -> Optional[Diagnosis]:
        """Check for memory leak (gradual increase over time)."""
        if len(self.history) < 100:
            return None

        # Compare recent to older memory usage
        recent = list(self.history)[-10:]
        older = list(self.history)[-100:-90]

        recent_avg = sum(s.allocated_gb for s in recent) / len(recent)
        older_avg = sum(s.allocated_gb for s in older) / len(older)

        growth = recent_avg - older_avg
        steps_elapsed = recent[-1].step - older[0].step if recent[-1].step != older[0].step else 90

        # MB per step
        growth_rate_mb = (growth * 1000) / max(steps_elapsed, 1)

        if growth_rate_mb > self.thresholds.memory_leak_threshold_mb:
            # Predict when we'll run out
            latest = self.history[-1]
            free_gb = latest.free_gb
            free_mb = free_gb * 1000
            steps_to_oom = int(free_mb / growth_rate_mb) if growth_rate_mb > 0 else None

            return Diagnosis(
                id="memory_leak",
                category=DiagnosisCategory.MEMORY,
                severity=DiagnosticSeverity.ERROR,
                summary=f"Memory leak: +{growth_rate_mb:.1f}MB/step",
                details=(
                    f"Memory is growing at {growth_rate_mb:.1f}MB per step. "
                    f"Increased from {older_avg:.2f}GB to {recent_avg:.2f}GB over {steps_elapsed} steps. "
                    f"Current free memory: {free_gb:.2f}GB."
                ),
                remediation=(
                    "Potential memory leak sources:\n"
                    "1. Check for tensors stored in lists (not garbage collected)\n"
                    "2. Ensure loss.backward() is called (releases graph)\n"
                    "3. Move metrics to CPU: `loss.item()` instead of storing tensor\n"
                    "4. Check custom callbacks for tensor accumulation\n"
                    "5. Call `torch.cuda.empty_cache()` periodically (last resort)"
                ),
                evidence={
                    "growth_rate_mb_per_step": growth_rate_mb,
                    "recent_avg_gb": recent_avg,
                    "older_avg_gb": older_avg,
                    "steps_elapsed": steps_elapsed,
                },
                step=latest.step,
                predicted_failure_in=steps_to_oom,
            )

        return None

    def _check_memory_spike(self) -> Optional[Diagnosis]:
        """Check for sudden memory spike."""
        if len(self.history) < 10:
            return None

        recent = list(self.history)[-10:]
        avg_before = sum(s.allocated_gb for s in recent[:-1]) / (len(recent) - 1)
        latest = recent[-1].allocated_gb

        spike_ratio = latest / avg_before if avg_before > 0 else 1.0

        if spike_ratio > 1.5:  # 50% increase
            return Diagnosis(
                id="memory_spike",
                category=DiagnosisCategory.MEMORY,
                severity=DiagnosticSeverity.WARN,
                summary=f"Memory spike: {spike_ratio:.1f}x increase",
                details=(
                    f"Memory jumped from {avg_before:.2f}GB to {latest:.2f}GB ({spike_ratio:.1f}x). "
                    "This could indicate a problematic batch or activation."
                ),
                remediation=(
                    "Memory spike causes:\n"
                    "1. Check batch size - did it change?\n"
                    "2. Check sequence length - longer sequences use more memory\n"
                    "3. Check if optimizer state was just initialized\n"
                    "4. Look for large intermediate activations"
                ),
                evidence={
                    "avg_before_gb": avg_before,
                    "current_gb": latest,
                    "spike_ratio": spike_ratio,
                },
                step=self.history[-1].step,
            )

        return None

    def _predict_oom(self) -> Optional[Diagnosis]:
        """Predict when OOM will occur based on trends."""
        if len(self.history) < 50:
            return None

        # Linear regression on memory usage
        snapshots = list(self.history)[-50:]
        steps = [s.step for s in snapshots]
        memory = [s.allocated_gb for s in snapshots]

        # Simple linear regression
        n = len(steps)
        sum_x = sum(steps)
        sum_y = sum(memory)
        sum_xy = sum(x * y for x, y in zip(steps, memory))
        sum_x2 = sum(x * x for x in steps)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Only predict if memory is growing
        if slope <= 0:
            return None

        # When will we hit total memory?
        latest = snapshots[-1]
        total = latest.total_gb
        current_step = latest.step

        # y = slope * x + intercept
        # total = slope * x_oom + intercept
        # x_oom = (total - intercept) / slope
        steps_to_oom = int((total - intercept) / slope) - current_step

        if steps_to_oom > 0 and steps_to_oom < 10000:  # Reasonable prediction window
            # Convert slope to MB/step for readability
            slope_mb = slope * 1000

            return Diagnosis(
                id="memory_oom_prediction",
                category=DiagnosisCategory.MEMORY,
                severity=DiagnosticSeverity.WARN if steps_to_oom > 1000 else DiagnosticSeverity.ERROR,
                summary=f"OOM predicted in ~{steps_to_oom} steps",
                details=(
                    f"Based on memory growth rate of {slope_mb:.2f}MB/step, "
                    f"GPU memory will be exhausted in approximately {steps_to_oom} steps. "
                    f"Current: {latest.allocated_gb:.2f}GB, Max: {total:.1f}GB."
                ),
                remediation=(
                    f"Actions before OOM (in ~{steps_to_oom} steps):\n"
                    "1. Reduce batch size or sequence length\n"
                    "2. Enable gradient checkpointing\n"
                    "3. Check for memory leak (tensors not being freed)\n"
                    "4. Consider checkpointing model before predicted OOM"
                ),
                evidence={
                    "slope_mb_per_step": slope_mb,
                    "steps_to_oom": steps_to_oom,
                    "current_gb": latest.allocated_gb,
                    "total_gb": total,
                },
                step=current_step,
                predicted_failure_in=steps_to_oom,
            )

        return None

    def get_health_score(self) -> float:
        """
        Get overall memory health score (0-1).

        1.0 = Plenty of free memory, no leaks
        0.0 = Critical memory usage or severe leak
        """
        if not self.history:
            return 1.0

        latest = self.history[-1]
        usage = latest.usage_percent

        # Base score from usage
        if usage >= self.thresholds.memory_critical_percent:
            base_score = 0.0
        elif usage >= self.thresholds.memory_warning_percent:
            # Linear interpolation between warning and critical
            base_score = (self.thresholds.memory_critical_percent - usage) / (
                self.thresholds.memory_critical_percent - self.thresholds.memory_warning_percent
            ) * 0.5
        else:
            # Linear interpolation between 0 and warning
            base_score = 0.5 + (self.thresholds.memory_warning_percent - usage) / self.thresholds.memory_warning_percent * 0.5

        # Penalize for leak
        if len(self.history) >= 100:
            recent = list(self.history)[-10:]
            older = list(self.history)[-100:-90]
            recent_avg = sum(s.allocated_gb for s in recent) / len(recent)
            older_avg = sum(s.allocated_gb for s in older) / len(older)
            growth = recent_avg - older_avg

            if growth > 0.1:  # Growing by more than 100MB
                leak_penalty = min(0.3, growth)  # Cap at 0.3
                base_score = max(0.0, base_score - leak_penalty)

        return base_score

    def get_summary(self) -> Dict[str, Any]:
        """Get memory status summary."""
        if not self.history:
            return {"status": "no_data"}

        latest = self.history[-1]

        return {
            "allocated_gb": round(latest.allocated_gb, 2),
            "reserved_gb": round(latest.reserved_gb, 2),
            "total_gb": round(latest.total_gb, 2),
            "usage_percent": round(latest.usage_percent * 100, 1),
            "free_gb": round(latest.free_gb, 2),
            "peak_gb": round(self.peak_allocated, 2),
            "health_score": round(self.get_health_score(), 2),
        }
