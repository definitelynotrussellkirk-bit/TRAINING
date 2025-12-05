"""
Unified Training Diagnostics - The Complete Diagnostic System
=============================================================

Brings together all diagnostic modules into a single, unified system
that can be hooked into training loops.

This is THE diagnostic tool everyone wishes they had:
- Real-time monitoring
- Predictive failure detection
- Root cause analysis
- Actionable remediation
- RPG-themed reporting

Usage:
    from temple.diagnostics import TrainingDiagnostics

    # Create unified diagnostics
    diagnostics = TrainingDiagnostics()

    # In training loop (after backward, before optimizer step)
    report = diagnostics.on_step(
        step=step,
        loss=loss,
        model=model,
        batch=batch,
        lr=optimizer.param_groups[0]['lr'],
    )

    # Check for issues
    if report.has_critical:
        print(f"STOP: {report.critical_issues}")
        # Save checkpoint and halt

    # Periodic full report
    if step % 100 == 0:
        print(diagnostics.get_full_report())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from temple.diagnostics.severity import (
    Diagnosis,
    DiagnosisCategory,
    DiagnosticSeverity,
    DiagnosticThresholds,
    DEFAULT_THRESHOLDS,
)
from temple.diagnostics.nan_detective import NaNDetective
from temple.diagnostics.gradient_health import GradientHealthProfiler
from temple.diagnostics.memory_prophet import MemoryProphet
from temple.diagnostics.lr_autopsy import LRAutopsy
from temple.diagnostics.data_sentinel import DataSentinel

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report from a training step.

    Contains all diagnoses, health scores, and predictions.
    """
    step: int
    timestamp: str
    diagnoses: List[Diagnosis] = field(default_factory=list)

    # Health scores (0-1, higher is better)
    overall_health: float = 1.0
    gradient_health: float = 1.0
    memory_health: float = 1.0
    lr_health: float = 1.0
    data_health: float = 1.0
    loss_health: float = 1.0

    # Predictions
    predicted_oom_steps: Optional[int] = None
    predicted_nan_steps: Optional[int] = None

    # Summary
    status: str = "healthy"  # "healthy", "warning", "error", "critical"
    summary: str = ""

    @property
    def has_critical(self) -> bool:
        return any(d.severity == DiagnosticSeverity.CRITICAL for d in self.diagnoses)

    @property
    def has_error(self) -> bool:
        return any(d.severity >= DiagnosticSeverity.ERROR for d in self.diagnoses)

    @property
    def has_warning(self) -> bool:
        return any(d.severity >= DiagnosticSeverity.WARN for d in self.diagnoses)

    @property
    def critical_issues(self) -> List[str]:
        return [d.summary for d in self.diagnoses if d.severity == DiagnosticSeverity.CRITICAL]

    @property
    def error_issues(self) -> List[str]:
        return [d.summary for d in self.diagnoses if d.severity == DiagnosticSeverity.ERROR]

    @property
    def warning_issues(self) -> List[str]:
        return [d.summary for d in self.diagnoses if d.severity == DiagnosticSeverity.WARN]

    def get_remediations(self) -> List[str]:
        """Get all remediation suggestions, sorted by severity."""
        sorted_diagnoses = sorted(self.diagnoses, key=lambda d: d.severity, reverse=True)
        return [d.remediation for d in sorted_diagnoses if d.remediation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "health": {
                "overall": self.overall_health,
                "gradient": self.gradient_health,
                "memory": self.memory_health,
                "lr": self.lr_health,
                "data": self.data_health,
                "loss": self.loss_health,
            },
            "predictions": {
                "oom_in_steps": self.predicted_oom_steps,
                "nan_in_steps": self.predicted_nan_steps,
            },
            "status": self.status,
            "summary": self.summary,
            "has_critical": self.has_critical,
            "critical_issues": self.critical_issues,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_rpg_report(self) -> str:
        """Generate RPG-themed report."""
        lines = []
        lines.append("=" * 60)
        lines.append("⚔️  TEMPLE DIAGNOSTIC REPORT  ⚔️")
        lines.append("=" * 60)
        lines.append(f"Step: {self.step}")
        lines.append(f"Time: {self.timestamp}")
        lines.append("")

        # Overall health bar
        health_bar = self._health_bar(self.overall_health)
        lines.append(f"Overall Health: {health_bar} ({self.overall_health:.0%})")
        lines.append("")

        # Individual health scores
        lines.append("Health Breakdown:")
        lines.append(f"  🌊 Energy Flow (Gradient): {self._health_bar(self.gradient_health, width=20)}")
        lines.append(f"  💾 Inventory (Memory):     {self._health_bar(self.memory_health, width=20)}")
        lines.append(f"  ⚡ Learning Pace (LR):     {self._health_bar(self.lr_health, width=20)}")
        lines.append(f"  📦 Quest Supplies (Data):  {self._health_bar(self.data_health, width=20)}")
        lines.append(f"  📉 Vital Signs (Loss):     {self._health_bar(self.loss_health, width=20)}")
        lines.append("")

        # Predictions
        if self.predicted_oom_steps or self.predicted_nan_steps:
            lines.append("⚠️ PROPHECIES:")
            if self.predicted_oom_steps:
                lines.append(f"  💾 Memory exhaustion in ~{self.predicted_oom_steps} steps")
            if self.predicted_nan_steps:
                lines.append(f"  📉 Corruption (NaN) in ~{self.predicted_nan_steps} steps")
            lines.append("")

        # Diagnoses by severity
        if self.diagnoses:
            lines.append("DIAGNOSES:")
            for severity in [DiagnosticSeverity.CRITICAL, DiagnosticSeverity.ERROR,
                           DiagnosticSeverity.WARN, DiagnosticSeverity.INFO]:
                issues = [d for d in self.diagnoses if d.severity == severity]
                if issues:
                    lines.append(f"\n{severity.icon} {severity.rpg_name}s:")
                    for d in issues:
                        lines.append(f"  • {d.summary}")
                        if severity >= DiagnosticSeverity.ERROR:
                            lines.append(f"    Fix: {d.remediation.split(chr(10))[0]}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _health_bar(self, value: float, width: int = 30) -> str:
        """Generate ASCII health bar."""
        filled = int(value * width)
        empty = width - filled

        if value >= 0.8:
            color = "🟢"
        elif value >= 0.5:
            color = "🟡"
        elif value >= 0.3:
            color = "🟠"
        else:
            color = "🔴"

        return f"{color} [{'█' * filled}{'░' * empty}]"


class TrainingDiagnostics:
    """
    Unified diagnostic system for training.

    Combines all diagnostic modules and provides a single interface
    for monitoring training health.
    """

    def __init__(
        self,
        thresholds: Optional[DiagnosticThresholds] = None,
        check_interval: int = 10,
        enable_memory: bool = True,
        enable_gradients: bool = True,
        enable_lr: bool = True,
        enable_data: bool = True,
        enable_nan: bool = True,
    ):
        """
        Initialize unified diagnostics.

        Args:
            thresholds: Custom diagnostic thresholds
            check_interval: How often to run full diagnostics (steps)
            enable_*: Enable/disable specific diagnostics
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.check_interval = check_interval

        # Initialize diagnostic modules
        self.nan_detective = NaNDetective(thresholds) if enable_nan else None
        self.gradient_profiler = GradientHealthProfiler(thresholds) if enable_gradients else None
        self.memory_prophet = MemoryProphet(thresholds) if enable_memory else None
        self.lr_autopsy = LRAutopsy(thresholds) if enable_lr else None
        self.data_sentinel = DataSentinel(thresholds) if enable_data else None

        # History
        self.reports: List[DiagnosticReport] = []
        self.step_count: int = 0
        self.last_full_check: int = 0

        # Callbacks
        self._on_critical: List[Callable[[DiagnosticReport], None]] = []
        self._on_prediction: List[Callable[[DiagnosticReport], None]] = []

    def on_critical(self, callback: Callable[[DiagnosticReport], None]):
        """Register callback for critical issues."""
        self._on_critical.append(callback)

    def on_prediction(self, callback: Callable[[DiagnosticReport], None]):
        """Register callback for failure predictions."""
        self._on_prediction.append(callback)

    def on_step(
        self,
        step: int,
        loss: "torch.Tensor | float",
        model: Optional["nn.Module"] = None,
        batch: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
        lr: float = 0.0,
        attention_mask: Optional["torch.Tensor"] = None,
        activations: Optional[Dict[str, "torch.Tensor"]] = None,
        force_full_check: bool = False,
    ) -> DiagnosticReport:
        """
        Run diagnostics for a training step.

        Args:
            step: Current training step
            loss: Loss value or tensor
            model: Model (for gradient inspection)
            batch: Input batch
            labels: Labels
            lr: Learning rate
            attention_mask: Attention mask
            activations: Dict of layer activations (optional)
            force_full_check: Force full diagnostic check

        Returns:
            DiagnosticReport with all findings
        """
        import torch

        self.step_count = step
        diagnoses = []
        health_scores = {}

        # Convert loss to float
        if isinstance(loss, torch.Tensor):
            loss_val = loss.item() if loss.numel() == 1 else float(loss.mean())
        else:
            loss_val = float(loss)

        # Always check NaN (critical)
        if self.nan_detective:
            nan_diag = self.nan_detective.investigate(
                loss=loss,
                model=model,
                batch=batch,
                lr=lr,
                step=step,
                activations=activations,
            )
            if nan_diag.severity >= DiagnosticSeverity.WARN:
                diagnoses.append(nan_diag)
            health_scores["loss"] = self.nan_detective.get_health_score()

        # Record data (every step, lightweight)
        if self.lr_autopsy:
            self.lr_autopsy.record(loss=loss_val, lr=lr, step=step)

        if self.memory_prophet:
            self.memory_prophet.record(step=step)

        # Full check at intervals or on force
        full_check = force_full_check or (step - self.last_full_check >= self.check_interval)

        if full_check:
            self.last_full_check = step

            # Gradient check
            if self.gradient_profiler and model is not None:
                self.gradient_profiler.record(model, step=step)
                grad_diagnoses = self.gradient_profiler.diagnose()
                diagnoses.extend(grad_diagnoses)
                health_scores["gradient"] = self.gradient_profiler.get_health_score()

            # Memory check
            if self.memory_prophet:
                memory_diagnoses = self.memory_prophet.diagnose()
                diagnoses.extend(memory_diagnoses)
                health_scores["memory"] = self.memory_prophet.get_health_score()

            # LR check
            if self.lr_autopsy:
                lr_diagnoses = self.lr_autopsy.diagnose()
                diagnoses.extend(lr_diagnoses)
                health_scores["lr"] = self.lr_autopsy.get_health_score()

            # Data check
            if self.data_sentinel and batch is not None:
                data_diagnoses = self.data_sentinel.check_batch(
                    batch=batch,
                    labels=labels,
                    attention_mask=attention_mask,
                    step=step,
                )
                diagnoses.extend(data_diagnoses)
                health_scores["data"] = self.data_sentinel.get_health_score()

        # Create report
        report = self._create_report(step, diagnoses, health_scores)

        # Store in history
        self.reports.append(report)
        if len(self.reports) > 1000:
            self.reports = self.reports[-500:]  # Keep last 500

        # Fire callbacks
        if report.has_critical:
            for callback in self._on_critical:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Critical callback error: {e}")

        if report.predicted_oom_steps or report.predicted_nan_steps:
            for callback in self._on_prediction:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Prediction callback error: {e}")

        return report

    def _create_report(
        self,
        step: int,
        diagnoses: List[Diagnosis],
        health_scores: Dict[str, float],
    ) -> DiagnosticReport:
        """Create a diagnostic report from findings."""

        # Extract predictions
        predicted_oom = None
        predicted_nan = None
        for d in diagnoses:
            if d.predicted_failure_in:
                if "oom" in d.id.lower() or "memory" in d.id.lower():
                    predicted_oom = d.predicted_failure_in
                elif "nan" in d.id.lower() or "gradient" in d.id.lower():
                    predicted_nan = d.predicted_failure_in

        # Calculate overall health
        scores = list(health_scores.values())
        overall = sum(scores) / len(scores) if scores else 1.0

        # Determine status
        if any(d.severity == DiagnosticSeverity.CRITICAL for d in diagnoses):
            status = "critical"
        elif any(d.severity == DiagnosticSeverity.ERROR for d in diagnoses):
            status = "error"
        elif any(d.severity == DiagnosticSeverity.WARN for d in diagnoses):
            status = "warning"
        else:
            status = "healthy"

        # Generate summary
        if status == "critical":
            critical_count = sum(1 for d in diagnoses if d.severity == DiagnosticSeverity.CRITICAL)
            summary = f"CRITICAL: {critical_count} critical issues detected - training at risk"
        elif status == "error":
            error_count = sum(1 for d in diagnoses if d.severity == DiagnosticSeverity.ERROR)
            summary = f"ERROR: {error_count} errors detected - training may fail soon"
        elif status == "warning":
            warn_count = sum(1 for d in diagnoses if d.severity == DiagnosticSeverity.WARN)
            summary = f"WARNING: {warn_count} warnings - training suboptimal"
        else:
            summary = "Training healthy"

        return DiagnosticReport(
            step=step,
            timestamp=datetime.now().isoformat(),
            diagnoses=diagnoses,
            overall_health=overall,
            gradient_health=health_scores.get("gradient", 1.0),
            memory_health=health_scores.get("memory", 1.0),
            lr_health=health_scores.get("lr", 1.0),
            data_health=health_scores.get("data", 1.0),
            loss_health=health_scores.get("loss", 1.0),
            predicted_oom_steps=predicted_oom,
            predicted_nan_steps=predicted_nan,
            status=status,
            summary=summary,
        )

    def get_current_health(self) -> Dict[str, float]:
        """Get current health scores."""
        health = {"overall": 1.0}

        if self.gradient_profiler:
            health["gradient"] = self.gradient_profiler.get_health_score()

        if self.memory_prophet:
            health["memory"] = self.memory_prophet.get_health_score()

        if self.lr_autopsy:
            health["lr"] = self.lr_autopsy.get_health_score()

        if self.data_sentinel:
            health["data"] = self.data_sentinel.get_health_score()

        if self.nan_detective:
            health["loss"] = self.nan_detective.get_health_score()

        scores = [v for k, v in health.items() if k != "overall"]
        health["overall"] = sum(scores) / len(scores) if scores else 1.0

        return health

    def get_latest_report(self) -> Optional[DiagnosticReport]:
        """Get most recent diagnostic report."""
        if self.reports:
            return self.reports[-1]
        return None

    def get_full_report(self) -> str:
        """Get full RPG-style diagnostic report."""
        if not self.reports:
            return "No diagnostic data yet"
        return self.reports[-1].to_rpg_report()

    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        health = self.get_current_health()
        latest = self.get_latest_report()

        return {
            "step": self.step_count,
            "health": health,
            "status": latest.status if latest else "unknown",
            "critical_count": len(latest.critical_issues) if latest else 0,
            "error_count": len(latest.error_issues) if latest else 0,
            "warning_count": len(latest.warning_issues) if latest else 0,
            "predictions": {
                "oom_in_steps": latest.predicted_oom_steps if latest else None,
                "nan_in_steps": latest.predicted_nan_steps if latest else None,
            },
            "memory": self.memory_prophet.get_summary() if self.memory_prophet else {},
            "lr": self.lr_autopsy.get_summary() if self.lr_autopsy else {},
        }

    def save_report(self, path: Path):
        """Save diagnostic history to file."""
        data = {
            "saved_at": datetime.now().isoformat(),
            "step_count": self.step_count,
            "reports": [r.to_dict() for r in self.reports[-100:]],  # Last 100
            "current_health": self.get_current_health(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Reset all diagnostic state."""
        if self.nan_detective:
            self.nan_detective = NaNDetective(self.thresholds)
        if self.gradient_profiler:
            self.gradient_profiler = GradientHealthProfiler(self.thresholds)
        if self.memory_prophet:
            self.memory_prophet = MemoryProphet(self.thresholds)
        if self.lr_autopsy:
            self.lr_autopsy = LRAutopsy(self.thresholds)
        if self.data_sentinel:
            self.data_sentinel = DataSentinel(self.thresholds)
        self.reports = []
        self.step_count = 0
        self.last_full_check = 0
