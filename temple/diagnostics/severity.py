"""
Diagnostic Severity Levels and Core Types
==========================================

Defines severity levels, diagnosis types, and categories used
throughout the Temple diagnostic system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic findings."""
    INFO = "info"           # FYI, no action needed
    WARN = "warn"           # Suboptimal but training can continue
    ERROR = "error"         # Will likely cause failure soon
    CRITICAL = "critical"   # Stop training NOW

    @property
    def icon(self) -> str:
        """Emoji icon for this severity."""
        return {
            DiagnosticSeverity.INFO: "ℹ️",
            DiagnosticSeverity.WARN: "⚠️",
            DiagnosticSeverity.ERROR: "❌",
            DiagnosticSeverity.CRITICAL: "🚨",
        }[self]

    @property
    def rpg_name(self) -> str:
        """RPG-themed name for this severity."""
        return {
            DiagnosticSeverity.INFO: "Whisper",
            DiagnosticSeverity.WARN: "Warning Sign",
            DiagnosticSeverity.ERROR: "Wound",
            DiagnosticSeverity.CRITICAL: "Mortal Blow",
        }[self]

    def _order(self) -> int:
        """Get ordering index for comparisons."""
        order = [DiagnosticSeverity.INFO, DiagnosticSeverity.WARN,
                 DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL]
        return order.index(self)

    def __lt__(self, other: "DiagnosticSeverity") -> bool:
        if not isinstance(other, DiagnosticSeverity):
            return NotImplemented
        return self._order() < other._order()

    def __le__(self, other: "DiagnosticSeverity") -> bool:
        if not isinstance(other, DiagnosticSeverity):
            return NotImplemented
        return self._order() <= other._order()

    def __gt__(self, other: "DiagnosticSeverity") -> bool:
        if not isinstance(other, DiagnosticSeverity):
            return NotImplemented
        return self._order() > other._order()

    def __ge__(self, other: "DiagnosticSeverity") -> bool:
        if not isinstance(other, DiagnosticSeverity):
            return NotImplemented
        return self._order() >= other._order()


class DiagnosisCategory(Enum):
    """Categories of diagnostic findings."""
    LOSS = "loss"               # NaN, Inf, divergence
    GRADIENT = "gradient"       # Vanishing, exploding
    MEMORY = "memory"           # OOM, leaks
    LEARNING_RATE = "lr"        # Too high, too low
    DATA = "data"               # Bad batches, quality issues
    CONVERGENCE = "convergence" # Plateaus, divergence
    HARDWARE = "hardware"       # GPU errors, temperature
    SYSTEM = "system"           # Disk, network, etc.

    @property
    def icon(self) -> str:
        """Emoji icon for this category."""
        return {
            DiagnosisCategory.LOSS: "📉",
            DiagnosisCategory.GRADIENT: "🌊",
            DiagnosisCategory.MEMORY: "💾",
            DiagnosisCategory.LEARNING_RATE: "⚡",
            DiagnosisCategory.DATA: "📦",
            DiagnosisCategory.CONVERGENCE: "🎯",
            DiagnosisCategory.HARDWARE: "🔥",
            DiagnosisCategory.SYSTEM: "🖥️",
        }[self]

    @property
    def rpg_name(self) -> str:
        """RPG-themed name for this category."""
        return {
            DiagnosisCategory.LOSS: "Vital Signs",
            DiagnosisCategory.GRADIENT: "Energy Flow",
            DiagnosisCategory.MEMORY: "Inventory",
            DiagnosisCategory.LEARNING_RATE: "Learning Pace",
            DiagnosisCategory.DATA: "Quest Supplies",
            DiagnosisCategory.CONVERGENCE: "Path Progress",
            DiagnosisCategory.HARDWARE: "Forge Heat",
            DiagnosisCategory.SYSTEM: "Realm Infrastructure",
        }[self]


@dataclass
class Diagnosis:
    """
    A single diagnostic finding.

    Represents one issue detected during training analysis,
    with severity, description, and suggested remediation.
    """
    id: str                                 # Unique identifier (e.g., "nan_gradient_explosion")
    category: DiagnosisCategory             # What type of issue
    severity: DiagnosticSeverity            # How bad is it
    summary: str                            # One-line summary
    details: str                            # Full explanation
    remediation: str                        # How to fix it
    evidence: Dict[str, Any] = field(default_factory=dict)  # Supporting data
    timestamp: str = ""                     # When detected
    step: Optional[int] = None              # Training step
    layer: Optional[str] = None             # Affected layer (if applicable)
    predicted_failure_in: Optional[int] = None  # Steps until failure (if predictable)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def icon(self) -> str:
        """Combined icon for category and severity."""
        return f"{self.category.icon}{self.severity.icon}"

    @property
    def is_critical(self) -> bool:
        return self.severity == DiagnosticSeverity.CRITICAL

    @property
    def is_error(self) -> bool:
        return self.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)

    @property
    def rpg_summary(self) -> str:
        """RPG-themed summary."""
        severity_name = self.severity.rpg_name
        category_name = self.category.rpg_name
        return f"[{severity_name}] {category_name}: {self.summary}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "summary": self.summary,
            "details": self.details,
            "remediation": self.remediation,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "step": self.step,
            "layer": self.layer,
            "predicted_failure_in": self.predicted_failure_in,
            "icon": self.icon,
            "is_critical": self.is_critical,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Diagnosis":
        return cls(
            id=data["id"],
            category=DiagnosisCategory(data["category"]),
            severity=DiagnosticSeverity(data["severity"]),
            summary=data["summary"],
            details=data["details"],
            remediation=data["remediation"],
            evidence=data.get("evidence", {}),
            timestamp=data.get("timestamp", ""),
            step=data.get("step"),
            layer=data.get("layer"),
            predicted_failure_in=data.get("predicted_failure_in"),
        )


@dataclass
class DiagnosticThresholds:
    """Configurable thresholds for diagnostic checks."""

    # Gradient thresholds
    gradient_vanishing_threshold: float = 1e-7
    gradient_exploding_threshold: float = 1e3
    gradient_diverging_ratio: float = 10.0  # 10x growth = diverging

    # Memory thresholds
    memory_warning_percent: float = 0.80    # 80% = warning
    memory_critical_percent: float = 0.90   # 90% = critical
    memory_leak_threshold_mb: float = 10.0  # 10 MB/step = leak

    # Learning rate thresholds
    lr_oscillation_threshold: float = 0.5   # std > 0.5 * mean = oscillating
    lr_plateau_threshold: float = 0.001     # < 0.1% change = plateau
    lr_divergence_threshold: float = 1.1    # 10% increase = diverging

    # Data thresholds
    data_max_value: float = 1e4             # Values > 10k are suspicious
    data_min_variance: float = 1e-6         # Variance < 1e-6 = dead

    # Loss thresholds
    loss_spike_threshold: float = 10.0      # 10x increase = spike

    def to_dict(self) -> Dict[str, float]:
        return {
            "gradient_vanishing_threshold": self.gradient_vanishing_threshold,
            "gradient_exploding_threshold": self.gradient_exploding_threshold,
            "gradient_diverging_ratio": self.gradient_diverging_ratio,
            "memory_warning_percent": self.memory_warning_percent,
            "memory_critical_percent": self.memory_critical_percent,
            "memory_leak_threshold_mb": self.memory_leak_threshold_mb,
            "lr_oscillation_threshold": self.lr_oscillation_threshold,
            "lr_plateau_threshold": self.lr_plateau_threshold,
            "lr_divergence_threshold": self.lr_divergence_threshold,
            "data_max_value": self.data_max_value,
            "data_min_variance": self.data_min_variance,
            "loss_spike_threshold": self.loss_spike_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DiagnosticThresholds":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# Default thresholds
DEFAULT_THRESHOLDS = DiagnosticThresholds()
