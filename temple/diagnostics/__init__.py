"""
Temple Diagnostics - The Diagnostic Tools Everyone Wishes They Had
===================================================================

Core diagnostic modules for training health analysis:

RUNTIME DIAGNOSTICS:
- NaN Detective: Root cause analysis for NaN/Inf losses
- Gradient Health: Per-layer gradient flow tracking
- Memory Prophet: OOM prediction and leak detection
- LR Autopsy: Learning rate health analysis
- Data Sentinel: Batch quality checking

PRE-FLIGHT & POST-HOC:
- Pre-flight Checks: Predict issues BEFORE training starts
- Checkpoint Health: A/B/C/D/F grades for checkpoints
- Config Validator: Validate config.json for mistakes
- Hang Detector: Detect training stalls (not just crashes)
- Historical Comparison: Compare to past campaigns

Usage:
    from temple.diagnostics import (
        # Runtime diagnostics
        TrainingDiagnostics,
        NaNDetective,
        GradientHealthProfiler,
        MemoryProphet,

        # Pre-flight
        run_preflight,
        ConfigValidator,

        # Post-hoc analysis
        CheckpointHealthScorer,
        HistoricalComparison,
        HangDetector,
    )

    # Create unified diagnostics
    diagnostics = TrainingDiagnostics()

    # In training loop
    report = diagnostics.on_step(
        step=step,
        loss=loss,
        model=model,
        batch=batch,
        lr=optimizer.param_groups[0]['lr'],
    )

    if report.has_critical:
        print(f"STOP TRAINING: {report.critical_issues}")
"""

# Core types
from temple.diagnostics.severity import DiagnosticSeverity, Diagnosis, DiagnosisCategory

# Runtime diagnostics
from temple.diagnostics.nan_detective import NaNDetective
from temple.diagnostics.gradient_health import GradientHealthProfiler
from temple.diagnostics.memory_prophet import MemoryProphet
from temple.diagnostics.lr_autopsy import LRAutopsy
from temple.diagnostics.data_sentinel import DataSentinel
from temple.diagnostics.unified import TrainingDiagnostics, DiagnosticReport

# Pre-flight checks
from temple.diagnostics.preflight import run_preflight, PreflightReport

# Checkpoint health scoring
from temple.diagnostics.checkpoint_health import CheckpointHealthScorer, CheckpointGrade

# Hang detection
from temple.diagnostics.hang_detector import HangDetector, HangStatus, check_training_hung

# Historical comparison
from temple.diagnostics.historical import HistoricalComparison, ComparisonResult

# Config validation
from temple.diagnostics.config_validator import ConfigValidator, ValidationResult

# Inference health (deep checks)
from temple.diagnostics.inference_health import (
    InferenceHealthChecker,
    InferenceHealthReport,
    InferenceIssue,
    FixGenerator,
    check_inference_health,
    quick_inference_test,
)

__all__ = [
    # Core types
    "DiagnosticSeverity",
    "Diagnosis",
    "DiagnosisCategory",

    # Runtime diagnostics
    "NaNDetective",
    "GradientHealthProfiler",
    "MemoryProphet",
    "LRAutopsy",
    "DataSentinel",
    "TrainingDiagnostics",
    "DiagnosticReport",

    # Pre-flight checks
    "run_preflight",
    "PreflightReport",

    # Checkpoint health
    "CheckpointHealthScorer",
    "CheckpointGrade",

    # Hang detection
    "HangDetector",
    "HangStatus",
    "check_training_hung",

    # Historical comparison
    "HistoricalComparison",
    "ComparisonResult",

    # Config validation
    "ConfigValidator",
    "ValidationResult",

    # Inference health
    "InferenceHealthChecker",
    "InferenceHealthReport",
    "InferenceIssue",
    "FixGenerator",
    "check_inference_health",
    "quick_inference_test",
]
