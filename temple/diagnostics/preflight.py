"""
Pre-flight Checks - Predict Issues BEFORE Training Starts
==========================================================

The most valuable diagnostic is one that catches problems before they happen.
Pre-flight checks analyze your config and environment to predict:

- VRAM usage (will you OOM?)
- Path validity (do models/data exist?)
- Config sanity (is LR reasonable?)
- GPU availability
- Common mistakes

Usage:
    from temple.diagnostics.preflight import run_preflight, PreflightReport

    report = run_preflight(config_path="config.json")
    if not report.ready_to_train:
        print(f"Cannot start training: {report.blockers}")
    else:
        print(f"All checks passed! Predicted VRAM: {report.predicted_vram_gb}GB")

    # Or run from CLI:
    # python -m temple.diagnostics.preflight --config config.json
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from temple.diagnostics.severity import Diagnosis, DiagnosisCategory, DiagnosticSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# VRAM ESTIMATION
# =============================================================================

# Model size estimates (in GB) for common architectures
MODEL_VRAM_ESTIMATES = {
    # Qwen family
    "qwen3-0.6b": {"params": 0.6, "base_vram": 1.5, "per_batch_vram": 0.3},
    "qwen3-1.7b": {"params": 1.7, "base_vram": 4.0, "per_batch_vram": 0.5},
    "qwen3-4b": {"params": 4.0, "base_vram": 9.0, "per_batch_vram": 1.0},
    "qwen3-8b": {"params": 8.0, "base_vram": 17.0, "per_batch_vram": 2.0},
    "qwen2.5-3b": {"params": 3.0, "base_vram": 7.0, "per_batch_vram": 0.8},
    "qwen2.5-7b": {"params": 7.0, "base_vram": 15.0, "per_batch_vram": 1.5},
    # Mistral family
    "mistral-7b": {"params": 7.0, "base_vram": 15.0, "per_batch_vram": 1.5},
    # Phi family
    "phi-3.5-mini": {"params": 3.8, "base_vram": 8.5, "per_batch_vram": 0.9},
    # SmolLM
    "smollm2-1.7b": {"params": 1.7, "base_vram": 4.0, "per_batch_vram": 0.5},
    # Generic fallbacks
    "1b": {"params": 1.0, "base_vram": 3.0, "per_batch_vram": 0.4},
    "3b": {"params": 3.0, "base_vram": 7.0, "per_batch_vram": 0.8},
    "7b": {"params": 7.0, "base_vram": 15.0, "per_batch_vram": 1.5},
    "8b": {"params": 8.0, "base_vram": 17.0, "per_batch_vram": 2.0},
}

# Learning rate recommendations by model size
LR_RECOMMENDATIONS = {
    "tiny": {"max": 1e-3, "recommended": 5e-4, "min": 1e-5},  # < 1B
    "small": {"max": 5e-4, "recommended": 2e-4, "min": 1e-5},  # 1-3B
    "medium": {"max": 3e-4, "recommended": 1e-4, "min": 1e-6},  # 3-7B
    "large": {"max": 1e-4, "recommended": 5e-5, "min": 1e-6},  # > 7B
}


def estimate_vram_usage(
    model_name: str,
    batch_size: int = 1,
    max_length: int = 512,
    gradient_accumulation: int = 1,
    precision: str = "bf16",
    gradient_checkpointing: bool = False,
    training_mode: str = "full",  # full, lora, qlora
) -> Dict[str, Any]:
    """
    Estimate VRAM usage for a training configuration.

    Returns:
        Dict with estimated VRAM breakdown
    """
    # Find matching model estimate
    model_lower = model_name.lower()
    estimate = None

    for key, est in MODEL_VRAM_ESTIMATES.items():
        if key in model_lower:
            estimate = est
            break

    # Fallback: guess from name
    if estimate is None:
        if "0.6" in model_lower or "0.5" in model_lower:
            estimate = MODEL_VRAM_ESTIMATES["qwen3-0.6b"]
        elif "1.7" in model_lower or "1b" in model_lower:
            estimate = MODEL_VRAM_ESTIMATES["1b"]
        elif "3b" in model_lower or "4b" in model_lower:
            estimate = MODEL_VRAM_ESTIMATES["3b"]
        elif "7b" in model_lower or "8b" in model_lower:
            estimate = MODEL_VRAM_ESTIMATES["7b"]
        else:
            # Conservative default
            estimate = {"params": 7.0, "base_vram": 15.0, "per_batch_vram": 1.5}

    # Base VRAM (model + optimizer states)
    base_vram = estimate["base_vram"]

    # Precision adjustment
    if precision == "fp32":
        base_vram *= 1.5  # More VRAM for fp32
    elif precision in ("fp16", "bf16"):
        base_vram *= 1.0  # Standard
    elif precision == "int8":
        base_vram *= 0.6  # Quantized

    # Training mode adjustment
    if training_mode == "qlora":
        base_vram *= 0.3  # QLoRA uses much less
    elif training_mode == "lora":
        base_vram *= 0.6  # LoRA uses less

    # Gradient checkpointing saves memory
    if gradient_checkpointing:
        base_vram *= 0.7

    # Activation memory (scales with batch size and sequence length)
    activation_vram = estimate["per_batch_vram"] * batch_size * (max_length / 512)

    # Gradient accumulation doesn't increase peak VRAM much
    # (gradients are accumulated, not stored separately)

    total_vram = base_vram + activation_vram

    return {
        "model_params_b": estimate["params"],
        "base_vram_gb": round(base_vram, 1),
        "activation_vram_gb": round(activation_vram, 1),
        "total_vram_gb": round(total_vram, 1),
        "precision": precision,
        "training_mode": training_mode,
        "gradient_checkpointing": gradient_checkpointing,
        "confidence": "estimate",  # vs "measured"
    }


def get_lr_recommendation(model_params_b: float) -> Dict[str, float]:
    """Get learning rate recommendation based on model size."""
    if model_params_b < 1.0:
        return LR_RECOMMENDATIONS["tiny"]
    elif model_params_b < 3.0:
        return LR_RECOMMENDATIONS["small"]
    elif model_params_b < 7.0:
        return LR_RECOMMENDATIONS["medium"]
    else:
        return LR_RECOMMENDATIONS["large"]


# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

@dataclass
class PreflightCheck:
    """Result of a single preflight check."""
    id: str
    name: str
    passed: bool
    severity: DiagnosticSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    @property
    def is_blocker(self) -> bool:
        """Blockers prevent training from starting."""
        return not self.passed and self.severity == DiagnosticSeverity.CRITICAL


@dataclass
class PreflightReport:
    """Complete preflight check report."""
    checks: List[PreflightCheck] = field(default_factory=list)
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Computed values
    predicted_vram_gb: float = 0.0
    available_vram_gb: float = 0.0

    @property
    def ready_to_train(self) -> bool:
        """True if no blockers found."""
        return not any(c.is_blocker for c in self.checks)

    @property
    def blockers(self) -> List[str]:
        """List of blocker messages."""
        return [c.message for c in self.checks if c.is_blocker]

    @property
    def warnings(self) -> List[str]:
        """List of warning messages."""
        return [c.message for c in self.checks
                if not c.passed and c.severity == DiagnosticSeverity.WARN]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready_to_train": self.ready_to_train,
            "checks": [
                {
                    "id": c.id,
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "message": c.message,
                    "details": c.details,
                    "recommendation": c.recommendation,
                }
                for c in self.checks
            ],
            "predicted_vram_gb": self.predicted_vram_gb,
            "available_vram_gb": self.available_vram_gb,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
        }

    def to_rpg_report(self) -> str:
        """Generate RPG-themed preflight report."""
        lines = []
        lines.append("=" * 60)
        lines.append("🛫  PRE-FLIGHT CHECK  🛫")
        lines.append("=" * 60)
        lines.append("")

        if self.ready_to_train:
            lines.append("✅ ALL SYSTEMS GO - Ready for training!")
        else:
            lines.append("❌ LAUNCH ABORTED - Issues found:")
            for blocker in self.blockers:
                lines.append(f"   🚨 {blocker}")

        lines.append("")
        lines.append(f"Checks: {self.passed_count}/{self.total_count} passed")
        lines.append(f"Predicted VRAM: {self.predicted_vram_gb:.1f}GB")
        lines.append(f"Available VRAM: {self.available_vram_gb:.1f}GB")
        lines.append("")

        # List all checks
        lines.append("CHECKLIST:")
        for check in self.checks:
            icon = "✓" if check.passed else "✗"
            sev_icon = check.severity.icon if not check.passed else ""
            lines.append(f"  [{icon}] {check.name} {sev_icon}")
            if not check.passed:
                lines.append(f"      {check.message}")
                if check.recommendation:
                    lines.append(f"      → {check.recommendation}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class PreflightChecker:
    """Runs preflight checks before training."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def run_all_checks(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> PreflightReport:
        """Run all preflight checks."""
        report = PreflightReport()

        # Load config
        if config is None and config_path:
            try:
                with open(config_path) as f:
                    config = json.load(f)
                report.config_path = str(config_path)
            except Exception as e:
                report.checks.append(PreflightCheck(
                    id="config_load",
                    name="Config Load",
                    passed=False,
                    severity=DiagnosticSeverity.CRITICAL,
                    message=f"Failed to load config: {e}",
                ))
                return report

        if config is None:
            # Try default path
            default_config = self.base_dir / "config.json"
            if default_config.exists():
                with open(default_config) as f:
                    config = json.load(f)
                report.config_path = str(default_config)
            else:
                report.checks.append(PreflightCheck(
                    id="config_load",
                    name="Config Load",
                    passed=False,
                    severity=DiagnosticSeverity.CRITICAL,
                    message="No config provided and config.json not found",
                ))
                return report

        report.config = config

        # Run checks
        report.checks.append(self._check_gpu_availability())
        report.checks.append(self._check_vram_estimate(config, report))
        report.checks.append(self._check_model_path(config))
        report.checks.append(self._check_data_path(config))
        report.checks.append(self._check_learning_rate(config))
        report.checks.append(self._check_batch_size(config))
        report.checks.append(self._check_precision(config))
        report.checks.append(self._check_disk_space())
        report.checks.append(self._check_common_mistakes(config))

        return report

    def _check_gpu_availability(self) -> PreflightCheck:
        """Check if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                return PreflightCheck(
                    id="gpu_available",
                    name="GPU Available",
                    passed=True,
                    severity=DiagnosticSeverity.INFO,
                    message=f"GPU available: {device_name} ({total_mem:.1f}GB)",
                    details={"device": device_name, "vram_gb": total_mem},
                )
            else:
                return PreflightCheck(
                    id="gpu_available",
                    name="GPU Available",
                    passed=False,
                    severity=DiagnosticSeverity.CRITICAL,
                    message="No GPU available - CUDA not found",
                    recommendation="Check NVIDIA drivers and CUDA installation",
                )
        except ImportError:
            return PreflightCheck(
                id="gpu_available",
                name="GPU Available",
                passed=False,
                severity=DiagnosticSeverity.CRITICAL,
                message="PyTorch not installed",
                recommendation="pip install torch",
            )

    def _check_vram_estimate(
        self, config: Dict[str, Any], report: PreflightReport
    ) -> PreflightCheck:
        """Estimate VRAM usage and check against available."""
        try:
            import torch

            # Get available VRAM
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                report.available_vram_gb = total_vram
            else:
                total_vram = 24.0  # Assume RTX 3090/4090
                report.available_vram_gb = total_vram

            # Extract config values
            model_name = config.get("model_name", config.get("base_model", "7b"))
            batch_size = config.get("batch_size", 1)
            max_length = config.get("max_length", 512)
            gradient_accumulation = config.get("gradient_accumulation_steps", 1)
            precision = config.get("precision", "bf16")
            gradient_checkpointing = config.get("gradient_checkpointing", False)
            training_mode = config.get("training_mode", "full")

            # Estimate VRAM
            estimate = estimate_vram_usage(
                model_name=model_name,
                batch_size=batch_size,
                max_length=max_length,
                gradient_accumulation=gradient_accumulation,
                precision=precision,
                gradient_checkpointing=gradient_checkpointing,
                training_mode=training_mode,
            )

            predicted = estimate["total_vram_gb"]
            report.predicted_vram_gb = predicted

            # Compare
            headroom = total_vram - predicted
            headroom_pct = headroom / total_vram * 100

            if predicted > total_vram:
                return PreflightCheck(
                    id="vram_estimate",
                    name="VRAM Estimate",
                    passed=False,
                    severity=DiagnosticSeverity.CRITICAL,
                    message=f"Predicted {predicted:.1f}GB exceeds available {total_vram:.1f}GB",
                    details=estimate,
                    recommendation=(
                        f"Options to reduce VRAM:\n"
                        f"  1. Reduce batch_size: {batch_size} → {max(1, batch_size // 2)}\n"
                        f"  2. Enable gradient_checkpointing: true\n"
                        f"  3. Use QLoRA: training_mode: 'qlora'"
                    ),
                )
            elif headroom_pct < 10:
                return PreflightCheck(
                    id="vram_estimate",
                    name="VRAM Estimate",
                    passed=True,
                    severity=DiagnosticSeverity.WARN,
                    message=f"Tight VRAM: {predicted:.1f}GB / {total_vram:.1f}GB ({headroom_pct:.0f}% free)",
                    details=estimate,
                    recommendation="Consider reducing batch size for safety margin",
                )
            else:
                return PreflightCheck(
                    id="vram_estimate",
                    name="VRAM Estimate",
                    passed=True,
                    severity=DiagnosticSeverity.INFO,
                    message=f"VRAM OK: {predicted:.1f}GB / {total_vram:.1f}GB ({headroom_pct:.0f}% free)",
                    details=estimate,
                )

        except Exception as e:
            return PreflightCheck(
                id="vram_estimate",
                name="VRAM Estimate",
                passed=True,  # Don't block on estimation failure
                severity=DiagnosticSeverity.WARN,
                message=f"Could not estimate VRAM: {e}",
            )

    def _check_model_path(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check if model path exists."""
        model_path = config.get("model_path") or config.get("base_model")

        if not model_path:
            return PreflightCheck(
                id="model_path",
                name="Model Path",
                passed=False,
                severity=DiagnosticSeverity.CRITICAL,
                message="No model_path or base_model specified in config",
            )

        # Check if it's a path or HF model ID
        path = Path(model_path)
        if path.exists():
            return PreflightCheck(
                id="model_path",
                name="Model Path",
                passed=True,
                severity=DiagnosticSeverity.INFO,
                message=f"Model found: {model_path}",
            )

        # Check common locations
        common_paths = [
            self.base_dir / "models" / model_path,
            self.base_dir / "models" / "current_model",
            Path(f"/home/russ/Desktop/TRAINING/models/{model_path}"),
        ]

        for p in common_paths:
            if p.exists():
                return PreflightCheck(
                    id="model_path",
                    name="Model Path",
                    passed=True,
                    severity=DiagnosticSeverity.INFO,
                    message=f"Model found at: {p}",
                )

        # Might be HuggingFace model ID
        if "/" in model_path:
            return PreflightCheck(
                id="model_path",
                name="Model Path",
                passed=True,
                severity=DiagnosticSeverity.WARN,
                message=f"Model path not local, assuming HF ID: {model_path}",
                recommendation="Model will be downloaded from HuggingFace",
            )

        return PreflightCheck(
            id="model_path",
            name="Model Path",
            passed=False,
            severity=DiagnosticSeverity.CRITICAL,
            message=f"Model not found: {model_path}",
            recommendation="Check model_path in config.json",
        )

    def _check_data_path(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check if training data exists."""
        data_path = config.get("dataset") or config.get("train_file")

        if not data_path:
            # Check for inbox
            inbox = self.base_dir / "inbox"
            if inbox.exists() and list(inbox.glob("*.jsonl")):
                return PreflightCheck(
                    id="data_path",
                    name="Training Data",
                    passed=True,
                    severity=DiagnosticSeverity.INFO,
                    message=f"Training data found in inbox/",
                )

            return PreflightCheck(
                id="data_path",
                name="Training Data",
                passed=False,
                severity=DiagnosticSeverity.WARN,
                message="No dataset specified and inbox/ empty",
                recommendation="Add training data to inbox/ or specify dataset in config",
            )

        path = Path(data_path)
        if not path.is_absolute():
            path = self.base_dir / path

        if path.exists():
            return PreflightCheck(
                id="data_path",
                name="Training Data",
                passed=True,
                severity=DiagnosticSeverity.INFO,
                message=f"Training data found: {data_path}",
            )

        return PreflightCheck(
            id="data_path",
            name="Training Data",
            passed=False,
            severity=DiagnosticSeverity.WARN,
            message=f"Training data not found: {data_path}",
        )

    def _check_learning_rate(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check if learning rate is reasonable."""
        lr = config.get("optimizer", {}).get("lr")
        if lr is None:
            lr = config.get("learning_rate", 1e-4)

        model_name = config.get("model_name", config.get("base_model", "7b"))

        # Estimate model size
        model_lower = model_name.lower()
        if "0.6" in model_lower or "0.5" in model_lower:
            params = 0.6
        elif "1.7" in model_lower or "1b" in model_lower:
            params = 1.5
        elif "3b" in model_lower or "4b" in model_lower:
            params = 3.5
        elif "7b" in model_lower or "8b" in model_lower:
            params = 7.5
        else:
            params = 7.0

        rec = get_lr_recommendation(params)

        if lr > rec["max"]:
            return PreflightCheck(
                id="learning_rate",
                name="Learning Rate",
                passed=False,
                severity=DiagnosticSeverity.WARN,
                message=f"LR {lr:.2e} likely too high for {params:.1f}B model",
                details={"lr": lr, "max_recommended": rec["max"], "model_params": params},
                recommendation=f"Try LR = {rec['recommended']:.2e}",
            )
        elif lr < rec["min"]:
            return PreflightCheck(
                id="learning_rate",
                name="Learning Rate",
                passed=True,
                severity=DiagnosticSeverity.WARN,
                message=f"LR {lr:.2e} may be too low for efficient training",
                details={"lr": lr, "min_recommended": rec["min"]},
                recommendation=f"Consider LR = {rec['recommended']:.2e}",
            )
        else:
            return PreflightCheck(
                id="learning_rate",
                name="Learning Rate",
                passed=True,
                severity=DiagnosticSeverity.INFO,
                message=f"LR {lr:.2e} looks reasonable for {params:.1f}B model",
            )

    def _check_batch_size(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check batch size configuration."""
        batch_size = config.get("batch_size", 1)
        gradient_accumulation = config.get("gradient_accumulation_steps", 1)
        effective_batch = batch_size * gradient_accumulation

        if effective_batch < 4:
            return PreflightCheck(
                id="batch_size",
                name="Batch Size",
                passed=True,
                severity=DiagnosticSeverity.WARN,
                message=f"Effective batch size {effective_batch} is small",
                recommendation="Consider increasing gradient_accumulation_steps",
            )
        elif effective_batch > 128:
            return PreflightCheck(
                id="batch_size",
                name="Batch Size",
                passed=True,
                severity=DiagnosticSeverity.WARN,
                message=f"Effective batch size {effective_batch} is large",
                recommendation="May need to reduce learning rate for large batches",
            )
        else:
            return PreflightCheck(
                id="batch_size",
                name="Batch Size",
                passed=True,
                severity=DiagnosticSeverity.INFO,
                message=f"Effective batch size: {effective_batch} ({batch_size} × {gradient_accumulation})",
            )

    def _check_precision(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check precision configuration."""
        precision = config.get("precision", "bf16")

        try:
            import torch
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                return PreflightCheck(
                    id="precision",
                    name="Precision",
                    passed=False,
                    severity=DiagnosticSeverity.WARN,
                    message="bf16 not supported on this GPU",
                    recommendation="Use precision: 'fp16' instead",
                )
        except:
            pass

        return PreflightCheck(
            id="precision",
            name="Precision",
            passed=True,
            severity=DiagnosticSeverity.INFO,
            message=f"Precision: {precision}",
        )

    def _check_disk_space(self) -> PreflightCheck:
        """Check available disk space."""
        import shutil

        # Check models directory
        models_dir = self.base_dir / "models"
        if models_dir.exists():
            usage = shutil.disk_usage(models_dir)
            free_gb = usage.free / 1e9

            if free_gb < 10:
                return PreflightCheck(
                    id="disk_space",
                    name="Disk Space",
                    passed=False,
                    severity=DiagnosticSeverity.WARN,
                    message=f"Low disk space: {free_gb:.1f}GB free",
                    recommendation="Clear old checkpoints or expand storage",
                )
            else:
                return PreflightCheck(
                    id="disk_space",
                    name="Disk Space",
                    passed=True,
                    severity=DiagnosticSeverity.INFO,
                    message=f"Disk space: {free_gb:.1f}GB free",
                )

        return PreflightCheck(
            id="disk_space",
            name="Disk Space",
            passed=True,
            severity=DiagnosticSeverity.INFO,
            message="Disk space check skipped (models/ not found)",
        )

    def _check_common_mistakes(self, config: Dict[str, Any]) -> PreflightCheck:
        """Check for common configuration mistakes."""
        issues = []

        # Check for empty/missing required fields
        if not config.get("model_name") and not config.get("base_model"):
            issues.append("No model_name or base_model specified")

        # Check for conflicting options
        if config.get("training_mode") == "qlora" and not config.get("load_in_4bit", True):
            issues.append("QLoRA mode but load_in_4bit not enabled")

        # Check for deprecated options
        deprecated = ["fp16", "bf16"]  # Should use "precision" instead
        for dep in deprecated:
            if config.get(dep) is not None:
                issues.append(f"Deprecated option '{dep}', use 'precision' instead")

        # Check optimizer config
        optimizer = config.get("optimizer", {})
        if optimizer.get("type") == "muon" and not optimizer.get("muon", {}).get("hidden_lr"):
            issues.append("Muon optimizer requires hidden_lr in muon config")

        if issues:
            return PreflightCheck(
                id="common_mistakes",
                name="Config Validation",
                passed=False,
                severity=DiagnosticSeverity.WARN,
                message=f"Found {len(issues)} potential issues",
                details={"issues": issues},
                recommendation="\n".join(f"  • {i}" for i in issues),
            )

        return PreflightCheck(
            id="common_mistakes",
            name="Config Validation",
            passed=True,
            severity=DiagnosticSeverity.INFO,
            message="No common mistakes detected",
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_preflight(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> PreflightReport:
    """
    Run preflight checks before training.

    Args:
        config: Config dict OR path string (for convenience)
        config_path: Path to config.json
        base_dir: Base directory

    Returns:
        PreflightReport with all check results

    Usage:
        run_preflight()  # Uses config.json
        run_preflight("config.json")  # Explicit path
        run_preflight(config_path="config.json")  # Named argument
        run_preflight(config={"base_model": ...})  # Dict directly
    """
    # Handle case where config is actually a path string
    if isinstance(config, str):
        config_path = config
        config = None

    # Default to config.json if nothing provided
    if config is None and config_path is None:
        config_path = "config.json"

    checker = PreflightChecker(base_dir=base_dir)
    path = Path(config_path) if config_path else None
    return checker.run_all_checks(config=config, config_path=path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-flight checks for training")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = run_preflight(config_path=args.config)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_rpg_report())

    exit(0 if report.ready_to_train else 1)
