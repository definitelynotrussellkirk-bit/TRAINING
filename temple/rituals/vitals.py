"""
Vitals Ritual - Live Training Health Diagnostics
=================================================

The Vitals ritual exposes Temple diagnostic state through the ritual system.
Unlike other rituals that check infrastructure, Vitals checks the TRAINING PROCESS.

Use this to:
- Get real-time training health status
- Check for predicted failures
- Get suggested remediations
- Monitor gradient/memory/LR health

Usage:
    from temple import run_ritual

    result = run_ritual("vitals")
    print(f"Training health: {result.status}")

The ritual reads from:
1. Live diagnostic state (if training is running)
2. Saved diagnostic history (if available)
3. Current GPU/system state (always)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from temple.cleric import register_ritual
from temple.protocols import RitualContext, RitualKind, RitualMeta
from temple.schemas import RitualCheckResult, ResultStatus

logger = logging.getLogger(__name__)


# Register the ritual
@register_ritual(RitualMeta(
    id="vitals",
    name="Ritual of Vitals",
    description="Live training health diagnostics - gradients, memory, LR, data quality",
    kind=RitualKind.TRAINING,
    icon="💓",
    tags=("diagnostics", "training", "health"),
))
def run(ctx: RitualContext) -> List[RitualCheckResult]:
    """
    Run the Vitals ritual - check training health.

    Checks:
    1. Diagnostic history file (if exists)
    2. GPU memory status
    3. Training status
    4. Recent loss trends
    """
    checks = []

    # Check 1: Diagnostic history
    checks.append(_check_diagnostic_history(ctx))

    # Check 2: GPU memory
    checks.append(_check_gpu_memory(ctx))

    # Check 3: Training status
    checks.append(_check_training_status(ctx))

    # Check 4: Recent checkpoints
    checks.append(_check_recent_checkpoints(ctx))

    # Check 5: Loss file analysis
    checks.append(_check_loss_trends(ctx))

    # Check 6: Fleet version (multi-machine code staleness)
    checks.append(_check_fleet_version(ctx))

    return checks


def _check_diagnostic_history(ctx: RitualContext) -> RitualCheckResult:
    """Check if diagnostic history exists and is recent."""
    history_path = ctx.base_dir / "status" / "temple_diagnostics.json"

    if not history_path.exists():
        return RitualCheckResult(
            id="vitals_history",
            name="Diagnostic History",
            description="Check for saved diagnostic history",
            status="skip",
            details={"message": "No diagnostic history found (training may not have run yet)"},
            remediation="Run training with Temple hooks enabled to generate diagnostic history",
        )

    try:
        with open(history_path) as f:
            history = json.load(f)

        saved_at = history.get("saved_at", "unknown")
        step_count = history.get("step_count", 0)
        current_health = history.get("current_health", {})
        reports = history.get("reports", [])

        # Check if history is fresh (within last hour)
        try:
            saved_time = datetime.fromisoformat(saved_at)
            age_seconds = (datetime.now() - saved_time).total_seconds()
            is_fresh = age_seconds < 3600
        except:
            is_fresh = False

        # Check health scores
        overall_health = current_health.get("overall", 1.0)

        if overall_health < 0.3:
            status = "fail"
            message = f"Training health critical: {overall_health:.0%}"
        elif overall_health < 0.7:
            status = "warn"
            message = f"Training health degraded: {overall_health:.0%}"
        else:
            status = "ok"
            message = f"Training health good: {overall_health:.0%}"

        # Check for predictions
        predictions = []
        if reports:
            latest = reports[-1]
            if latest.get("predictions", {}).get("oom_in_steps"):
                predictions.append(f"OOM in ~{latest['predictions']['oom_in_steps']} steps")
            if latest.get("predictions", {}).get("nan_in_steps"):
                predictions.append(f"NaN in ~{latest['predictions']['nan_in_steps']} steps")

        return RitualCheckResult(
            id="vitals_history",
            name="Diagnostic History",
            description="Training diagnostic history",
            status=status,
            details={
                "message": message,
                "saved_at": saved_at,
                "is_fresh": is_fresh,
                "step_count": step_count,
                "health": current_health,
                "predictions": predictions,
                "report_count": len(reports),
            },
            remediation="Review diagnostic reports for specific issues" if status != "ok" else None,
        )

    except Exception as e:
        return RitualCheckResult(
            id="vitals_history",
            name="Diagnostic History",
            description="Training diagnostic history",
            status="warn",
            details={"error": str(e)},
            remediation="Check diagnostic history file format",
        )


def _check_gpu_memory(ctx: RitualContext) -> RitualCheckResult:
    """Check current GPU memory status."""
    try:
        import torch

        if not torch.cuda.is_available():
            return RitualCheckResult(
                id="vitals_gpu_memory",
                name="GPU Memory",
                description="Current GPU memory status",
                status="skip",
                details={"message": "CUDA not available"},
            )

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        usage = allocated / total

        if usage > 0.95:
            status = "fail"
            message = f"GPU memory critical: {usage:.0%} used ({free:.1f}GB free)"
            remediation = "Reduce batch size or enable gradient checkpointing"
        elif usage > 0.85:
            status = "warn"
            message = f"GPU memory high: {usage:.0%} used ({free:.1f}GB free)"
            remediation = "Monitor for OOM, consider reducing batch size"
        else:
            status = "ok"
            message = f"GPU memory OK: {usage:.0%} used ({free:.1f}GB free)"
            remediation = None

        return RitualCheckResult(
            id="vitals_gpu_memory",
            name="GPU Memory",
            description="Current GPU memory status",
            status=status,
            details={
                "message": message,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(free, 2),
                "usage_percent": round(usage * 100, 1),
            },
            remediation=remediation,
        )

    except ImportError:
        return RitualCheckResult(
            id="vitals_gpu_memory",
            name="GPU Memory",
            description="Current GPU memory status",
            status="skip",
            details={"message": "PyTorch not available"},
        )
    except Exception as e:
        return RitualCheckResult(
            id="vitals_gpu_memory",
            name="GPU Memory",
            description="Current GPU memory status",
            status="warn",
            details={"error": str(e)},
        )


def _check_training_status(ctx: RitualContext) -> RitualCheckResult:
    """Check if training is currently running."""
    training_status_path = ctx.base_dir / "status" / "training_status.json"

    if not training_status_path.exists():
        return RitualCheckResult(
            id="vitals_training_status",
            name="Training Status",
            description="Is training currently running?",
            status="skip",
            details={"message": "No training status file found"},
        )

    try:
        with open(training_status_path) as f:
            status = json.load(f)

        is_training = status.get("is_training", False)
        current_step = status.get("current_step", 0)
        current_loss = status.get("current_loss")
        updated_at = status.get("updated_at", "unknown")

        if is_training:
            status_result = "ok"
            message = f"Training active at step {current_step}"
        else:
            status_result = "ok"
            message = f"Training not running (last step: {current_step})"

        return RitualCheckResult(
            id="vitals_training_status",
            name="Training Status",
            description="Training run status",
            status=status_result,
            details={
                "message": message,
                "is_training": is_training,
                "current_step": current_step,
                "current_loss": current_loss,
                "updated_at": updated_at,
            },
        )

    except Exception as e:
        return RitualCheckResult(
            id="vitals_training_status",
            name="Training Status",
            description="Training run status",
            status="warn",
            details={"error": str(e)},
        )


def _check_recent_checkpoints(ctx: RitualContext) -> RitualCheckResult:
    """Check recent checkpoint health."""
    ledger_path = ctx.base_dir / "status" / "checkpoint_ledger.json"

    if not ledger_path.exists():
        return RitualCheckResult(
            id="vitals_checkpoints",
            name="Recent Checkpoints",
            description="Health of recent checkpoints",
            status="skip",
            details={"message": "No checkpoint ledger found"},
        )

    try:
        with open(ledger_path) as f:
            ledger = json.load(f)

        entries = ledger.get("entries", {})

        if not entries:
            return RitualCheckResult(
                id="vitals_checkpoints",
                name="Recent Checkpoints",
                description="Health of recent checkpoints",
                status="skip",
                details={"message": "No checkpoints in ledger"},
            )

        # Get recent checkpoints (by step)
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].get("step", 0),
            reverse=True
        )[:10]

        # Analyze loss trend
        losses = []
        for _, entry in sorted_entries:
            train_loss = entry.get("stats", {}).get("train_loss")
            if train_loss is not None:
                losses.append(train_loss)

        if len(losses) >= 2:
            recent_loss = losses[0]
            older_loss = losses[-1]

            if recent_loss < older_loss * 0.95:
                trend = "improving"
                status = "ok"
                message = f"Loss improving: {older_loss:.4f} → {recent_loss:.4f}"
            elif recent_loss > older_loss * 1.1:
                trend = "degrading"
                status = "warn"
                message = f"Loss degrading: {older_loss:.4f} → {recent_loss:.4f}"
            else:
                trend = "stable"
                status = "ok"
                message = f"Loss stable around {recent_loss:.4f}"
        else:
            trend = "unknown"
            status = "ok"
            message = "Not enough data for trend analysis"

        return RitualCheckResult(
            id="vitals_checkpoints",
            name="Recent Checkpoints",
            description="Checkpoint loss trends",
            status=status,
            details={
                "message": message,
                "trend": trend,
                "checkpoint_count": len(entries),
                "recent_losses": losses[:5],
            },
            remediation="Check learning rate if loss is degrading" if status != "ok" else None,
        )

    except Exception as e:
        return RitualCheckResult(
            id="vitals_checkpoints",
            name="Recent Checkpoints",
            description="Checkpoint loss trends",
            status="warn",
            details={"error": str(e)},
        )


def _check_loss_trends(ctx: RitualContext) -> RitualCheckResult:
    """Analyze loss trends from battle log."""
    battle_log_path = ctx.base_dir / "status" / "battle_log.jsonl"

    if not battle_log_path.exists():
        return RitualCheckResult(
            id="vitals_loss_trends",
            name="Loss Trends",
            description="Analysis of recent loss values",
            status="skip",
            details={"message": "No battle log found"},
        )

    try:
        # Read last 100 lines
        lines = []
        with open(battle_log_path) as f:
            for line in f:
                lines.append(line)
                if len(lines) > 100:
                    lines.pop(0)

        if not lines:
            return RitualCheckResult(
                id="vitals_loss_trends",
                name="Loss Trends",
                description="Analysis of recent loss values",
                status="skip",
                details={"message": "Battle log is empty"},
            )

        losses = []
        for line in lines:
            try:
                entry = json.loads(line.strip())
                loss = entry.get("loss") or entry.get("train_loss")
                if loss is not None and not (isinstance(loss, float) and (loss != loss)):  # NaN check
                    losses.append(loss)
            except:
                continue

        if len(losses) < 10:
            return RitualCheckResult(
                id="vitals_loss_trends",
                name="Loss Trends",
                description="Analysis of recent loss values",
                status="skip",
                details={"message": f"Only {len(losses)} valid loss values found"},
            )

        # Statistics
        mean_loss = sum(losses) / len(losses)
        variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
        std_loss = variance ** 0.5
        cv = std_loss / mean_loss if mean_loss > 0 else 0

        # Trend (compare first half to second half)
        first_half = losses[:len(losses)//2]
        second_half = losses[len(losses)//2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg < first_avg * 0.95:
            trend = "improving"
        elif second_avg > first_avg * 1.1:
            trend = "degrading"
        else:
            trend = "stable"

        # Detect oscillation
        if cv > 0.3:
            status = "warn"
            message = f"High loss variance (CV={cv:.2f}), possible oscillation"
            remediation = "Consider reducing learning rate"
        elif trend == "degrading":
            status = "warn"
            message = f"Loss trending upward: {first_avg:.4f} → {second_avg:.4f}"
            remediation = "Check learning rate and data quality"
        else:
            status = "ok"
            message = f"Loss {trend}: mean={mean_loss:.4f}, std={std_loss:.4f}"
            remediation = None

        return RitualCheckResult(
            id="vitals_loss_trends",
            name="Loss Trends",
            description="Analysis of recent loss values",
            status=status,
            details={
                "message": message,
                "trend": trend,
                "mean_loss": round(mean_loss, 6),
                "std_loss": round(std_loss, 6),
                "cv": round(cv, 4),
                "sample_count": len(losses),
            },
            remediation=remediation,
        )

    except Exception as e:
        return RitualCheckResult(
            id="vitals_loss_trends",
            name="Loss Trends",
            description="Analysis of recent loss values",
            status="warn",
            details={"error": str(e)},
        )


def _check_fleet_version(ctx: RitualContext) -> RitualCheckResult:
    """Check fleet version consistency across machines."""
    try:
        from core.fleet_version import get_fleet_manager

        manager = get_fleet_manager()
        health = manager.check_fleet_health()

        stale_services = health.get("stale_services", [])
        code_match = health.get("code_hash_match", True)
        config_match = health.get("config_hash_match", True)

        if stale_services:
            status = "warn"
            message = f"Stale services need restart: {', '.join(stale_services)}"
            remediation = "Restart stale services to pick up code changes: " + ", ".join(stale_services)
        elif not code_match:
            status = "warn"
            message = "Code has changed since manifest was generated"
            remediation = "Run `python3 core/fleet_version.py update` to update manifest"
        elif not config_match:
            status = "ok"  # Config changes are less critical
            message = "Config changed but code is current"
            remediation = None
        else:
            status = "ok"
            message = f"Fleet version {health.get('manifest_version', 'unknown')} - all services current"
            remediation = None

        return RitualCheckResult(
            id="vitals_fleet_version",
            name="Fleet Version",
            description="Multi-machine code version consistency",
            status=status,
            details={
                "message": message,
                "version": health.get("manifest_version"),
                "code_hash_match": code_match,
                "config_hash_match": config_match,
                "stale_services": stale_services,
                "current_hash": health.get("current_code_hash"),
                "manifest_hash": health.get("manifest_code_hash"),
            },
            remediation=remediation,
        )

    except ImportError:
        return RitualCheckResult(
            id="vitals_fleet_version",
            name="Fleet Version",
            description="Multi-machine code version consistency",
            status="skip",
            details={"message": "Fleet version module not available"},
        )
    except Exception as e:
        return RitualCheckResult(
            id="vitals_fleet_version",
            name="Fleet Version",
            description="Multi-machine code version consistency",
            status="warn",
            details={"error": str(e)},
        )
