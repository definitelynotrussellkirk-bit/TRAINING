"""
Recovery Suggestions - Smart Recovery from Training Failures
============================================================

When training fails or degrades, the Recovery system suggests
specific actions to recover:

1. Resume points (which checkpoint to use)
2. Config changes (LR, batch size, etc.)
3. Emergency actions (save now, halt, etc.)
4. Long-term fixes (architecture, data changes)

Usage:
    from temple.recovery import RecoverySuggester, suggest_recovery

    # After a failure
    suggestions = suggest_recovery(
        report=diagnostic_report,
        config=current_config,
        ledger=checkpoint_ledger,
    )

    for s in suggestions:
        print(f"{s.priority}. {s.action}: {s.description}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from temple.diagnostics.severity import Diagnosis, DiagnosisCategory, DiagnosticSeverity

logger = logging.getLogger(__name__)


class RecoveryUrgency(Enum):
    """Urgency level for recovery actions."""
    IMMEDIATE = "immediate"   # Do this NOW
    SOON = "soon"             # Do this before next run
    LATER = "later"           # Consider for next campaign


class RecoveryType(Enum):
    """Type of recovery action."""
    RESUME = "resume"           # Resume from checkpoint
    CONFIG = "config"           # Change configuration
    EMERGENCY = "emergency"     # Emergency action (halt, save)
    FIX = "fix"                 # Fix underlying issue
    INVESTIGATE = "investigate" # Need more information


@dataclass
class RecoverySuggestion:
    """A single recovery suggestion."""
    id: str
    urgency: RecoveryUrgency
    recovery_type: RecoveryType
    action: str                           # Short action name
    description: str                      # Detailed description
    commands: List[str] = field(default_factory=list)  # Commands to execute
    config_changes: Dict[str, Any] = field(default_factory=dict)  # Config updates
    checkpoint_step: Optional[int] = None  # Checkpoint to resume from
    priority: int = 5                      # 1 = highest priority

    @property
    def icon(self) -> str:
        """Icon for this suggestion type."""
        return {
            RecoveryType.RESUME: "🔄",
            RecoveryType.CONFIG: "⚙️",
            RecoveryType.EMERGENCY: "🚨",
            RecoveryType.FIX: "🔧",
            RecoveryType.INVESTIGATE: "🔍",
        }.get(self.recovery_type, "📋")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "urgency": self.urgency.value,
            "recovery_type": self.recovery_type.value,
            "action": self.action,
            "description": self.description,
            "commands": self.commands,
            "config_changes": self.config_changes,
            "checkpoint_step": self.checkpoint_step,
            "priority": self.priority,
        }


class RecoverySuggester:
    """
    Suggests recovery actions based on diagnostic reports.

    Analyzes failures and recommends specific actions,
    config changes, and resume points.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config = config or {}

    def suggest(
        self,
        diagnoses: List[Diagnosis],
        current_step: int = 0,
    ) -> List[RecoverySuggestion]:
        """
        Generate recovery suggestions from diagnoses.

        Args:
            diagnoses: List of diagnostic findings
            current_step: Current training step

        Returns:
            List of RecoverySuggestion sorted by priority
        """
        suggestions = []

        for diagnosis in diagnoses:
            diag_suggestions = self._suggest_for_diagnosis(diagnosis, current_step)
            suggestions.extend(diag_suggestions)

        # Deduplicate and sort by priority
        seen_ids = set()
        unique_suggestions = []
        for s in suggestions:
            if s.id not in seen_ids:
                seen_ids.add(s.id)
                unique_suggestions.append(s)

        unique_suggestions.sort(key=lambda s: s.priority)
        return unique_suggestions

    def _suggest_for_diagnosis(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Generate suggestions for a specific diagnosis."""
        suggestions = []

        # Route to specific handlers based on diagnosis ID pattern
        if "nan" in diagnosis.id.lower():
            suggestions.extend(self._suggest_nan_recovery(diagnosis, current_step))
        elif "gradient" in diagnosis.id.lower():
            suggestions.extend(self._suggest_gradient_recovery(diagnosis, current_step))
        elif "memory" in diagnosis.id.lower() or "oom" in diagnosis.id.lower():
            suggestions.extend(self._suggest_memory_recovery(diagnosis, current_step))
        elif "lr" in diagnosis.id.lower() or "learning_rate" in diagnosis.id.lower():
            suggestions.extend(self._suggest_lr_recovery(diagnosis, current_step))
        elif "data" in diagnosis.id.lower():
            suggestions.extend(self._suggest_data_recovery(diagnosis, current_step))

        return suggestions

    def _suggest_nan_recovery(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Suggest recovery from NaN/Inf issues."""
        suggestions = []
        current_lr = self.config.get("optimizer", {}).get("lr", 1e-4)

        # Emergency: Save state
        if diagnosis.severity == DiagnosticSeverity.CRITICAL:
            suggestions.append(RecoverySuggestion(
                id="nan_emergency_save",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.EMERGENCY,
                action="Save emergency checkpoint",
                description="Save current state before it gets worse",
                commands=[
                    f"python3 -c \"from core.train import save_checkpoint; save_checkpoint({current_step}, emergency=True)\"",
                ],
                priority=1,
            ))

        # Find resume point
        safe_step = self._find_safe_checkpoint(current_step)
        if safe_step:
            suggestions.append(RecoverySuggestion(
                id="nan_resume",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.RESUME,
                action="Resume from safe checkpoint",
                description=f"Resume from checkpoint-{safe_step} (before NaN occurred)",
                checkpoint_step=safe_step,
                commands=[
                    f"python3 core/train.py --resume checkpoint-{safe_step} --yes",
                ],
                priority=2,
            ))

        # LR reduction
        new_lr = current_lr * 0.1
        suggestions.append(RecoverySuggestion(
            id="nan_reduce_lr",
            urgency=RecoveryUrgency.IMMEDIATE,
            recovery_type=RecoveryType.CONFIG,
            action="Reduce learning rate",
            description=f"Reduce LR from {current_lr:.2e} to {new_lr:.2e}",
            config_changes={"optimizer.lr": new_lr},
            priority=3,
        ))

        # Gradient clipping
        suggestions.append(RecoverySuggestion(
            id="nan_gradient_clip",
            urgency=RecoveryUrgency.SOON,
            recovery_type=RecoveryType.CONFIG,
            action="Enable gradient clipping",
            description="Add gradient clipping to prevent explosion",
            config_changes={"optimizer.max_grad_norm": 1.0},
            priority=4,
        ))

        return suggestions

    def _suggest_gradient_recovery(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Suggest recovery from gradient issues."""
        suggestions = []
        layer = diagnosis.layer

        if "vanishing" in diagnosis.id.lower() or "dead" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="gradient_architecture",
                urgency=RecoveryUrgency.LATER,
                recovery_type=RecoveryType.FIX,
                action="Review architecture",
                description=f"Layer {layer} has vanishing/dead gradients. Consider architecture changes.",
                priority=7,
            ))

            # Increase LR for this specific case
            current_lr = self.config.get("optimizer", {}).get("lr", 1e-4)
            suggestions.append(RecoverySuggestion(
                id="gradient_increase_lr",
                urgency=RecoveryUrgency.SOON,
                recovery_type=RecoveryType.CONFIG,
                action="Increase learning rate",
                description=f"Higher LR may help with vanishing gradients: {current_lr:.2e} → {current_lr * 3:.2e}",
                config_changes={"optimizer.lr": current_lr * 3},
                priority=5,
            ))

        elif "exploding" in diagnosis.id.lower() or "diverging" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="gradient_clip_urgent",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.CONFIG,
                action="Enable gradient clipping NOW",
                description="Gradients are exploding - clip immediately",
                config_changes={"optimizer.max_grad_norm": 1.0},
                priority=1,
            ))

            current_lr = self.config.get("optimizer", {}).get("lr", 1e-4)
            suggestions.append(RecoverySuggestion(
                id="gradient_reduce_lr",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.CONFIG,
                action="Reduce learning rate",
                description=f"Lower LR to stabilize: {current_lr:.2e} → {current_lr * 0.5:.2e}",
                config_changes={"optimizer.lr": current_lr * 0.5},
                priority=2,
            ))

        return suggestions

    def _suggest_memory_recovery(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Suggest recovery from memory issues."""
        suggestions = []

        if "critical" in diagnosis.id.lower() or diagnosis.severity == DiagnosticSeverity.CRITICAL:
            # Emergency: Save and halt
            suggestions.append(RecoverySuggestion(
                id="memory_emergency",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.EMERGENCY,
                action="Save checkpoint and halt",
                description="Memory critical - save state before OOM crash",
                priority=1,
            ))

        # Batch size reduction
        current_batch = self.config.get("batch_size", 16)
        new_batch = max(1, current_batch // 2)
        suggestions.append(RecoverySuggestion(
            id="memory_batch_size",
            urgency=RecoveryUrgency.SOON,
            recovery_type=RecoveryType.CONFIG,
            action="Reduce batch size",
            description=f"Reduce batch size: {current_batch} → {new_batch}",
            config_changes={"batch_size": new_batch},
            priority=2,
        ))

        # Gradient checkpointing
        suggestions.append(RecoverySuggestion(
            id="memory_gradient_checkpoint",
            urgency=RecoveryUrgency.SOON,
            recovery_type=RecoveryType.CONFIG,
            action="Enable gradient checkpointing",
            description="Trade compute for memory with gradient checkpointing",
            config_changes={"gradient_checkpointing": True},
            priority=3,
        ))

        # Mixed precision
        suggestions.append(RecoverySuggestion(
            id="memory_mixed_precision",
            urgency=RecoveryUrgency.SOON,
            recovery_type=RecoveryType.CONFIG,
            action="Enable mixed precision",
            description="Use fp16/bf16 to reduce memory usage",
            config_changes={"precision": "bf16"},
            priority=4,
        ))

        # Memory leak investigation
        if "leak" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="memory_leak_investigate",
                urgency=RecoveryUrgency.SOON,
                recovery_type=RecoveryType.INVESTIGATE,
                action="Investigate memory leak",
                description=(
                    "Check for:\n"
                    "1. Tensors stored in lists\n"
                    "2. Missing loss.backward()\n"
                    "3. Storing loss tensor instead of loss.item()"
                ),
                priority=5,
            ))

        return suggestions

    def _suggest_lr_recovery(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Suggest recovery from learning rate issues."""
        suggestions = []
        current_lr = self.config.get("optimizer", {}).get("lr", 1e-4)

        if "oscillation" in diagnosis.id.lower():
            new_lr = current_lr * 0.5
            suggestions.append(RecoverySuggestion(
                id="lr_oscillation_fix",
                urgency=RecoveryUrgency.SOON,
                recovery_type=RecoveryType.CONFIG,
                action="Reduce learning rate",
                description=f"Oscillating loss suggests LR too high: {current_lr:.2e} → {new_lr:.2e}",
                config_changes={"optimizer.lr": new_lr},
                priority=3,
            ))

        elif "divergence" in diagnosis.id.lower():
            new_lr = current_lr * 0.1
            suggestions.append(RecoverySuggestion(
                id="lr_divergence_fix",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.CONFIG,
                action="Significantly reduce learning rate",
                description=f"Diverging loss - drastic LR cut needed: {current_lr:.2e} → {new_lr:.2e}",
                config_changes={"optimizer.lr": new_lr},
                priority=1,
            ))

            safe_step = self._find_safe_checkpoint(current_step)
            if safe_step:
                suggestions.append(RecoverySuggestion(
                    id="lr_divergence_resume",
                    urgency=RecoveryUrgency.IMMEDIATE,
                    recovery_type=RecoveryType.RESUME,
                    action="Resume from earlier checkpoint",
                    description=f"Resume from checkpoint-{safe_step} with lower LR",
                    checkpoint_step=safe_step,
                    priority=2,
                ))

        elif "plateau" in diagnosis.id.lower():
            new_lr = current_lr * 3
            suggestions.append(RecoverySuggestion(
                id="lr_plateau_fix",
                urgency=RecoveryUrgency.LATER,
                recovery_type=RecoveryType.CONFIG,
                action="Increase learning rate",
                description=f"Loss plateau - try higher LR: {current_lr:.2e} → {new_lr:.2e}",
                config_changes={"optimizer.lr": new_lr},
                priority=5,
            ))

            suggestions.append(RecoverySuggestion(
                id="lr_plateau_scheduler",
                urgency=RecoveryUrgency.LATER,
                recovery_type=RecoveryType.CONFIG,
                action="Use cosine scheduler with restarts",
                description="Cosine annealing with warm restarts can escape plateaus",
                config_changes={"scheduler.type": "cosine_with_restarts"},
                priority=6,
            ))

        return suggestions

    def _suggest_data_recovery(
        self,
        diagnosis: Diagnosis,
        current_step: int,
    ) -> List[RecoverySuggestion]:
        """Suggest recovery from data issues."""
        suggestions = []

        if "nan" in diagnosis.id.lower() or "inf" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="data_validation",
                urgency=RecoveryUrgency.IMMEDIATE,
                recovery_type=RecoveryType.FIX,
                action="Add data validation",
                description="Filter NaN/Inf from data pipeline",
                commands=[
                    "# Add to data loading:",
                    "batch = torch.nan_to_num(batch, nan=0.0, posinf=1e6, neginf=-1e6)",
                ],
                priority=1,
            ))

        if "padding" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="data_padding_fix",
                urgency=RecoveryUrgency.LATER,
                recovery_type=RecoveryType.FIX,
                action="Optimize padding",
                description="Reduce wasted compute on padding tokens",
                priority=5,
            ))

        if "variance" in diagnosis.id.lower():
            suggestions.append(RecoverySuggestion(
                id="data_variance_investigate",
                urgency=RecoveryUrgency.SOON,
                recovery_type=RecoveryType.INVESTIGATE,
                action="Investigate data variance",
                description="Check data loading - batch may be all-zeros or all-same",
                priority=3,
            ))

        return suggestions

    def _find_safe_checkpoint(self, current_step: int) -> Optional[int]:
        """Find a safe checkpoint to resume from."""
        ledger_path = self.base_dir / "status" / "checkpoint_ledger.json"

        if not ledger_path.exists():
            return None

        try:
            with open(ledger_path) as f:
                ledger = json.load(f)

            entries = ledger.get("entries", {})

            # Find checkpoints before current step
            valid_steps = []
            for key, entry in entries.items():
                step = entry.get("step", 0)
                # Only consider checkpoints that are significantly before current
                # and have reasonable loss
                if step < current_step - 100:
                    train_loss = entry.get("stats", {}).get("train_loss")
                    if train_loss is not None and train_loss < 10:  # Reasonable loss
                        valid_steps.append(step)

            if valid_steps:
                return max(valid_steps)

        except Exception as e:
            logger.debug(f"Failed to find safe checkpoint: {e}")

        return None


def suggest_recovery(
    diagnoses: List[Diagnosis],
    current_step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Path] = None,
) -> List[RecoverySuggestion]:
    """
    Convenience function to get recovery suggestions.

    Args:
        diagnoses: List of diagnostic findings
        current_step: Current training step
        config: Current training config
        base_dir: Base directory

    Returns:
        List of RecoverySuggestion sorted by priority
    """
    suggester = RecoverySuggester(base_dir=base_dir, config=config or {})
    return suggester.suggest(diagnoses, current_step)


def format_recovery_plan(suggestions: List[RecoverySuggestion]) -> str:
    """Format recovery suggestions as a readable plan."""
    if not suggestions:
        return "No recovery actions needed."

    lines = []
    lines.append("=" * 60)
    lines.append("🏥  RECOVERY PLAN  🏥")
    lines.append("=" * 60)
    lines.append("")

    # Group by urgency
    immediate = [s for s in suggestions if s.urgency == RecoveryUrgency.IMMEDIATE]
    soon = [s for s in suggestions if s.urgency == RecoveryUrgency.SOON]
    later = [s for s in suggestions if s.urgency == RecoveryUrgency.LATER]

    if immediate:
        lines.append("🚨 IMMEDIATE ACTIONS:")
        for s in immediate:
            lines.append(f"  {s.icon} {s.action}")
            lines.append(f"     {s.description}")
            if s.config_changes:
                lines.append(f"     Config: {s.config_changes}")
            if s.checkpoint_step:
                lines.append(f"     Resume from: checkpoint-{s.checkpoint_step}")
            lines.append("")

    if soon:
        lines.append("⚠️ DO SOON:")
        for s in soon:
            lines.append(f"  {s.icon} {s.action}")
            lines.append(f"     {s.description}")
            lines.append("")

    if later:
        lines.append("📋 CONSIDER LATER:")
        for s in later:
            lines.append(f"  {s.icon} {s.action}")
            lines.append(f"     {s.description}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
