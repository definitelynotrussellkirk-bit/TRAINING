"""
Temple Training Hooks - Integrate Diagnostics into Training Loop
================================================================

Provides HuggingFace Trainer-compatible callbacks that hook Temple
diagnostics into the training loop.

Usage with HuggingFace Trainer:
    from temple.hooks import TempleDiagnosticsCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[TempleDiagnosticsCallback()],
    )

Usage with custom training loop:
    from temple.hooks import create_training_hooks

    hooks = create_training_hooks()

    for step, batch in enumerate(dataloader):
        loss = model(batch).loss
        loss.backward()

        # Call temple hook
        hooks.on_step(step=step, loss=loss, model=model, batch=batch, lr=lr)

        optimizer.step()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from temple.diagnostics import TrainingDiagnostics, DiagnosticReport

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    """Configuration for training hooks."""
    check_interval: int = 10           # Full check every N steps
    log_interval: int = 100            # Log report every N steps
    save_interval: int = 1000          # Save history every N steps
    save_path: Optional[Path] = None   # Where to save history
    halt_on_critical: bool = False     # Raise exception on critical issue
    halt_on_nan: bool = True           # Raise exception on NaN
    auto_checkpoint_before_failure: bool = True  # Save checkpoint if failure predicted


class TrainingHaltException(Exception):
    """Exception raised when training should halt due to critical issue."""
    def __init__(self, report: DiagnosticReport):
        self.report = report
        super().__init__(f"Training halted: {report.summary}")


class TempleTrainingHooks:
    """
    Training hooks that integrate Temple diagnostics.

    Can be used with any training loop by calling methods at appropriate points.
    """

    def __init__(
        self,
        config: Optional[HookConfig] = None,
        diagnostics: Optional[TrainingDiagnostics] = None,
    ):
        self.config = config or HookConfig()
        self.diagnostics = diagnostics or TrainingDiagnostics(
            check_interval=self.config.check_interval
        )

        # Callbacks
        self._checkpoint_callback: Optional[Callable[[int], None]] = None
        self._alert_callback: Optional[Callable[[DiagnosticReport], None]] = None

        # State
        self.last_log_step = 0
        self.last_save_step = 0
        self.warnings_logged: set = set()

    def set_checkpoint_callback(self, callback: Callable[[int], None]):
        """Set callback to save checkpoint (step) -> None."""
        self._checkpoint_callback = callback

    def set_alert_callback(self, callback: Callable[[DiagnosticReport], None]):
        """Set callback for alerts (report) -> None."""
        self._alert_callback = callback

    def on_step(
        self,
        step: int,
        loss: "torch.Tensor | float",
        model: Optional["nn.Module"] = None,
        batch: Optional["torch.Tensor"] = None,
        labels: Optional["torch.Tensor"] = None,
        lr: float = 0.0,
        attention_mask: Optional["torch.Tensor"] = None,
    ) -> DiagnosticReport:
        """
        Hook to call after each training step.

        Args:
            step: Current step
            loss: Loss value
            model: Model (for gradient inspection)
            batch: Input batch
            labels: Labels
            lr: Learning rate
            attention_mask: Attention mask

        Returns:
            DiagnosticReport

        Raises:
            TrainingHaltException: If critical issue and halt_on_critical=True
        """
        import torch

        # Run diagnostics
        report = self.diagnostics.on_step(
            step=step,
            loss=loss,
            model=model,
            batch=batch,
            labels=labels,
            lr=lr,
            attention_mask=attention_mask,
        )

        # Check for NaN
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        if self.config.halt_on_nan and (math.isnan(loss_val) or math.isinf(loss_val)):
            logger.critical(f"NaN/Inf loss at step {step}")
            if self._alert_callback:
                self._alert_callback(report)
            raise TrainingHaltException(report)

        # Handle critical issues
        if report.has_critical:
            logger.critical(f"Critical issues at step {step}: {report.critical_issues}")

            # Try to save checkpoint before failure
            if self.config.auto_checkpoint_before_failure and self._checkpoint_callback:
                logger.info(f"Auto-saving emergency checkpoint at step {step}")
                try:
                    self._checkpoint_callback(step)
                except Exception as e:
                    logger.error(f"Failed to save emergency checkpoint: {e}")

            if self._alert_callback:
                self._alert_callback(report)

            if self.config.halt_on_critical:
                raise TrainingHaltException(report)

        # Handle predictions (upcoming failures)
        if report.predicted_oom_steps and report.predicted_oom_steps < 500:
            if "oom_warning" not in self.warnings_logged:
                logger.warning(f"OOM predicted in ~{report.predicted_oom_steps} steps")
                self.warnings_logged.add("oom_warning")

        if report.predicted_nan_steps and report.predicted_nan_steps < 200:
            if "nan_warning" not in self.warnings_logged:
                logger.warning(f"NaN predicted in ~{report.predicted_nan_steps} steps")
                self.warnings_logged.add("nan_warning")

        # Log report at intervals
        if step - self.last_log_step >= self.config.log_interval:
            self.last_log_step = step
            self._log_report(report)

        # Save history at intervals
        if self.config.save_path and step - self.last_save_step >= self.config.save_interval:
            self.last_save_step = step
            self.diagnostics.save_report(self.config.save_path)

        return report

    def _log_report(self, report: DiagnosticReport):
        """Log diagnostic report."""
        health = f"Health: {report.overall_health:.0%}"

        issues = []
        if report.critical_issues:
            issues.append(f"{len(report.critical_issues)} critical")
        if report.error_issues:
            issues.append(f"{len(report.error_issues)} errors")
        if report.warning_issues:
            issues.append(f"{len(report.warning_issues)} warnings")

        issue_str = ", ".join(issues) if issues else "no issues"

        logger.info(f"[Temple] Step {report.step}: {health} ({issue_str})")

        # Log specific issues at appropriate levels
        for d in report.diagnoses:
            if d.severity == DiagnosticSeverity.CRITICAL:
                logger.critical(f"  🚨 {d.summary}")
            elif d.severity == DiagnosticSeverity.ERROR:
                logger.error(f"  ❌ {d.summary}")
            elif d.severity == DiagnosticSeverity.WARN:
                logger.warning(f"  ⚠️ {d.summary}")

    def on_epoch_end(self, epoch: int):
        """Hook for end of epoch."""
        # Clear per-epoch warnings
        self.warnings_logged.clear()

        # Force full diagnostic check
        if self.diagnostics.reports:
            logger.info(f"[Temple] Epoch {epoch} complete. Final health: "
                       f"{self.diagnostics.get_current_health()['overall']:.0%}")

    def get_report(self) -> Optional[DiagnosticReport]:
        """Get latest diagnostic report."""
        return self.diagnostics.get_latest_report()

    def get_full_report(self) -> str:
        """Get full RPG-style report."""
        return self.diagnostics.get_full_report()


# Import for callback
from temple.diagnostics.severity import DiagnosticSeverity


def create_training_hooks(
    check_interval: int = 10,
    log_interval: int = 100,
    halt_on_nan: bool = True,
    halt_on_critical: bool = False,
    save_path: Optional[Path] = None,
) -> TempleTrainingHooks:
    """
    Create training hooks with common configuration.

    Args:
        check_interval: Full diagnostic check every N steps
        log_interval: Log report every N steps
        halt_on_nan: Raise exception on NaN loss
        halt_on_critical: Raise exception on critical issues
        save_path: Path to save diagnostic history

    Returns:
        TempleTrainingHooks instance
    """
    config = HookConfig(
        check_interval=check_interval,
        log_interval=log_interval,
        halt_on_nan=halt_on_nan,
        halt_on_critical=halt_on_critical,
        save_path=save_path,
    )
    return TempleTrainingHooks(config=config)


# HuggingFace Trainer compatible callback
try:
    from transformers import TrainerCallback, TrainerState, TrainerControl
    from transformers.trainer_callback import TrainerCallback

    class TempleDiagnosticsCallback(TrainerCallback):
        """
        HuggingFace Trainer callback for Temple diagnostics.

        Usage:
            trainer = Trainer(
                model=model,
                args=training_args,
                callbacks=[TempleDiagnosticsCallback()],
            )
        """

        def __init__(
            self,
            check_interval: int = 10,
            log_interval: int = 100,
            halt_on_nan: bool = True,
            halt_on_critical: bool = False,
        ):
            self.hooks = create_training_hooks(
                check_interval=check_interval,
                log_interval=log_interval,
                halt_on_nan=halt_on_nan,
                halt_on_critical=halt_on_critical,
            )
            self._model = None

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Called at the beginning of training."""
            self._model = kwargs.get("model")
            logger.info("[Temple] Diagnostics callback initialized")

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Called at the end of each training step."""
            model = kwargs.get("model", self._model)

            # Get loss from state
            loss = state.log_history[-1].get("loss") if state.log_history else None
            if loss is None:
                return

            # Get learning rate
            lr = state.log_history[-1].get("learning_rate", args.learning_rate)

            try:
                report = self.hooks.on_step(
                    step=state.global_step,
                    loss=loss,
                    model=model,
                    lr=lr,
                )

                # Let trainer know if we should stop
                if report.has_critical and self.hooks.config.halt_on_critical:
                    control.should_training_stop = True

            except TrainingHaltException as e:
                logger.critical(f"[Temple] Training halted: {e}")
                control.should_training_stop = True

        def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Called at the end of each epoch."""
            self.hooks.on_epoch_end(epoch=state.epoch)

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Called at the end of training."""
            logger.info("[Temple] Training complete. Final report:")
            logger.info(self.hooks.get_full_report())

except ImportError:
    # transformers not installed
    TempleDiagnosticsCallback = None  # type: ignore


__all__ = [
    "HookConfig",
    "TrainingHaltException",
    "TempleTrainingHooks",
    "create_training_hooks",
    "TempleDiagnosticsCallback",
]
