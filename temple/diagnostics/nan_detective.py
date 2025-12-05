"""
NaN Detective - Root Cause Analysis for NaN/Inf Losses
=======================================================

The #1 training killer is NaN loss. This module doesn't just detect NaN -
it traces the root cause and suggests specific fixes.

Root Causes (in order of likelihood):
1. Learning rate too high → Gradient explosion → NaN
2. Gradient explosion in specific layer → NaN propagates
3. log(0) or log(negative) in loss function
4. Division by zero in normalization
5. Bad data (NaN/Inf in input)
6. Activation explosion (values > 1e4)
7. Numerical instability in fp16

Usage:
    from temple.diagnostics import NaNDetective

    detective = NaNDetective()

    # When loss goes NaN
    diagnosis = detective.investigate(
        loss=loss,
        model=model,
        batch=batch,
        lr=0.001,
        step=500,
    )

    print(f"Root cause: {diagnosis.summary}")
    print(f"Fix: {diagnosis.remediation}")
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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
class NaNEvidence:
    """Evidence collected during NaN investigation."""
    has_nan_loss: bool = False
    has_inf_loss: bool = False
    nan_in_gradients: List[str] = field(default_factory=list)  # Layer names
    inf_in_gradients: List[str] = field(default_factory=list)
    exploding_gradients: List[Tuple[str, float]] = field(default_factory=list)  # (layer, magnitude)
    vanishing_gradients: List[str] = field(default_factory=list)
    nan_in_activations: List[str] = field(default_factory=list)
    large_activations: List[Tuple[str, float]] = field(default_factory=list)
    nan_in_batch: bool = False
    inf_in_batch: bool = False
    step: int = 0
    lr: float = 0.0
    loss_value: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    using_fp16: bool = False
    has_log_ops: bool = False
    has_div_ops: bool = False


class NaNDetective:
    """
    Investigates NaN/Inf losses to find root cause.

    The detective maintains history to detect patterns and
    provides specific, actionable diagnosis.
    """

    # Common layer patterns that are NaN-prone
    NAN_PRONE_LAYERS = [
        "LayerNorm", "BatchNorm", "RMSNorm",  # Normalization (div by zero)
        "Softmax", "LogSoftmax",               # Numerical instability
        "CrossEntropyLoss",                    # log(0)
        "Embedding",                           # Out-of-vocab indices
    ]

    def __init__(self, thresholds: Optional[DiagnosticThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.loss_history: deque = deque(maxlen=1000)
        self.gradient_history: Dict[str, deque] = {}
        self.last_healthy_step: int = 0
        self._activation_hooks: List[Any] = []

    def record_loss(self, loss: float, step: int):
        """Record loss for trend analysis."""
        if not math.isnan(loss) and not math.isinf(loss):
            self.loss_history.append((step, loss))
            self.last_healthy_step = step

    def investigate(
        self,
        loss: "torch.Tensor | float",
        model: Optional["nn.Module"] = None,
        batch: Optional["torch.Tensor"] = None,
        lr: float = 0.0,
        step: int = 0,
        activations: Optional[Dict[str, "torch.Tensor"]] = None,
    ) -> Diagnosis:
        """
        Investigate NaN/Inf loss and determine root cause.

        Args:
            loss: The loss tensor or value
            model: The model (for gradient inspection)
            batch: The input batch (for data inspection)
            lr: Current learning rate
            step: Current training step
            activations: Optional dict of layer activations

        Returns:
            Diagnosis with root cause and remediation
        """
        import torch

        evidence = NaNEvidence(step=step, lr=lr)

        # Convert loss to float
        if isinstance(loss, torch.Tensor):
            loss_val = loss.item() if loss.numel() == 1 else float(loss.mean())
        else:
            loss_val = float(loss)

        evidence.loss_value = loss_val
        evidence.loss_history = [l for _, l in list(self.loss_history)[-100:]]

        # Check the loss itself
        evidence.has_nan_loss = math.isnan(loss_val)
        evidence.has_inf_loss = math.isinf(loss_val)

        if not evidence.has_nan_loss and not evidence.has_inf_loss:
            # Not a NaN/Inf situation - just record and return info
            self.record_loss(loss_val, step)
            return Diagnosis(
                id="nan_not_detected",
                category=DiagnosisCategory.LOSS,
                severity=DiagnosticSeverity.INFO,
                summary="Loss is healthy",
                details=f"Loss value {loss_val:.6f} is finite",
                remediation="No action needed",
                evidence={"loss": loss_val},
                step=step,
            )

        # Investigation begins!
        logger.warning(f"NaN/Inf detected at step {step}, investigating...")

        # Check batch for bad data
        if batch is not None:
            evidence.nan_in_batch = torch.isnan(batch).any().item()
            evidence.inf_in_batch = torch.isinf(batch).any().item()

        # Check model gradients
        if model is not None:
            evidence = self._check_gradients(model, evidence)
            evidence = self._check_model_properties(model, evidence)

        # Check activations
        if activations is not None:
            evidence = self._check_activations(activations, evidence)

        # Determine root cause based on evidence
        return self._diagnose(evidence)

    def _check_gradients(self, model: "nn.Module", evidence: NaNEvidence) -> NaNEvidence:
        """Check model gradients for issues."""
        import torch

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad

                # NaN in gradients
                if torch.isnan(grad).any():
                    evidence.nan_in_gradients.append(name)

                # Inf in gradients
                if torch.isinf(grad).any():
                    evidence.inf_in_gradients.append(name)

                # Exploding gradients
                grad_norm = grad.norm().item()
                if grad_norm > self.thresholds.gradient_exploding_threshold:
                    evidence.exploding_gradients.append((name, grad_norm))

                # Vanishing gradients
                if grad_norm < self.thresholds.gradient_vanishing_threshold:
                    evidence.vanishing_gradients.append(name)

        return evidence

    def _check_activations(
        self, activations: Dict[str, "torch.Tensor"], evidence: NaNEvidence
    ) -> NaNEvidence:
        """Check activations for issues."""
        import torch

        for name, act in activations.items():
            if torch.isnan(act).any():
                evidence.nan_in_activations.append(name)

            max_val = act.abs().max().item()
            if max_val > self.thresholds.data_max_value:
                evidence.large_activations.append((name, max_val))

        return evidence

    def _check_model_properties(self, model: "nn.Module", evidence: NaNEvidence) -> NaNEvidence:
        """Check model for NaN-prone patterns."""
        model_str = str(model)

        # Check for log operations
        evidence.has_log_ops = any(x in model_str for x in ["LogSoftmax", "log_", "NLLLoss"])

        # Check for division operations (normalization)
        evidence.has_div_ops = any(x in model_str for x in ["LayerNorm", "BatchNorm", "RMSNorm"])

        # Check if using fp16
        for param in model.parameters():
            if param.dtype in [getattr(__import__('torch'), 'float16', None),
                              getattr(__import__('torch'), 'bfloat16', None)]:
                evidence.using_fp16 = True
                break

        return evidence

    def _diagnose(self, evidence: NaNEvidence) -> Diagnosis:
        """Determine root cause from evidence."""

        # Priority 1: Bad data in batch (most common, easiest to fix)
        if evidence.nan_in_batch or evidence.inf_in_batch:
            return Diagnosis(
                id="nan_bad_input_data",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.CRITICAL,
                summary="NaN/Inf in input data",
                details=(
                    f"Detected {'NaN' if evidence.nan_in_batch else 'Inf'} values in the input batch. "
                    "The model received corrupted data that propagated through computation."
                ),
                remediation=(
                    "Clean your dataset:\n"
                    "1. Add data validation: `assert not torch.isnan(batch).any()`\n"
                    "2. Replace NaN: `batch = torch.nan_to_num(batch, nan=0.0)`\n"
                    "3. Filter bad samples during data loading"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
            )

        # Priority 2: Early NaN (< 100 steps) → LR too high
        if evidence.step < 100:
            return Diagnosis(
                id="nan_lr_too_high_early",
                category=DiagnosisCategory.LEARNING_RATE,
                severity=DiagnosticSeverity.CRITICAL,
                summary="Learning rate too high (early training NaN)",
                details=(
                    f"NaN occurred at step {evidence.step} (< 100 steps). "
                    f"Current LR: {evidence.lr:.2e}. "
                    "Early NaN is almost always caused by learning rate being too high."
                ),
                remediation=(
                    f"Reduce learning rate:\n"
                    f"1. Try LR = {evidence.lr * 0.1:.2e} (10x lower)\n"
                    f"2. Use learning rate warmup: `warmup_steps=100`\n"
                    f"3. Enable gradient clipping: `max_grad_norm=1.0`"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
            )

        # Priority 3: Gradient explosion
        if evidence.exploding_gradients:
            worst_layer, worst_norm = max(evidence.exploding_gradients, key=lambda x: x[1])
            return Diagnosis(
                id="nan_gradient_explosion",
                category=DiagnosisCategory.GRADIENT,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"Gradient explosion in {worst_layer}",
                details=(
                    f"Gradient norm exploded to {worst_norm:.2e} in layer `{worst_layer}`. "
                    f"Affected layers: {len(evidence.exploding_gradients)}. "
                    "Gradients this large cause numerical overflow."
                ),
                remediation=(
                    f"Add gradient clipping:\n"
                    f"1. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`\n"
                    f"2. Reduce learning rate: {evidence.lr:.2e} → {evidence.lr * 0.5:.2e}\n"
                    f"3. Add weight decay: `weight_decay=0.01`"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
                layer=worst_layer,
            )

        # Priority 4: NaN in gradients (but not explosion)
        if evidence.nan_in_gradients:
            first_nan_layer = evidence.nan_in_gradients[0]
            return Diagnosis(
                id="nan_gradient_propagation",
                category=DiagnosisCategory.GRADIENT,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"NaN gradient started at {first_nan_layer}",
                details=(
                    f"NaN first appeared in gradient of layer `{first_nan_layer}`. "
                    f"Total layers with NaN gradients: {len(evidence.nan_in_gradients)}. "
                    "This likely indicates a non-differentiable operation or numerical instability."
                ),
                remediation=(
                    f"Check layer `{first_nan_layer}` for:\n"
                    "1. Division by zero (add epsilon: `x / (y + 1e-8)`)\n"
                    "2. Log of zero (use `torch.log(x + 1e-8)`)\n"
                    "3. Square root of negative (use `torch.sqrt(torch.clamp(x, min=0))`)"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
                layer=first_nan_layer,
            )

        # Priority 5: Activation explosion
        if evidence.large_activations:
            worst_layer, worst_val = max(evidence.large_activations, key=lambda x: x[1])
            return Diagnosis(
                id="nan_activation_explosion",
                category=DiagnosisCategory.GRADIENT,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"Activation explosion in {worst_layer}",
                details=(
                    f"Activation values reached {worst_val:.2e} in layer `{worst_layer}`. "
                    f"Values > 1e4 often cause overflow in subsequent operations."
                ),
                remediation=(
                    f"Stabilize activations in `{worst_layer}`:\n"
                    "1. Add LayerNorm or BatchNorm before this layer\n"
                    "2. Use activation function that bounds outputs (tanh, sigmoid)\n"
                    "3. Initialize weights smaller: `torch.nn.init.xavier_uniform_()`"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
                layer=worst_layer,
            )

        # Priority 6: fp16 numerical instability
        if evidence.using_fp16:
            return Diagnosis(
                id="nan_fp16_instability",
                category=DiagnosisCategory.HARDWARE,
                severity=DiagnosticSeverity.CRITICAL,
                summary="fp16 numerical instability",
                details=(
                    "Model is using fp16/bf16 precision. "
                    "Half-precision has limited dynamic range and can overflow/underflow more easily."
                ),
                remediation=(
                    "Options for fp16 stability:\n"
                    "1. Enable gradient scaling: `scaler = torch.cuda.amp.GradScaler()`\n"
                    "2. Use bf16 instead of fp16 (better for gradients)\n"
                    "3. Switch to fp32 for debugging, then optimize\n"
                    "4. Reduce learning rate specifically for fp16"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
            )

        # Priority 7: log(0) or similar
        if evidence.has_log_ops:
            return Diagnosis(
                id="nan_log_instability",
                category=DiagnosisCategory.LOSS,
                severity=DiagnosticSeverity.CRITICAL,
                summary="Likely log(0) in loss computation",
                details=(
                    "Model uses log-based operations (LogSoftmax, NLLLoss, etc.). "
                    "log(0) = -inf, which propagates as NaN through gradients."
                ),
                remediation=(
                    "Add epsilon to log operations:\n"
                    "1. Use `log_softmax` instead of `log(softmax(x))`\n"
                    "2. Add clipping: `torch.log(torch.clamp(x, min=1e-8))`\n"
                    "3. Use label smoothing: `label_smoothing=0.1`"
                ),
                evidence=evidence.__dict__,
                step=evidence.step,
            )

        # Priority 8: Loss spike (sudden increase before NaN)
        if len(evidence.loss_history) > 10:
            recent = evidence.loss_history[-10:]
            older = evidence.loss_history[-20:-10] if len(evidence.loss_history) > 20 else evidence.loss_history[:10]
            if older and recent:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                if recent_avg > older_avg * self.thresholds.loss_spike_threshold:
                    return Diagnosis(
                        id="nan_loss_spike",
                        category=DiagnosisCategory.CONVERGENCE,
                        severity=DiagnosticSeverity.CRITICAL,
                        summary="Loss spiked before NaN",
                        details=(
                            f"Loss increased {recent_avg / older_avg:.1f}x before going NaN. "
                            f"Recent avg: {recent_avg:.4f}, Previous avg: {older_avg:.4f}. "
                            "This suggests instability that accumulated over several steps."
                        ),
                        remediation=(
                            "Address the instability:\n"
                            f"1. Reduce learning rate: {evidence.lr:.2e} → {evidence.lr * 0.3:.2e}\n"
                            "2. Add gradient clipping: `max_grad_norm=1.0`\n"
                            "3. Resume from earlier checkpoint\n"
                            f"4. Last healthy step was {self.last_healthy_step}"
                        ),
                        evidence=evidence.__dict__,
                        step=evidence.step,
                    )

        # Fallback: Unknown cause
        return Diagnosis(
            id="nan_unknown_cause",
            category=DiagnosisCategory.LOSS,
            severity=DiagnosticSeverity.CRITICAL,
            summary="NaN from unknown cause",
            details=(
                f"NaN detected at step {evidence.step} but root cause unclear. "
                "Evidence collected but no definitive pattern matched."
            ),
            remediation=(
                "General debugging steps:\n"
                f"1. Resume from checkpoint before step {self.last_healthy_step}\n"
                f"2. Reduce learning rate: {evidence.lr:.2e} → {evidence.lr * 0.1:.2e}\n"
                "3. Enable gradient clipping: `max_grad_norm=1.0`\n"
                "4. Check for division by zero in custom code\n"
                "5. Try fp32 precision for debugging"
            ),
            evidence=evidence.__dict__,
            step=evidence.step,
        )

    def get_health_score(self) -> float:
        """Get current health score (0-1) based on loss history."""
        if len(self.loss_history) < 2:
            return 1.0  # Not enough data

        losses = [l for _, l in self.loss_history]

        # Check for any NaN/Inf in history
        valid_losses = [l for l in losses if not math.isnan(l) and not math.isinf(l)]
        if len(valid_losses) < len(losses):
            return 0.0  # Had NaN/Inf

        # Check for stability
        if len(valid_losses) > 10:
            recent = valid_losses[-10:]
            std = (sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)) ** 0.5
            mean = sum(recent) / len(recent)

            if mean > 0:
                cv = std / mean  # Coefficient of variation
                return max(0.0, 1.0 - cv)  # High CV = low health

        return 1.0
