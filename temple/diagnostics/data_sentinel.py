"""
Data Sentinel - Batch Quality Checking
======================================

The Data Sentinel inspects every batch for quality issues:
- NaN/Inf values (poison pills)
- Abnormal value ranges
- Zero variance (dead batches)
- Tokenization issues
- Label distribution problems

The key insight: Bad data causes silent failures that are hard to debug later.
Catch them at the source.

Usage:
    from temple.diagnostics import DataSentinel

    sentinel = DataSentinel()

    # In data loading
    diagnoses = sentinel.check_batch(batch, labels=labels)
    if diagnoses:
        print(f"Data issue: {diagnoses[0].summary}")
"""

from __future__ import annotations

import logging
import math
from collections import deque
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

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    """Statistics from a batch check."""
    step: int
    has_nan: bool
    has_inf: bool
    max_value: float
    min_value: float
    mean: float
    std: float
    shape: Tuple[int, ...]
    dtype: str
    nan_count: int = 0
    inf_count: int = 0


class DataSentinel:
    """
    Monitors data quality and catches problematic batches.

    Inspects input tensors for common issues that can cause
    training instability or failure.
    """

    # Common problematic token IDs (model-specific, these are examples)
    SUSPICIOUS_TOKENS = {
        0: "PAD",       # Padding - shouldn't dominate
        1: "UNK",       # Unknown - indicates vocab mismatch
        2: "BOS",       # Beginning of sequence
        3: "EOS",       # End of sequence
    }

    def __init__(self, thresholds: Optional[DiagnosticThresholds] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.history: deque = deque(maxlen=100)
        self.bad_batch_count: int = 0
        self.total_batch_count: int = 0
        self.cumulative_nan_count: int = 0
        self.cumulative_inf_count: int = 0

    def check_batch(
        self,
        batch: "torch.Tensor",
        labels: Optional["torch.Tensor"] = None,
        attention_mask: Optional["torch.Tensor"] = None,
        step: int = 0,
    ) -> List[Diagnosis]:
        """
        Check a batch for quality issues.

        Args:
            batch: Input tensor (typically input_ids or embeddings)
            labels: Optional label tensor
            attention_mask: Optional attention mask
            step: Current training step

        Returns:
            List of Diagnosis objects for detected issues
        """
        import torch

        diagnoses = []
        self.total_batch_count += 1

        # Basic tensor checks
        batch_diagnoses = self._check_tensor(batch, "batch", step)
        diagnoses.extend(batch_diagnoses)

        if labels is not None:
            label_diagnoses = self._check_tensor(labels, "labels", step)
            diagnoses.extend(label_diagnoses)

        if attention_mask is not None:
            mask_diagnoses = self._check_attention_mask(attention_mask, step)
            diagnoses.extend(mask_diagnoses)

        # Check for integer inputs (token IDs)
        if batch.dtype in [torch.int32, torch.int64, torch.long]:
            token_diagnoses = self._check_token_ids(batch, step)
            diagnoses.extend(token_diagnoses)

        # Track bad batches
        if diagnoses and any(d.severity >= DiagnosticSeverity.WARN for d in diagnoses):
            self.bad_batch_count += 1

        return diagnoses

    def _check_tensor(
        self,
        tensor: "torch.Tensor",
        name: str,
        step: int,
    ) -> List[Diagnosis]:
        """Check a tensor for common issues."""
        import torch

        diagnoses = []

        # Convert to float for stats if needed
        if tensor.is_floating_point():
            float_tensor = tensor.float()
        else:
            float_tensor = tensor.float()

        # Check for NaN
        nan_mask = torch.isnan(float_tensor)
        nan_count = nan_mask.sum().item()
        has_nan = nan_count > 0

        if has_nan:
            self.cumulative_nan_count += nan_count
            diagnoses.append(Diagnosis(
                id=f"data_nan_{name}",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"NaN in {name}: {nan_count} values",
                details=(
                    f"Found {nan_count} NaN values in {name} tensor "
                    f"(shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}). "
                    "NaN will propagate through computation and corrupt training."
                ),
                remediation=(
                    f"Fix NaN in {name}:\n"
                    "1. Check data preprocessing - division by zero?\n"
                    "2. Check data loading - corrupt file?\n"
                    "3. Add NaN filtering: `batch = batch[~torch.isnan(batch).any(dim=-1)]`\n"
                    "4. Replace NaN: `batch = torch.nan_to_num(batch, nan=0.0)`"
                ),
                evidence={
                    "nan_count": nan_count,
                    "total_elements": tensor.numel(),
                    "nan_ratio": nan_count / tensor.numel(),
                    "shape": tuple(tensor.shape),
                },
                step=step,
            ))

        # Check for Inf
        inf_mask = torch.isinf(float_tensor)
        inf_count = inf_mask.sum().item()
        has_inf = inf_count > 0

        if has_inf:
            self.cumulative_inf_count += inf_count
            diagnoses.append(Diagnosis(
                id=f"data_inf_{name}",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.CRITICAL,
                summary=f"Inf in {name}: {inf_count} values",
                details=(
                    f"Found {inf_count} Inf values in {name} tensor. "
                    "Infinity will cause numerical overflow."
                ),
                remediation=(
                    f"Fix Inf in {name}:\n"
                    "1. Check for overflow in preprocessing\n"
                    "2. Clamp values: `batch = torch.clamp(batch, min=-1e6, max=1e6)`\n"
                    "3. Check for log(0) or exp(large) in data pipeline"
                ),
                evidence={
                    "inf_count": inf_count,
                    "total_elements": tensor.numel(),
                },
                step=step,
            ))

        # Skip remaining checks if NaN/Inf
        if has_nan or has_inf:
            return diagnoses

        # Check value range
        if tensor.is_floating_point():
            max_val = float_tensor.abs().max().item()
            if max_val > self.thresholds.data_max_value:
                diagnoses.append(Diagnosis(
                    id=f"data_large_values_{name}",
                    category=DiagnosisCategory.DATA,
                    severity=DiagnosticSeverity.WARN,
                    summary=f"Large values in {name}: max={max_val:.2e}",
                    details=(
                        f"Found values up to {max_val:.2e} in {name}. "
                        f"Values > {self.thresholds.data_max_value} can cause instability. "
                    ),
                    remediation=(
                        f"Normalize {name} data:\n"
                        "1. Apply normalization (BatchNorm, LayerNorm)\n"
                        "2. Scale down: `batch = batch / batch.abs().max()`\n"
                        "3. Check preprocessing for overflow"
                    ),
                    evidence={
                        "max_value": max_val,
                        "threshold": self.thresholds.data_max_value,
                    },
                    step=step,
                ))

            # Check variance
            if tensor.numel() > 1:
                std = float_tensor.std().item()
                if std < self.thresholds.data_min_variance:
                    diagnoses.append(Diagnosis(
                        id=f"data_low_variance_{name}",
                        category=DiagnosisCategory.DATA,
                        severity=DiagnosticSeverity.WARN,
                        summary=f"Near-zero variance in {name}",
                        details=(
                            f"Tensor {name} has std={std:.2e}, nearly constant. "
                            "This provides little training signal."
                        ),
                        remediation=(
                            f"Check {name} data:\n"
                            "1. Verify data loading is working correctly\n"
                            "2. Check for all-padding batches\n"
                            "3. Ensure data augmentation is applied"
                        ),
                        evidence={
                            "std": std,
                            "mean": float_tensor.mean().item(),
                            "threshold": self.thresholds.data_min_variance,
                        },
                        step=step,
                    ))

        # Record stats
        stats = BatchStats(
            step=step,
            has_nan=has_nan,
            has_inf=has_inf,
            max_value=float_tensor.max().item() if not has_nan else float('nan'),
            min_value=float_tensor.min().item() if not has_nan else float('nan'),
            mean=float_tensor.mean().item() if not has_nan else float('nan'),
            std=float_tensor.std().item() if not has_nan and tensor.numel() > 1 else 0.0,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            nan_count=nan_count,
            inf_count=inf_count,
        )
        self.history.append(stats)

        return diagnoses

    def _check_attention_mask(
        self,
        mask: "torch.Tensor",
        step: int,
    ) -> List[Diagnosis]:
        """Check attention mask for issues."""
        import torch

        diagnoses = []

        # Attention mask should be 0s and 1s
        unique_vals = torch.unique(mask)
        if len(unique_vals) > 2 or not all(v in [0, 1] for v in unique_vals.tolist()):
            diagnoses.append(Diagnosis(
                id="data_bad_attention_mask",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.ERROR,
                summary="Invalid attention mask values",
                details=(
                    f"Attention mask should contain only 0s and 1s, "
                    f"but found values: {unique_vals.tolist()}"
                ),
                remediation=(
                    "Fix attention mask:\n"
                    "1. Binarize: `mask = (mask > 0).long()`\n"
                    "2. Check tokenizer settings\n"
                    "3. Verify data collation function"
                ),
                evidence={"unique_values": unique_vals.tolist()},
                step=step,
            ))

        # Check if mask is mostly zeros (too much padding)
        padding_ratio = (mask == 0).float().mean().item()
        if padding_ratio > 0.9:
            diagnoses.append(Diagnosis(
                id="data_excessive_padding",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.WARN,
                summary=f"Excessive padding: {padding_ratio:.0%}",
                details=(
                    f"Attention mask is {padding_ratio:.0%} padding. "
                    "This wastes compute and can affect training dynamics."
                ),
                remediation=(
                    "Reduce padding:\n"
                    "1. Use dynamic batching by sequence length\n"
                    "2. Reduce max_length\n"
                    "3. Pack multiple sequences per sample"
                ),
                evidence={"padding_ratio": padding_ratio},
                step=step,
            ))

        return diagnoses

    def _check_token_ids(
        self,
        tokens: "torch.Tensor",
        step: int,
    ) -> List[Diagnosis]:
        """Check token IDs for issues."""
        import torch

        diagnoses = []

        # Check for negative token IDs
        if (tokens < 0).any():
            neg_count = (tokens < 0).sum().item()
            diagnoses.append(Diagnosis(
                id="data_negative_tokens",
                category=DiagnosisCategory.DATA,
                severity=DiagnosticSeverity.ERROR,
                summary=f"Negative token IDs: {neg_count}",
                details=(
                    f"Found {neg_count} negative token IDs. "
                    "Embeddings will fail with negative indices."
                ),
                remediation=(
                    "Fix token IDs:\n"
                    "1. Check tokenizer - is it returning errors as -1?\n"
                    "2. Filter: `tokens = tokens[tokens >= 0]`\n"
                    "3. Check data corruption"
                ),
                evidence={"negative_count": neg_count},
                step=step,
            ))

        # Check for excessive padding tokens
        if tokens.numel() > 0:
            pad_count = (tokens == 0).sum().item()  # Assuming 0 is pad
            pad_ratio = pad_count / tokens.numel()
            if pad_ratio > 0.8:
                diagnoses.append(Diagnosis(
                    id="data_mostly_padding",
                    category=DiagnosisCategory.DATA,
                    severity=DiagnosticSeverity.WARN,
                    summary=f"Batch is {pad_ratio:.0%} padding tokens",
                    details=(
                        f"This batch contains {pad_ratio:.0%} padding tokens. "
                        "Most computation is wasted on padding."
                    ),
                    remediation=(
                        "Reduce padding:\n"
                        "1. Use sequence packing\n"
                        "2. Sort batches by length\n"
                        "3. Reduce max_length"
                    ),
                    evidence={"padding_ratio": pad_ratio},
                    step=step,
                ))

            # Check for excessive UNK tokens
            unk_count = (tokens == 1).sum().item()  # Assuming 1 is UNK
            unk_ratio = unk_count / tokens.numel()
            if unk_ratio > 0.1:
                diagnoses.append(Diagnosis(
                    id="data_many_unk",
                    category=DiagnosisCategory.DATA,
                    severity=DiagnosticSeverity.WARN,
                    summary=f"High UNK token ratio: {unk_ratio:.0%}",
                    details=(
                        f"This batch contains {unk_ratio:.0%} UNK tokens. "
                        "This suggests vocabulary mismatch or corrupt text."
                    ),
                    remediation=(
                        "Fix UNK tokens:\n"
                        "1. Check tokenizer matches training data encoding\n"
                        "2. Verify text preprocessing (special chars, encoding)\n"
                        "3. Consider expanding vocabulary"
                    ),
                    evidence={"unk_ratio": unk_ratio},
                    step=step,
                ))

        return diagnoses

    def get_health_score(self) -> float:
        """
        Get data quality health score (0-1).

        1.0 = All batches clean
        0.0 = Many problematic batches
        """
        if self.total_batch_count == 0:
            return 1.0

        bad_ratio = self.bad_batch_count / self.total_batch_count

        # Any NaN/Inf is a big problem
        if self.cumulative_nan_count > 0 or self.cumulative_inf_count > 0:
            return max(0.0, 0.3 - bad_ratio)

        return max(0.0, 1.0 - bad_ratio * 2)

    def get_summary(self) -> Dict[str, Any]:
        """Get data quality summary."""
        return {
            "total_batches": self.total_batch_count,
            "bad_batches": self.bad_batch_count,
            "bad_batch_ratio": (
                self.bad_batch_count / self.total_batch_count
                if self.total_batch_count > 0 else 0.0
            ),
            "cumulative_nan_count": self.cumulative_nan_count,
            "cumulative_inf_count": self.cumulative_inf_count,
            "health_score": round(self.get_health_score(), 2),
        }
