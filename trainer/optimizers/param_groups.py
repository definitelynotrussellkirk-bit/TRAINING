"""
Parameter Grouping for Muon Optimizer

Splits transformer parameters into groups for Muon (hidden weights) and AdamW (rest).

Muon should only be applied to:
- Attention projection matrices (q_proj, k_proj, v_proj, o_proj)
- MLP/FFN matrices (gate_proj, up_proj, down_proj, fc1, fc2)

AdamW should be used for:
- Embeddings (embed_tokens, wte, wpe)
- Output heads (lm_head)
- Layer normalization weights
- All biases (1D parameters)
"""

import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Patterns that indicate a parameter should NOT use Muon
EXCLUDE_PATTERNS = [
    "embed",       # embed_tokens, token_embedding, word_embedding
    "wte",         # GPT-style token embedding
    "wpe",         # GPT-style position embedding
    "lm_head",     # Output projection
    "head",        # Various output heads
    "norm",        # LayerNorm, RMSNorm
    "layernorm",   # Alternative naming
    "ln_",         # GPT-style layer norm
    "bias",        # Any bias terms
]


@dataclass
class ParamGroupSummary:
    """Summary of parameter grouping."""
    muon_params: int
    muon_tensors: int
    adam_params: int
    adam_tensors: int
    muon_names: List[str]
    adam_names: List[str]


def should_use_muon(name: str, param: torch.Tensor) -> bool:
    """
    Determine if a parameter should use Muon optimizer.

    Args:
        name: Parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
        param: The parameter tensor

    Returns:
        True if parameter should use Muon, False for AdamW
    """
    # Must be at least 2D (matrix)
    if param.ndim < 2:
        return False

    # Check exclusion patterns
    name_lower = name.lower()
    for pattern in EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return False

    return True


def split_transformer_params(
    model: nn.Module,
    hidden_lr: float = 0.02,
    aux_lr: float = 3e-4,
    hidden_momentum: float = 0.95,
    weight_decay: float = 0.0,
    aux_betas: Tuple[float, float] = (0.9, 0.95),
    aux_eps: float = 1e-10,
) -> List[Dict[str, Any]]:
    """
    Split model parameters into Muon and AdamW groups.

    Args:
        model: The model to optimize
        hidden_lr: Learning rate for hidden weights (Muon) - default 0.02
        aux_lr: Learning rate for other params (AdamW) - default 3e-4
        hidden_momentum: Momentum for Muon - default 0.95
        weight_decay: Weight decay for all params - default 0.0
        aux_betas: Beta coefficients for AdamW - default (0.9, 0.95)
        aux_eps: Epsilon for AdamW - default 1e-10

    Returns:
        List of param_groups suitable for SingleDeviceMuonWithAuxAdam

    Example:
        param_groups = split_transformer_params(model, hidden_lr=0.02, aux_lr=3e-4)
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    """
    muon_params = []
    adam_params = []
    muon_names = []
    adam_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if should_use_muon(name, param):
            muon_params.append(param)
            muon_names.append(name)
        else:
            adam_params.append(param)
            adam_names.append(name)

    # Log the split
    muon_total = sum(p.numel() for p in muon_params)
    adam_total = sum(p.numel() for p in adam_params)
    total = muon_total + adam_total

    logger.info(
        f"Parameter split: Muon {muon_total:,} ({100*muon_total/total:.1f}%), "
        f"AdamW {adam_total:,} ({100*adam_total/total:.1f}%)"
    )

    param_groups = []

    if muon_params:
        param_groups.append({
            "params": muon_params,
            "use_muon": True,
            "lr": hidden_lr,
            "momentum": hidden_momentum,
            "weight_decay": weight_decay,
        })

    if adam_params:
        param_groups.append({
            "params": adam_params,
            "use_muon": False,
            "lr": aux_lr,
            "betas": aux_betas,
            "eps": aux_eps,
            "weight_decay": weight_decay,
        })

    return param_groups


def get_param_group_summary(model: nn.Module) -> ParamGroupSummary:
    """
    Get a detailed summary of how parameters would be split.

    Useful for debugging and understanding the optimizer configuration.

    Args:
        model: The model to analyze

    Returns:
        ParamGroupSummary with counts and names
    """
    muon_params = 0
    muon_tensors = 0
    adam_params = 0
    adam_tensors = 0
    muon_names = []
    adam_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if should_use_muon(name, param):
            muon_params += param.numel()
            muon_tensors += 1
            muon_names.append(name)
        else:
            adam_params += param.numel()
            adam_tensors += 1
            adam_names.append(name)

    return ParamGroupSummary(
        muon_params=muon_params,
        muon_tensors=muon_tensors,
        adam_params=adam_params,
        adam_tensors=adam_tensors,
        muon_names=muon_names,
        adam_names=adam_names,
    )


def print_param_group_summary(model: nn.Module) -> None:
    """Print a human-readable summary of parameter grouping."""
    summary = get_param_group_summary(model)
    total_params = summary.muon_params + summary.adam_params
    total_tensors = summary.muon_tensors + summary.adam_tensors

    print("\n" + "=" * 60)
    print("MUON OPTIMIZER - PARAMETER GROUPING")
    print("=" * 60)

    print(f"\nMuon (hidden weights): {summary.muon_params:,} params ({summary.muon_tensors} tensors)")
    print(f"AdamW (other):         {summary.adam_params:,} params ({summary.adam_tensors} tensors)")
    print(f"Total:                 {total_params:,} params ({total_tensors} tensors)")

    muon_pct = 100 * summary.muon_params / total_params if total_params > 0 else 0
    print(f"\nMuon coverage: {muon_pct:.1f}% of parameters")

    print("\n--- Muon Parameters (sample) ---")
    for name in summary.muon_names[:10]:
        print(f"  {name}")
    if len(summary.muon_names) > 10:
        print(f"  ... and {len(summary.muon_names) - 10} more")

    print("\n--- AdamW Parameters (sample) ---")
    for name in summary.adam_names[:10]:
        print(f"  {name}")
    if len(summary.adam_names) > 10:
        print(f"  ... and {len(summary.adam_names) - 10} more")

    print("=" * 60 + "\n")
