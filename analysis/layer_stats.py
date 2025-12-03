"""
Layer Statistics Computation

The core of Model Archaeology - computes:
- Per-layer weight norms (L2, Frobenius)
- Per-submodule breakdown (q_proj, k_proj, etc.)
- Activation statistics over probe dataset
- Drift vs reference checkpoint

Usage:
    from analysis.layer_stats import run_layer_stats_analysis
    from analysis.probe_datasets import get_default_probes

    result = run_layer_stats_analysis(
        checkpoint_path="/path/to/checkpoint",
        campaign_id="campaign-001",
        hero_id="dio-qwen3-0.6b",
        probe_sequences=get_default_probes(),
        reference_checkpoint_path="/path/to/reference",
    )
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from .schemas import (
    LayerWeightStats,
    LayerActivationStats,
    LayerDriftStats,
    GlobalStats,
    ProbeInfo,
    LayerStatsResult,
)

logger = logging.getLogger("analysis.layer_stats")


# =============================================================================
# SUBMODULE PATTERNS
# =============================================================================

# Map submodule names to layer types
SUBMODULE_TYPE_MAP = {
    # Attention
    "q_proj": "attention",
    "k_proj": "attention",
    "v_proj": "attention",
    "o_proj": "attention",
    "qkv_proj": "attention",
    "self_attn": "attention",

    # MLP
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
    "mlp": "mlp",
    "fc1": "mlp",
    "fc2": "mlp",

    # Embeddings
    "embed_tokens": "embedding",
    "wte": "embedding",
    "wpe": "embedding",

    # Output
    "lm_head": "output",

    # Norms
    "input_layernorm": "norm",
    "post_attention_layernorm": "norm",
    "ln_1": "norm",
    "ln_2": "norm",
    "ln_f": "norm",
    "norm": "norm",
    "layernorm": "norm",
}


def get_layer_type(name: str) -> str:
    """Determine layer type from parameter name."""
    name_lower = name.lower()

    for pattern, layer_type in SUBMODULE_TYPE_MAP.items():
        if pattern in name_lower:
            return layer_type

    return "other"


def get_layer_name(param_name: str) -> str:
    """
    Extract the containing layer name from a parameter name.

    'model.layers.0.self_attn.q_proj.weight' -> 'model.layers.0'
    'model.embed_tokens.weight' -> 'model.embed_tokens'
    """
    parts = param_name.split(".")

    # Find 'layers.N' pattern
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return ".".join(parts[: i + 2])

    # Otherwise, use all but the last 1-2 parts
    if len(parts) >= 3:
        return ".".join(parts[:-2])
    elif len(parts) >= 2:
        return ".".join(parts[:-1])
    else:
        return param_name


def get_submodule_name(param_name: str) -> str:
    """
    Extract the submodule identifier from parameter name.

    'model.layers.0.self_attn.q_proj.weight' -> 'self_attn.q_proj'
    """
    parts = param_name.split(".")

    # Skip 'model.layers.N' prefix
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            remaining = parts[i + 2:]
            if remaining:
                return ".".join(remaining[:-1]) if remaining[-1] in ("weight", "bias") else ".".join(remaining)
            break

    # Default: last 2 parts before weight/bias
    if parts[-1] in ("weight", "bias"):
        return parts[-2] if len(parts) >= 2 else param_name
    return parts[-1]


# =============================================================================
# WEIGHT STATISTICS
# =============================================================================

def compute_weight_stats(
    model: torch.nn.Module,
    model_ref: str = "unknown",
) -> Tuple[Dict[str, LayerWeightStats], GlobalStats]:
    """
    Compute per-layer weight statistics.

    Args:
        model: The loaded model
        model_ref: Model reference string (for layer name patterns)

    Returns:
        Tuple of (layer_stats_dict, global_stats)
    """
    layer_stats: Dict[str, LayerWeightStats] = {}
    all_norms: List[float] = []
    total_params = 0

    for name, param in model.named_parameters():
        # Skip non-trainable params
        if not param.requires_grad:
            continue

        # Get layer info
        layer_name = get_layer_name(name)
        submodule = get_submodule_name(name)
        layer_type = get_layer_type(name)

        # Compute L2 norm
        with torch.no_grad():
            norm = param.data.float().norm(2).item()

        all_norms.append(norm)

        # Initialize layer if needed
        if layer_name not in layer_stats:
            layer_stats[layer_name] = LayerWeightStats(
                name=layer_name,
                layer_type=layer_type,
            )

        stats = layer_stats[layer_name]

        # Add to appropriate category
        if name.endswith(".weight") or "weight" in name:
            stats.weight_norms[submodule] = norm
        elif name.endswith(".bias") or "bias" in name:
            stats.bias_norms[submodule] = norm
        else:
            stats.weight_norms[submodule] = norm

        stats.param_count += param.numel()
        total_params += param.numel()

    # Compute per-layer totals
    for stats in layer_stats.values():
        if stats.weight_norms:
            stats.total_norm = sum(stats.weight_norms.values())

    # Compute global stats
    global_stats = GlobalStats(
        avg_weight_norm=sum(all_norms) / len(all_norms) if all_norms else 0,
        max_weight_norm=max(all_norms) if all_norms else 0,
        min_weight_norm=min(all_norms) if all_norms else 0,
        total_params=total_params,
    )

    logger.info(f"Computed weight stats for {len(layer_stats)} layers, {total_params/1e6:.1f}M params")

    return layer_stats, global_stats


# =============================================================================
# ACTIVATION STATISTICS
# =============================================================================

def compute_activation_stats(
    model: torch.nn.Module,
    tokenizer,
    probe_sequences: List[str],
    max_tokens: int = 4096,
    device: str = "cuda",
) -> Tuple[Dict[str, LayerActivationStats], ProbeInfo]:
    """
    Compute activation statistics by running probes through the model.

    Uses forward hooks to capture layer outputs without storing full activations.

    Args:
        model: The loaded model (in eval mode)
        tokenizer: Tokenizer for encoding probes
        probe_sequences: List of probe strings
        max_tokens: Max total tokens to process
        device: Device to run on

    Returns:
        Tuple of (activation_stats_dict, probe_info)
    """
    model.eval()
    activation_stats: Dict[str, LayerActivationStats] = {}

    # Running statistics accumulators (Welford's online algorithm)
    accumulators: Dict[str, Dict[str, Any]] = {}
    hooks = []

    def make_hook(name: str):
        """Create a hook that accumulates statistics without storing all activations."""
        def hook(module, input, output):
            # Get output tensor
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            # Skip if not a tensor
            if not isinstance(out, torch.Tensor):
                return

            # Flatten and convert to float for accurate stats
            flat = out.detach().float().flatten()

            # Initialize accumulator
            if name not in accumulators:
                accumulators[name] = {
                    "count": 0,
                    "mean": 0.0,
                    "M2": 0.0,  # For variance
                    "max": float("-inf"),
                    "min": float("inf"),
                    "zero_count": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                }

            acc = accumulators[name]
            n = flat.numel()

            # Update running statistics (Welford's algorithm)
            for val in [flat.mean().item()]:  # Use batch mean for efficiency
                acc["count"] += n
                delta = val - acc["mean"]
                acc["mean"] += delta * n / acc["count"]
                delta2 = val - acc["mean"]
                acc["M2"] += delta * delta2 * n

            # Update extremes
            acc["max"] = max(acc["max"], flat.max().item())
            acc["min"] = min(acc["min"], flat.min().item())

            # Count special values
            acc["zero_count"] += (flat == 0).sum().item()
            acc["nan_count"] += torch.isnan(flat).sum().item()
            acc["inf_count"] += torch.isinf(flat).sum().item()

        return hook

    # Register hooks on transformer layers
    for name, module in model.named_modules():
        # Hook into layer outputs (model.layers.N)
        if re.match(r".*layers\.\d+$", name):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Process probes
    total_tokens = 0
    num_sequences = 0

    logger.info(f"Running {len(probe_sequences)} probes through model...")

    with torch.no_grad():
        for seq in probe_sequences:
            if total_tokens >= max_tokens:
                break

            try:
                inputs = tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                model(**inputs)

                seq_tokens = inputs["input_ids"].shape[1]
                total_tokens += seq_tokens
                num_sequences += 1

            except Exception as e:
                logger.warning(f"Failed to process probe: {e}")
                continue

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Convert accumulators to LayerActivationStats
    for name, acc in accumulators.items():
        if acc["count"] == 0:
            continue

        # Compute standard deviation from M2
        variance = acc["M2"] / acc["count"] if acc["count"] > 1 else 0
        std = variance ** 0.5

        activation_stats[name] = LayerActivationStats(
            name=name,
            mean=acc["mean"],
            std=std,
            max=acc["max"],
            min=acc["min"],
            zero_fraction=acc["zero_count"] / acc["count"] if acc["count"] > 0 else 0,
            nan_fraction=acc["nan_count"] / acc["count"] if acc["count"] > 0 else 0,
            inf_fraction=acc["inf_count"] / acc["count"] if acc["count"] > 0 else 0,
        )

    probe_info = ProbeInfo(
        dataset_id="inline",
        num_sequences=num_sequences,
        total_tokens=total_tokens,
        source_file="probes",
        avg_sequence_length=total_tokens / num_sequences if num_sequences > 0 else 0,
    )

    logger.info(f"Computed activation stats for {len(activation_stats)} layers")
    logger.info(f"  Processed {num_sequences} sequences, {total_tokens} tokens")

    return activation_stats, probe_info


# =============================================================================
# DRIFT COMPUTATION
# =============================================================================

def compute_weight_drift(
    model: torch.nn.Module,
    reference_state_dict: Dict[str, torch.Tensor],
    model_ref: str = "unknown",
) -> Tuple[Dict[str, LayerDriftStats], GlobalStats]:
    """
    Compute weight drift between current model and reference.

    Args:
        model: Current model
        reference_state_dict: State dict from reference checkpoint
        model_ref: Model reference string

    Returns:
        Tuple of (drift_stats_dict, global_drift_stats)
    """
    drift_stats: Dict[str, LayerDriftStats] = {}
    all_l2_drifts: List[float] = []
    all_cosine_sims: List[float] = []

    for name, param in model.named_parameters():
        if name not in reference_state_dict:
            logger.debug(f"Skipping {name} - not in reference")
            continue

        ref_param = reference_state_dict[name]

        # Ensure same device and dtype
        ref_param = ref_param.to(param.device).to(param.dtype)

        with torch.no_grad():
            # Compute L2 distance
            diff = param.data - ref_param
            l2_diff = diff.float().norm(2).item()
            all_l2_drifts.append(l2_diff)

            # Compute cosine similarity (flatten both)
            flat_cur = param.data.float().flatten()
            flat_ref = ref_param.float().flatten()

            # Avoid division by zero
            norm_cur = flat_cur.norm(2)
            norm_ref = flat_ref.norm(2)

            if norm_cur > 1e-8 and norm_ref > 1e-8:
                cosine = F.cosine_similarity(
                    flat_cur.unsqueeze(0),
                    flat_ref.unsqueeze(0)
                ).item()
            else:
                cosine = 1.0  # Treat zero vectors as identical

            all_cosine_sims.append(cosine)

        # Get layer name and submodule
        layer_name = get_layer_name(name)
        submodule = get_submodule_name(name)

        if layer_name not in drift_stats:
            drift_stats[layer_name] = LayerDriftStats(name=layer_name)

        drift_stats[layer_name].weight_l2[submodule] = l2_diff
        drift_stats[layer_name].weight_cosine[submodule] = cosine

    # Compute per-layer totals
    for layer_name, stats in drift_stats.items():
        if stats.weight_l2:
            stats.total_l2 = sum(stats.weight_l2.values())
        if stats.weight_cosine:
            stats.avg_cosine = sum(stats.weight_cosine.values()) / len(stats.weight_cosine)

    # Find most/least changed layers
    sorted_by_drift = sorted(
        [(name, stats.total_l2) for name, stats in drift_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )

    most_changed = sorted_by_drift[0] if sorted_by_drift else (None, 0)
    least_changed = sorted_by_drift[-1] if sorted_by_drift else (None, 0)

    global_stats = GlobalStats(
        avg_weight_norm=0,  # Not used for drift
        max_weight_norm=0,
        min_weight_norm=0,
        avg_drift_l2=sum(all_l2_drifts) / len(all_l2_drifts) if all_l2_drifts else 0,
        max_drift_l2=max(all_l2_drifts) if all_l2_drifts else 0,
        min_drift_l2=min(all_l2_drifts) if all_l2_drifts else 0,
        avg_cosine_similarity=sum(all_cosine_sims) / len(all_cosine_sims) if all_cosine_sims else 1.0,
        most_changed_layer=most_changed[0],
        most_changed_drift=most_changed[1],
        least_changed_layer=least_changed[0],
        least_changed_drift=least_changed[1],
    )

    logger.info(f"Computed drift for {len(drift_stats)} layers")
    logger.info(f"  Most changed: {global_stats.most_changed_layer} (L2={global_stats.most_changed_drift:.4f})")
    logger.info(f"  Least changed: {global_stats.least_changed_layer} (L2={global_stats.least_changed_drift:.6f})")

    return drift_stats, global_stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_layer_stats_analysis(
    checkpoint_path: str,
    campaign_id: str,
    hero_id: str,
    model_ref: str = "qwen3-0.6b",
    reference_checkpoint_path: Optional[str] = None,
    probe_sequences: Optional[List[str]] = None,
    max_probe_tokens: int = 4096,
    compute_activations: bool = True,
    device: str = "cuda",
) -> LayerStatsResult:
    """
    Run complete layer stats analysis.

    This is the main entry point for the layer_stats job.

    Args:
        checkpoint_path: Path to checkpoint to analyze
        campaign_id: Campaign identifier
        hero_id: Hero identifier
        model_ref: Model reference for architecture detection
        reference_checkpoint_path: Optional reference for drift
        probe_sequences: Optional probe dataset for activations
        max_probe_tokens: Max tokens for activation probing
        compute_activations: Whether to compute activation stats
        device: Device to use

    Returns:
        LayerStatsResult with all computed statistics
    """
    from .model_loader import load_model_for_analysis, load_tokenizer, load_reference_state_dict

    start_time = time.time()

    result = LayerStatsResult(
        campaign_id=campaign_id,
        hero_id=hero_id,
        checkpoint_path=checkpoint_path,
        model_ref=model_ref,
        device=device,
    )

    # Extract step from checkpoint path
    ckpt_name = Path(checkpoint_path).name
    step_match = re.search(r"checkpoint-(\d+)", ckpt_name)
    if step_match:
        result.checkpoint_step = int(step_match.group(1))
    else:
        # Try to find step in path
        for part in Path(checkpoint_path).parts:
            if part.startswith("checkpoint-"):
                step_match = re.search(r"(\d+)", part)
                if step_match:
                    result.checkpoint_step = int(step_match.group(1))
                    break

    logger.info(f"Analyzing checkpoint step {result.checkpoint_step}")

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model_for_analysis(checkpoint_path, model_ref, device=device)

    # Compute weight stats (always)
    logger.info("Computing weight statistics...")
    weight_stats, global_weight = compute_weight_stats(model, model_ref)

    # Convert to dicts for JSON serialization
    result.weight_stats = {k: v.to_dict() for k, v in weight_stats.items()}
    result.global_weight_stats = global_weight.to_dict()

    # Compute activation stats (optional)
    if compute_activations and probe_sequences:
        logger.info("Computing activation statistics...")
        tokenizer = load_tokenizer(model_ref)
        act_stats, probe_info = compute_activation_stats(
            model, tokenizer, probe_sequences, max_probe_tokens, device
        )
        result.activation_stats = {k: v.to_dict() for k, v in act_stats.items()}
        result.probe_info = probe_info.to_dict()

    # Compute drift (optional)
    if reference_checkpoint_path:
        logger.info(f"Computing drift vs {reference_checkpoint_path}")

        # Extract reference step
        ref_match = re.search(r"checkpoint-(\d+)", reference_checkpoint_path)
        if ref_match:
            result.reference_checkpoint_step = int(ref_match.group(1))

        ref_state = load_reference_state_dict(reference_checkpoint_path, device=device)
        drift_stats, global_drift = compute_weight_drift(model, ref_state, model_ref)

        result.drift_stats = {k: v.to_dict() for k, v in drift_stats.items()}
        result.global_drift_stats = global_drift.to_dict()
        result.reference_checkpoint_path = reference_checkpoint_path

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    result.compute_duration_sec = time.time() - start_time
    logger.info(f"Analysis complete in {result.compute_duration_sec:.1f}s")

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    # Get default hero from active campaign
    def _get_active_hero_id():
        try:
            from core.hero import get_active_campaign
            campaign = get_active_campaign()
            return campaign.get("hero_id", "dio-qwen3-0.6b")
        except Exception:
            return "dio-qwen3-0.6b"

    default_hero = _get_active_hero_id()

    parser = argparse.ArgumentParser(description="Run layer stats analysis")
    parser.add_argument("checkpoint", help="Checkpoint path")
    parser.add_argument("--campaign", default="campaign-001")
    parser.add_argument("--hero", default=default_hero)
    parser.add_argument("--model-ref", default="qwen3-0.6b")
    parser.add_argument("--reference", help="Reference checkpoint for drift")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-activations", action="store_true")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get probes
    from .probe_datasets import get_default_probes
    probes = None if args.no_activations else get_default_probes()

    result = run_layer_stats_analysis(
        checkpoint_path=args.checkpoint,
        campaign_id=args.campaign,
        hero_id=args.hero,
        model_ref=args.model_ref,
        reference_checkpoint_path=args.reference,
        probe_sequences=probes,
        compute_activations=not args.no_activations,
        device=args.device,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.to_json())
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(result.summary(), indent=2))
