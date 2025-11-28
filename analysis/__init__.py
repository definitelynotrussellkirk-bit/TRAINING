"""
Model Archaeology - Interpretability Analysis Module

This module provides tools for analyzing model internals:
- Weight statistics per layer
- Activation statistics over probe datasets
- Weight drift between checkpoints

Usage:
    from analysis import run_layer_stats_analysis
    from analysis.probe_datasets import get_default_probes

    result = run_layer_stats_analysis(
        checkpoint_path="/path/to/checkpoint",
        campaign_id="campaign-001",
        hero_id="dio-qwen3-0.6b",
        probe_sequences=get_default_probes(),
    )

    print(f"Most changed layer: {result.global_drift_stats.most_changed_layer}")
"""

from .schemas import (
    LayerWeightStats,
    LayerActivationStats,
    LayerDriftStats,
    GlobalStats,
    ProbeInfo,
    LayerStatsResult,
)

from .layer_stats import (
    compute_weight_stats,
    compute_activation_stats,
    compute_weight_drift,
    run_layer_stats_analysis,
)

from .probe_datasets import (
    get_default_probes,
    load_probe_dataset,
)

from .model_loader import (
    load_model_for_analysis,
    load_tokenizer,
    load_reference_state_dict,
)

__all__ = [
    # Schemas
    "LayerWeightStats",
    "LayerActivationStats",
    "LayerDriftStats",
    "GlobalStats",
    "ProbeInfo",
    "LayerStatsResult",
    # Core functions
    "compute_weight_stats",
    "compute_activation_stats",
    "compute_weight_drift",
    "run_layer_stats_analysis",
    # Probe datasets
    "get_default_probes",
    "load_probe_dataset",
    # Model loading
    "load_model_for_analysis",
    "load_tokenizer",
    "load_reference_state_dict",
]
