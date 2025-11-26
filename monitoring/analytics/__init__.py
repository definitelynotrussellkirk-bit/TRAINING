#!/usr/bin/env python3
"""
Training Analytics Suite

Advanced analytics for understanding what's happening inside the model during training.
Runs on RTX 3090 as a "training analyst" for the 4090.

Modules:
    - layer_drift_monitor: Track which layers are changing (weight-only)
    - parameter_stability: Monitor weight norms for pathologies
    - data_file_impact: Measure per-file training impact
    - skill_map_monitor: Multi-skill progress tracking

All modules write to status/*.json and integrate with the unified API.
"""

from .layer_drift_monitor import LayerDriftMonitor
from .parameter_stability import ParameterStabilityMonitor
from .data_file_impact import DataFileImpactAnalyzer

__all__ = [
    'LayerDriftMonitor',
    'ParameterStabilityMonitor',
    'DataFileImpactAnalyzer',
]
