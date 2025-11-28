"""
Optimizer Module - Muon and AdamW optimizer support

This module provides:
- Muon optimizer (orthogonalized momentum) for hidden layers
- AdamW for embeddings, heads, and biases
- Automatic parameter grouping for transformers
- Config-driven optimizer selection

Usage:
    from trainer.optimizers import create_optimizer

    optimizer = create_optimizer(model, config)
"""

from .factory import create_optimizer, get_optimizer_info
from .param_groups import split_transformer_params, get_param_group_summary
from .muon import (
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
    zeropower_via_newtonschulz5,
    muon_update,
)

__all__ = [
    # Factory
    "create_optimizer",
    "get_optimizer_info",
    # Parameter grouping
    "split_transformer_params",
    "get_param_group_summary",
    # Muon classes
    "SingleDeviceMuon",
    "SingleDeviceMuonWithAuxAdam",
    "zeropower_via_newtonschulz5",
    "muon_update",
]
