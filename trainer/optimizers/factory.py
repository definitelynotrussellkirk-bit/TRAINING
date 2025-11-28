"""
Optimizer Factory - Config-driven optimizer creation

Creates optimizers based on config.json settings. Supports:
- AdamW (default): Standard AdamW with fused kernels
- Muon: Hybrid Muon+AdamW for transformer training

Usage:
    from trainer.optimizers import create_optimizer

    # With config dict
    optimizer, scheduler = create_optimizer(model, config)

    # Or with explicit type
    optimizer, scheduler = create_optimizer(
        model, config,
        optimizer_type="muon",
        num_training_steps=10000
    )
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .muon import SingleDeviceMuonWithAuxAdam
from .param_groups import split_transformer_params, get_param_group_summary

logger = logging.getLogger(__name__)


@dataclass
class OptimizerInfo:
    """Information about the created optimizer."""
    type: str
    learning_rate: float
    weight_decay: float
    total_params: int
    trainable_params: int
    muon_params: Optional[int] = None
    adam_params: Optional[int] = None
    hidden_lr: Optional[float] = None
    aux_lr: Optional[float] = None


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimizer configuration from config dict.

    Handles both new-style (config.optimizer.*) and legacy (config.learning_rate) formats.
    """
    opt_config = config.get("optimizer", {})

    return {
        "type": opt_config.get("type", "adamw"),
        "adamw": {
            "lr": opt_config.get("adamw", {}).get("lr", config.get("learning_rate", 4e-4)),
            "betas": tuple(opt_config.get("adamw", {}).get("betas", [0.9, 0.999])),
            "eps": opt_config.get("adamw", {}).get("eps", 1e-8),
            "weight_decay": opt_config.get("adamw", {}).get("weight_decay", 0.01),
        },
        "muon": {
            "hidden_lr": opt_config.get("muon", {}).get("hidden_lr", 0.02),
            "aux_lr": opt_config.get("muon", {}).get("aux_lr", 3e-4),
            "momentum": opt_config.get("muon", {}).get("momentum", 0.95),
            "weight_decay": opt_config.get("muon", {}).get("weight_decay", 0.0),
            "aux_betas": tuple(opt_config.get("muon", {}).get("aux_betas", [0.9, 0.95])),
            "aux_eps": opt_config.get("muon", {}).get("aux_eps", 1e-10),
        },
    }


def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 4e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> Tuple[AdamW, OptimizerInfo]:
    """
    Create a standard AdamW optimizer with fused kernels.

    Args:
        model: The model to optimize
        lr: Learning rate
        betas: Beta coefficients
        eps: Epsilon for numerical stability
        weight_decay: Weight decay coefficient

    Returns:
        (optimizer, info) tuple
    """
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params)

    # Try to use fused AdamW for better performance
    try:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=True,
        )
        logger.info(f"Created fused AdamW optimizer (lr={lr})")
    except Exception:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        logger.info(f"Created AdamW optimizer (lr={lr})")

    info = OptimizerInfo(
        type="adamw",
        learning_rate=lr,
        weight_decay=weight_decay,
        total_params=total_params,
        trainable_params=trainable_params,
    )

    return optimizer, info


def create_muon_optimizer(
    model: nn.Module,
    hidden_lr: float = 0.02,
    aux_lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    aux_betas: Tuple[float, float] = (0.9, 0.95),
    aux_eps: float = 1e-10,
) -> Tuple[SingleDeviceMuonWithAuxAdam, OptimizerInfo]:
    """
    Create a Muon optimizer with AdamW auxiliary.

    Muon is applied to hidden weight matrices, AdamW to embeddings/heads/biases.

    Args:
        model: The model to optimize
        hidden_lr: Learning rate for hidden weights (Muon)
        aux_lr: Learning rate for other params (AdamW)
        momentum: Momentum for Muon
        weight_decay: Weight decay for all params
        aux_betas: Beta coefficients for AdamW
        aux_eps: Epsilon for AdamW

    Returns:
        (optimizer, info) tuple
    """
    # Get parameter grouping summary first
    summary = get_param_group_summary(model)
    total_params = summary.muon_params + summary.adam_params

    logger.info(
        f"Muon optimizer: {summary.muon_params:,} params ({100*summary.muon_params/total_params:.1f}%) with Muon, "
        f"{summary.adam_params:,} params ({100*summary.adam_params/total_params:.1f}%) with AdamW"
    )

    # Create parameter groups
    param_groups = split_transformer_params(
        model,
        hidden_lr=hidden_lr,
        aux_lr=aux_lr,
        hidden_momentum=momentum,
        weight_decay=weight_decay,
        aux_betas=aux_betas,
        aux_eps=aux_eps,
    )

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    info = OptimizerInfo(
        type="muon",
        learning_rate=hidden_lr,  # Primary LR
        weight_decay=weight_decay,
        total_params=total_params,
        trainable_params=total_params,
        muon_params=summary.muon_params,
        adam_params=summary.adam_params,
        hidden_lr=hidden_lr,
        aux_lr=aux_lr,
    )

    return optimizer, info


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 100,
    scheduler_type: str = "cosine",
) -> LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: "cosine" or "linear"

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    optimizer_type: Optional[str] = None,
    num_training_steps: Optional[int] = None,
    num_warmup_steps: Optional[int] = None,
) -> Tuple[torch.optim.Optimizer, Optional[LRScheduler], OptimizerInfo]:
    """
    Create optimizer and scheduler based on config.

    This is the main entry point for optimizer creation.

    Args:
        model: The model to optimize
        config: Configuration dict (config.json contents)
        optimizer_type: Override optimizer type ("adamw" or "muon")
        num_training_steps: Total training steps (for scheduler)
        num_warmup_steps: Warmup steps (for scheduler)

    Returns:
        (optimizer, scheduler, info) tuple. Scheduler is None if num_training_steps not provided.

    Example:
        # Basic usage
        optimizer, scheduler, info = create_optimizer(model, config, num_training_steps=10000)

        # Force Muon
        optimizer, scheduler, info = create_optimizer(
            model, config,
            optimizer_type="muon",
            num_training_steps=10000
        )

        # Without scheduler
        optimizer, _, info = create_optimizer(model, config)
    """
    opt_config = get_optimizer_config(config)
    opt_type = optimizer_type or opt_config["type"]

    if opt_type == "muon":
        muon_cfg = opt_config["muon"]
        optimizer, info = create_muon_optimizer(
            model,
            hidden_lr=muon_cfg["hidden_lr"],
            aux_lr=muon_cfg["aux_lr"],
            momentum=muon_cfg["momentum"],
            weight_decay=muon_cfg["weight_decay"],
            aux_betas=muon_cfg["aux_betas"],
            aux_eps=muon_cfg["aux_eps"],
        )
        logger.info(
            f"Created Muon optimizer: hidden_lr={muon_cfg['hidden_lr']}, "
            f"aux_lr={muon_cfg['aux_lr']}, momentum={muon_cfg['momentum']}"
        )
    else:
        # Default to AdamW
        adamw_cfg = opt_config["adamw"]
        optimizer, info = create_adamw_optimizer(
            model,
            lr=adamw_cfg["lr"],
            betas=adamw_cfg["betas"],
            eps=adamw_cfg["eps"],
            weight_decay=adamw_cfg["weight_decay"],
        )

    # Create scheduler if training steps provided
    scheduler = None
    if num_training_steps is not None:
        warmup = num_warmup_steps or config.get("warmup_steps", 100)
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=warmup,
            scheduler_type="cosine",
        )
        logger.info(f"Created cosine scheduler: {warmup} warmup, {num_training_steps} total steps")

    return optimizer, scheduler, info


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Get information about an existing optimizer.

    Args:
        optimizer: The optimizer to inspect

    Returns:
        Dict with optimizer details
    """
    info = {
        "type": type(optimizer).__name__,
        "param_groups": len(optimizer.param_groups),
    }

    if isinstance(optimizer, SingleDeviceMuonWithAuxAdam):
        counts = optimizer.get_param_counts()
        info["muon_params"] = counts["muon"]
        info["adam_params"] = counts["adam"]
        info["total_params"] = counts["total"]

        # Extract learning rates
        for group in optimizer.param_groups:
            if group.get("use_muon"):
                info["hidden_lr"] = group["lr"]
                info["momentum"] = group["momentum"]
            else:
                info["aux_lr"] = group["lr"]

    elif isinstance(optimizer, AdamW):
        info["total_params"] = sum(
            sum(p.numel() for p in g["params"])
            for g in optimizer.param_groups
        )
        info["lr"] = optimizer.param_groups[0]["lr"]

    return info
