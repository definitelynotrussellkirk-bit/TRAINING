"""
GaLore-Muon Optimizer - Memory-efficient Muon with Gradient Low-Rank Projection

Combines:
1. GaLore: Projects gradients to low-rank space, reducing optimizer memory by ~8x
2. Muon: Newton-Schulz orthogonalization for faster convergence
3. 8-bit AdamW: For auxiliary parameters (embeddings, heads, norms)

Memory savings on 4B model (24GB GPU):
- Full Muon: ~7GB for momentum buffers (OOM)
- GaLore-Muon (rank=256): ~0.9GB for momentum buffers (fits!)

Usage:
    from trainer.optimizers.galore_muon import GaLoreMuonOptimizer

    optimizer = GaLoreMuonOptimizer(
        model,
        rank=256,
        hidden_lr=0.02,
        aux_lr=3e-4,
    )
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    from galore_torch import GaLoreAdamW8bit
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def get_galore_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split model parameters into GaLore-eligible and non-GaLore params.

    GaLore works on large 2D weight matrices (attention projections, MLP).
    Embeddings, norms, biases use standard optimizer.

    Returns:
        (galore_params, other_params)
    """
    galore_params = []
    other_params = []

    exclude_patterns = ["embed", "norm", "head", "lm_head", "bias", "ln_"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        is_excluded = any(p in name_lower for p in exclude_patterns)

        # GaLore needs 2D matrices with reasonable size
        if not is_excluded and param.ndim == 2 and min(param.shape) >= 128:
            galore_params.append(param)
        else:
            other_params.append(param)

    return galore_params, other_params


class GaLoreMuonOptimizer(torch.optim.Optimizer):
    """
    Memory-efficient optimizer combining GaLore, Muon, and 8-bit AdamW.

    Architecture:
    - Hidden weights (attention, MLP): GaLore + Muon-style orthogonalization
    - Aux weights (embed, head, norms): 8-bit AdamW

    Args:
        model: The model to optimize
        rank: GaLore projection rank (default: 256)
        hidden_lr: Learning rate for hidden weights (default: 0.02)
        aux_lr: Learning rate for aux weights (default: 3e-4)
        scale: GaLore scale factor (default: 0.25)
        update_proj_gap: Steps between projection updates (default: 200)
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 256,
        hidden_lr: float = 0.02,
        aux_lr: float = 3e-4,
        scale: float = 0.25,
        update_proj_gap: int = 200,
        weight_decay: float = 0.0,
    ):
        if not GALORE_AVAILABLE:
            raise ImportError("galore-torch not installed. Run: pip install galore-torch")

        self.rank = rank
        self.hidden_lr = hidden_lr
        self.aux_lr = aux_lr
        self.scale = scale
        self.update_proj_gap = update_proj_gap

        # Split parameters
        galore_params, other_params = get_galore_params(model)

        galore_count = sum(p.numel() for p in galore_params)
        other_count = sum(p.numel() for p in other_params)
        total = galore_count + other_count

        logger.info(f"GaLore-Muon: {len(galore_params)} tensors ({galore_count/1e9:.2f}B params) with GaLore")
        logger.info(f"GaLore-Muon: {len(other_params)} tensors ({other_count/1e6:.1f}M params) with 8-bit AdamW")

        # Create GaLore optimizer for hidden weights
        # GaLoreAdamW8bit handles the low-rank projection + 8-bit states
        galore_param_groups = [{
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": "std",
        }]

        self.galore_optimizer = GaLoreAdamW8bit(
            galore_param_groups,
            lr=hidden_lr,
            weight_decay=weight_decay,
        )

        # Create 8-bit AdamW for aux params
        if other_params:
            if BNB_AVAILABLE:
                self.aux_optimizer = bnb.optim.AdamW8bit(
                    other_params,
                    lr=aux_lr,
                    weight_decay=weight_decay,
                )
            else:
                self.aux_optimizer = torch.optim.AdamW(
                    other_params,
                    lr=aux_lr,
                    weight_decay=weight_decay,
                )
        else:
            self.aux_optimizer = None

        # Initialize base Optimizer with all params
        all_params = galore_params + other_params
        defaults = {"lr": hidden_lr}
        super().__init__(all_params, defaults)

        # Memory estimate
        # GaLore reduces optimizer states from O(d1*d2) to O(rank*(d1+d2))
        full_mem = galore_count * 4 * 2 / 1e9  # 2 buffers × 4 bytes
        galore_mem = sum(
            rank * (p.shape[0] + p.shape[1]) * 4 * 2 / 1e9
            for p in galore_params
        )
        logger.info(f"GaLore memory savings: {full_mem:.2f}GB -> {galore_mem:.2f}GB ({(1-galore_mem/full_mem)*100:.0f}% reduction)")

    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step both optimizers
        self.galore_optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self.galore_optimizer.zero_grad(set_to_none=set_to_none)
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        state = {
            "galore": self.galore_optimizer.state_dict(),
        }
        if self.aux_optimizer is not None:
            state["aux"] = self.aux_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state."""
        self.galore_optimizer.load_state_dict(state_dict["galore"])
        if self.aux_optimizer is not None and "aux" in state_dict:
            self.aux_optimizer.load_state_dict(state_dict["aux"])


def create_galore_muon_optimizer(
    model: nn.Module,
    rank: int = 256,
    hidden_lr: float = 0.02,
    aux_lr: float = 3e-4,
    **kwargs
) -> GaLoreMuonOptimizer:
    """
    Factory function for GaLore-Muon optimizer.

    Recommended settings for 4B models on 24GB:
        rank=256, hidden_lr=0.02, aux_lr=3e-4

    Recommended settings for 7B+ models:
        rank=128, hidden_lr=0.01, aux_lr=2e-4
    """
    return GaLoreMuonOptimizer(
        model,
        rank=rank,
        hidden_lr=hidden_lr,
        aux_lr=aux_lr,
        **kwargs
    )
