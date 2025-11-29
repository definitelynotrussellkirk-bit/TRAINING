"""
Muon Optimizer - MomentUm Orthogonalized by Newton-schulz

Vendored from: https://github.com/KellerJordan/Muon
Author: Keller Jordan
License: MIT

Muon is an optimizer for hidden layers in neural networks that:
1. Runs standard SGD-momentum internally
2. Orthogonalizes each 2D parameter's update using Newton-Schulz iteration
3. Achieves faster convergence than AdamW on many tasks

IMPORTANT: Muon should ONLY be used for hidden weight matrices (ndim >= 2).
Embeddings, output heads, and biases should use AdamW.

This file contains single-device variants suitable for single-GPU training.
For distributed training, see the original repository.

Usage:
    from trainer.optimizers import SingleDeviceMuonWithAuxAdam, split_transformer_params

    param_groups = split_transformer_params(model, hidden_lr=0.02, aux_lr=3e-4)
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    # With 8-bit AdamW for aux params (saves ~2GB on 4B models):
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups, use_8bit_adam=True)
"""

import torch
import logging

logger = logging.getLogger(__name__)

# Try to import bitsandbytes for 8-bit Adam
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute orthogonalization of G.

    Uses a quintic iteration with coefficients selected to maximize slope at zero.
    This produces something like US'V^T where S' is diagonal with values ~ Uniform(0.5, 1.5),
    which empirically doesn't hurt model performance relative to exact UV^T.

    Args:
        G: Gradient tensor, must be at least 2D
        steps: Number of Newton-Schulz iterations (default: 5)

    Returns:
        Orthogonalized tensor approximating the nearest orthogonal matrix to G
    """
    assert G.ndim >= 2, "Input must be at least 2D"

    # Quintic coefficients for fast convergence
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True
) -> torch.Tensor:
    """
    Compute Muon update: momentum + orthogonalization.

    Args:
        grad: Current gradient
        momentum: Momentum buffer (modified in-place)
        beta: Momentum coefficient (default: 0.95)
        ns_steps: Newton-Schulz iterations (default: 5)
        nesterov: Use Nesterov momentum (default: True)

    Returns:
        Orthogonalized update tensor
    """
    # Update momentum buffer
    momentum.lerp_(grad, 1 - beta)

    # Nesterov lookahead or standard momentum
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Handle 4D conv filters by reshaping to 2D
    if update.ndim == 4:
        update = update.view(len(update), -1)

    # Orthogonalize via Newton-Schulz
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    # Scale by aspect ratio
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5

    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple,
    eps: float
) -> torch.Tensor:
    """
    Compute AdamW-style update.

    Args:
        grad: Current gradient
        buf1: First moment buffer (modified in-place)
        buf2: Second moment buffer (modified in-place)
        step: Current optimization step
        betas: (beta1, beta2) momentum coefficients
        eps: Epsilon for numerical stability

    Returns:
        Adam update direction
    """
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])

    # Bias correction
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)

    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon optimizer for single-GPU training.

    Muon (MomentUm Orthogonalized by Newton-schulz) runs standard SGD-momentum
    internally, then orthogonalizes each 2D parameter's update using Newton-Schulz
    iteration.

    IMPORTANT: Only use for hidden weight matrices. Use AdamW for embeddings,
    output heads, and biases.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate in units of spectral norm per update (default: 0.02)
        weight_decay: AdamW-style weight decay (default: 0)
        momentum: Momentum coefficient (default: 0.95)

    Example:
        hidden_weights = [p for n, p in model.named_parameters()
                         if p.ndim >= 2 and 'embed' not in n]
        optimizer = SingleDeviceMuon(hidden_weights, lr=0.02)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"]
                )

                # AdamW-style weight decay (before update)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Hybrid optimizer: Muon for hidden weights, AdamW for everything else.

    This is the recommended optimizer for transformer training. It automatically
    applies Muon to parameters marked with use_muon=True and AdamW to others.

    Args:
        param_groups: List of parameter group dicts, each must have 'use_muon' key.
            - use_muon=True: Uses Muon (lr, momentum, weight_decay)
            - use_muon=False: Uses AdamW (lr, betas, eps, weight_decay)
        use_8bit_adam: Use 8-bit AdamW for aux params (saves ~2GB VRAM on 4B models)

    Example:
        from trainer.optimizers import split_transformer_params

        param_groups = split_transformer_params(model, hidden_lr=0.02, aux_lr=3e-4)
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # With 8-bit AdamW for memory savings:
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups, use_8bit_adam=True)

    Hyperparameters:
        Muon groups (use_muon=True):
            - lr: Learning rate (default: 0.02)
            - momentum: Momentum coefficient (default: 0.95)
            - weight_decay: Weight decay (default: 0)

        Adam groups (use_muon=False):
            - lr: Learning rate (default: 3e-4)
            - betas: (beta1, beta2) (default: (0.9, 0.95))
            - eps: Epsilon (default: 1e-10)
            - weight_decay: Weight decay (default: 0)
    """

    def __init__(self, param_groups: list, use_8bit_adam: bool = False):
        self.use_8bit_adam = use_8bit_adam and BNB_AVAILABLE
        self._aux_optimizer = None

        muon_count = 0
        adam_count = 0
        adam_params = []

        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param_group must have 'use_muon' key")

            if group["use_muon"]:
                # Muon defaults
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0)
                group.setdefault("ns_steps", 5)  # Newton-Schulz iterations
                muon_count += len(group["params"])
            else:
                # AdamW defaults
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0)
                adam_count += len(group["params"])
                adam_params.extend(group["params"])

        # Create 8-bit optimizer for aux params if requested
        if self.use_8bit_adam and adam_params:
            # Find the adam group to get hyperparams
            adam_group = next(g for g in param_groups if not g["use_muon"])
            self._aux_optimizer = bnb.optim.AdamW8bit(
                adam_params,
                lr=adam_group["lr"],
                betas=adam_group["betas"],
                eps=adam_group["eps"],
                weight_decay=adam_group["weight_decay"],
            )
            logger.info(f"Muon optimizer: {muon_count} tensors with Muon, {adam_count} tensors with 8-bit AdamW")
        else:
            if use_8bit_adam and not BNB_AVAILABLE:
                logger.warning("8-bit AdamW requested but bitsandbytes not available. Using standard AdamW.")
            logger.info(f"Muon optimizer: {muon_count} tensors with Muon, {adam_count} tensors with AdamW")

        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon update for hidden weights
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group.get("ns_steps", 5)
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            elif self._aux_optimizer is not None:
                # Use 8-bit AdamW (already has the params)
                pass  # Handled below
            else:
                # Standard AdamW update for embeddings, heads, biases
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"]
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        # Step the 8-bit optimizer if we're using it
        if self._aux_optimizer is not None:
            self._aux_optimizer.step()

        return loss

    def get_param_counts(self) -> dict:
        """Get count of parameters by optimizer type."""
        muon_params = 0
        adam_params = 0
        for group in self.param_groups:
            count = sum(p.numel() for p in group["params"])
            if group["use_muon"]:
                muon_params += count
            else:
                adam_params += count
        return {"muon": muon_params, "adam": adam_params, "total": muon_params + adam_params}
