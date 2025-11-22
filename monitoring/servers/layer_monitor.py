#!/usr/bin/env python3
"""Layer activity monitor for full-model training.

Computes per-layer weight norms and how much they change between snapshots.
This lets the UI highlight which transformer blocks are adapting the most.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch


@dataclass
class LayerChange:
    """Single layer summary used by the UI."""

    name: str
    norm: float
    delta: float
    relative_delta: float
    avg_delta: float
    stability: float


class LayerMonitor:
    """Tracks weight-norm changes per logical layer group."""

    def __init__(self, model: torch.nn.Module, max_history: int = 10):
        self.model = model
        self.max_history = max_history
        self.layer_params: Dict[str, List[torch.nn.Parameter]] = self._group_parameters()
        self.last_norms: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {name: [] for name in self.layer_params}
        self.enabled = bool(self.layer_params)

        total_layers = len(self.layer_params)
        if self.enabled:
            print(f"📊 LayerMonitor enabled ({total_layers} layer groups tracked)")
        else:
            print("⚠️ LayerMonitor disabled: no trainable parameters discovered")

    def _group_parameters(self) -> Dict[str, List[torch.nn.Parameter]]:
        groups: Dict[str, List[torch.nn.Parameter]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group = self._extract_group_name(name)
            groups.setdefault(group, []).append(param)
        return groups

    @staticmethod
    def _extract_group_name(param_name: str) -> str:
        tokens = param_name.split('.')
        markers = ["layers", "blocks", "h", "layer", "transformer"]
        for marker in markers:
            if marker in tokens:
                idx = tokens.index(marker)
                if idx + 1 < len(tokens):
                    identifier = tokens[idx + 1]
                    return f"{marker}.{identifier}"
        # Fallback: first token (e.g., embedding, lm_head)
        return tokens[0]

    @torch.no_grad()
    def snapshot(self) -> Dict:
        if not self.enabled:
            return {}

        changes: List[LayerChange] = []
        total_delta = 0.0

        for name, params in self.layer_params.items():
            current_norm = 0.0
            for param in params:
                # Operate on GPU tensors to avoid large host transfers
                current_norm += torch.linalg.norm(param.data).item()

            previous = self.last_norms.get(name)
            delta = abs(current_norm - previous) if previous is not None else 0.0
            rel_delta = delta / current_norm if current_norm > 0 else 0.0
            history = self.history.setdefault(name, [])
            if previous is not None:
                history.append(delta)
                if len(history) > self.max_history:
                    history.pop(0)
            avg_delta = (sum(history) / len(history)) if history else delta
            stability = 0.0
            if history:
                mean = avg_delta
                stability = (sum((h - mean) ** 2 for h in history) / len(history)) ** 0.5

            changes.append(LayerChange(
                name=name,
                norm=current_norm,
                delta=delta,
                relative_delta=rel_delta,
                avg_delta=avg_delta,
                stability=stability,
            ))

            self.last_norms[name] = current_norm
            total_delta += delta

        changes.sort(key=lambda c: c.delta, reverse=True)

        overall = {
            "total_layers": len(changes),
            "avg_delta": total_delta / len(changes) if changes else 0.0,
            "max_delta": changes[0].delta if changes else 0.0,
        }

        stability_rank = sorted(changes, key=lambda c: c.stability, reverse=True)
        stability_summary = {
            "top": [
                {
                    "name": change.name,
                    "stability": change.stability,
                    "avg_delta": change.avg_delta,
                }
                for change in stability_rank[:5]
            ],
            "average": sum(c.stability for c in changes) / len(changes) if changes else 0.0,
        }

        return {
            "generated_at": datetime.now().isoformat(),
            "top_changes": [change.__dict__ for change in changes[:5]],
            "all_layers": [change.__dict__ for change in changes],
            "overall": overall,
            "stability": stability_summary,
        }
