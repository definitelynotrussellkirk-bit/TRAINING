#!/usr/bin/env python3
"""
LoRA Layer Monitoring

Tracks LoRA adapter activity during training:
- Gradient norms per layer (how much signal each layer receives)
- Update magnitudes (how much each layer is changing)
- Identifies which layers are learning vs. frozen

Helps diagnose:
- Dead layers (zero gradients)
- Vanishing/exploding gradients
- Unbalanced learning across layers
"""

import torch
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class LoRAMonitor:
    """Monitor LoRA layer activity during training."""

    def __init__(self, model):
        """
        Initialize LoRA monitor.

        Args:
            model: PEFT model with LoRA adapters
        """
        self.model = model
        self.layer_names = self._extract_lora_layers()

        # Track gradient norms over time
        self.gradient_history = defaultdict(list)

        # Track update magnitudes (weight changes)
        self.update_history = defaultdict(list)

        # Store previous weights for delta calculation
        self.previous_weights = {}

        # Max history length (keep last N updates)
        self.max_history = 100

        print(f"📊 LoRA Monitor initialized: {len(self.layer_names)} LoRA layers detected")

    def _extract_lora_layers(self) -> List[str]:
        """Extract names of LoRA adapter layers from model."""
        lora_layers = []

        for name, module in self.model.named_modules():
            # PEFT LoRA layers contain "lora_A" or "lora_B" in their names
            if "lora_A" in name or "lora_B" in name:
                # Extract base layer name (remove .lora_A/.lora_B suffix)
                base_name = name.rsplit(".", 1)[0]
                if base_name not in lora_layers:
                    lora_layers.append(base_name)

        return sorted(lora_layers)

    def collect_gradients(self) -> Dict[str, float]:
        """
        Collect gradient norms for each LoRA layer.

        Returns:
            Dict mapping layer name to gradient norm
        """
        gradient_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and "lora" in name:
                # Extract base layer name
                base_name = name.rsplit(".", 2)[0]  # Remove .lora_A/B.weight

                # Calculate L2 norm of gradients
                grad_norm = param.grad.norm(2).item()

                # Aggregate if multiple params per layer (A and B matrices)
                if base_name in gradient_norms:
                    gradient_norms[base_name] += grad_norm
                else:
                    gradient_norms[base_name] = grad_norm

        # Store in history
        for layer_name, norm in gradient_norms.items():
            self.gradient_history[layer_name].append(norm)

            # Trim history if too long
            if len(self.gradient_history[layer_name]) > self.max_history:
                self.gradient_history[layer_name] = self.gradient_history[layer_name][-self.max_history:]

        return gradient_norms

    def collect_updates(self) -> Dict[str, float]:
        """
        Collect update magnitudes (weight changes) for each LoRA layer.

        Should be called AFTER optimizer step.

        Returns:
            Dict mapping layer name to update magnitude
        """
        update_magnitudes = {}

        for name, param in self.model.named_parameters():
            if "lora" in name and param.requires_grad:
                # Extract base layer name
                base_name = name.rsplit(".", 2)[0]

                # Calculate update magnitude (difference from previous weights)
                if base_name in self.previous_weights:
                    prev_weight = self.previous_weights[base_name]
                    delta = (param.data - prev_weight).norm(2).item()

                    # Aggregate if multiple params per layer
                    if base_name in update_magnitudes:
                        update_magnitudes[base_name] += delta
                    else:
                        update_magnitudes[base_name] = delta

                # Store current weights for next comparison
                self.previous_weights[base_name] = param.data.clone()

        # Store in history
        for layer_name, magnitude in update_magnitudes.items():
            self.update_history[layer_name].append(magnitude)

            # Trim history if too long
            if len(self.update_history[layer_name]) > self.max_history:
                self.update_history[layer_name] = self.update_history[layer_name][-self.max_history:]

        return update_magnitudes

    def get_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each LoRA layer.

        Returns:
            Dict mapping layer name to stats dict containing:
            - mean_grad_norm: Average gradient norm
            - max_grad_norm: Maximum gradient norm
            - mean_update: Average update magnitude
            - max_update: Maximum update magnitude
            - num_updates: Number of updates tracked
        """
        stats = {}

        for layer_name in self.layer_names:
            layer_stats = {}

            # Gradient stats
            if layer_name in self.gradient_history and self.gradient_history[layer_name]:
                grads = self.gradient_history[layer_name]
                layer_stats["mean_grad_norm"] = float(np.mean(grads))
                layer_stats["max_grad_norm"] = float(np.max(grads))
                layer_stats["current_grad_norm"] = float(grads[-1])
            else:
                layer_stats["mean_grad_norm"] = 0.0
                layer_stats["max_grad_norm"] = 0.0
                layer_stats["current_grad_norm"] = 0.0

            # Update stats
            if layer_name in self.update_history and self.update_history[layer_name]:
                updates = self.update_history[layer_name]
                layer_stats["mean_update"] = float(np.mean(updates))
                layer_stats["max_update"] = float(np.max(updates))
                layer_stats["current_update"] = float(updates[-1])
                layer_stats["num_updates"] = len(updates)
            else:
                layer_stats["mean_update"] = 0.0
                layer_stats["max_update"] = 0.0
                layer_stats["current_update"] = 0.0
                layer_stats["num_updates"] = 0

            stats[layer_name] = layer_stats

        return stats

    def get_summary(self) -> Dict:
        """
        Get high-level summary of LoRA activity.

        Returns:
            Dict containing:
            - total_layers: Number of LoRA layers
            - active_layers: Number of layers with non-zero gradients
            - dead_layers: List of layer names with zero activity
            - top_layers: Top 5 layers by gradient norm
        """
        stats = self.get_layer_stats()

        active_layers = [name for name, s in stats.items() if s["current_grad_norm"] > 1e-8]
        dead_layers = [name for name, s in stats.items() if s["current_grad_norm"] <= 1e-8]

        # Sort by current gradient norm
        sorted_layers = sorted(
            stats.items(),
            key=lambda x: x[1]["current_grad_norm"],
            reverse=True
        )

        return {
            "total_layers": len(self.layer_names),
            "active_layers": len(active_layers),
            "dead_layers": dead_layers,
            "top_layers": [
                {"name": name, "grad_norm": s["current_grad_norm"]}
                for name, s in sorted_layers[:5]
            ]
        }

    def detect_issues(self) -> List[Dict[str, str]]:
        """
        Detect potential training issues based on LoRA layer activity.

        Returns:
            List of issue dicts with 'severity', 'type', and 'message'
        """
        issues = []
        stats = self.get_layer_stats()

        # Check for vanishing gradients
        vanishing_layers = [
            name for name, s in stats.items()
            if s["current_grad_norm"] > 0 and s["current_grad_norm"] < 1e-5
        ]
        if vanishing_layers:
            issues.append({
                "severity": "warning",
                "type": "vanishing_gradients",
                "message": f"{len(vanishing_layers)} layers have very small gradients (< 1e-5)"
            })

        # Check for exploding gradients
        exploding_layers = [
            name for name, s in stats.items()
            if s["current_grad_norm"] > 100.0
        ]
        if exploding_layers:
            issues.append({
                "severity": "critical",
                "type": "exploding_gradients",
                "message": f"{len(exploding_layers)} layers have very large gradients (> 100)"
            })

        # Check for dead layers (no activity)
        if stats:
            total_layers = len(stats)
            active_layers = sum(1 for s in stats.values() if s["current_grad_norm"] > 1e-8)
            if active_layers < total_layers * 0.5:
                issues.append({
                    "severity": "warning",
                    "type": "dead_layers",
                    "message": f"Only {active_layers}/{total_layers} layers are active"
                })

        # Check for gradient imbalance
        if stats:
            grad_norms = [s["current_grad_norm"] for s in stats.values() if s["current_grad_norm"] > 0]
            if grad_norms:
                max_norm = max(grad_norms)
                min_norm = min(grad_norms)
                if max_norm > 0 and min_norm > 0:
                    ratio = max_norm / min_norm
                    if ratio > 1000:
                        issues.append({
                            "severity": "warning",
                            "type": "gradient_imbalance",
                            "message": f"Large gradient imbalance (max/min ratio: {ratio:.0f}x)"
                        })

        return issues


def create_lora_monitor(model) -> LoRAMonitor:
    """
    Factory function to create a LoRA monitor.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        LoRAMonitor instance
    """
    return LoRAMonitor(model)


if __name__ == "__main__":
    print("LoRA Monitor - Test Mode")
    print("This module should be imported and used during training.")
