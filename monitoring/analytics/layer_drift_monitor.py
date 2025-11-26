#!/usr/bin/env python3
"""
Layer Drift Monitor - Track which layers are changing during training.

This module compares checkpoint weights to identify:
- Which transformer layers are actively adapting
- Relative change magnitude per layer
- Drift patterns that might indicate instability or forgetting

No inference required - pure weight analysis makes this very cheap to run.

Usage:
    python3 layer_drift_monitor.py --base-dir ~/TRAINING --interval 600

Output:
    status/layer_drift.json - Per-layer drift metrics

Architecture:
    - Loads two checkpoints (reference + current)
    - For each transformer block, computes L2 norm of weight delta
    - Identifies which layers are changing most/least
    - Tracks patterns over time (e.g., "top layers moving, bottom frozen")
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LayerDrift:
    """Drift metrics for a single layer."""
    layer_idx: int
    layer_name: str
    delta_norm: float        # ||W_current - W_reference||_2
    reference_norm: float    # ||W_reference||_2
    relative_change: float   # delta_norm / reference_norm
    param_count: int         # Number of parameters in layer


@dataclass
class DriftAnalysis:
    """Complete drift analysis between two checkpoints."""
    reference_checkpoint: str
    current_checkpoint: str
    reference_step: Optional[int]
    current_step: Optional[int]
    timestamp: str
    total_params: int
    total_delta_norm: float
    total_relative_change: float
    layers: List[Dict[str, Any]]
    summary: Dict[str, Any]


class LayerDriftMonitor:
    """
    Monitor weight changes across transformer layers between checkpoints.

    This is a "cheap" analysis - no inference required, just weight comparison.
    Useful for understanding:
    - Where learning is happening (which layers)
    - Whether early layers have stabilized
    - Signs of catastrophic forgetting (sudden change in previously stable layers)

    Attributes:
        base_dir: Base directory for TRAINING
        status_dir: Where to write status JSON
        reference_path: Path to reference checkpoint (base model or best)
        history: List of past analyses for trend detection
    """

    def __init__(self, base_dir: Path, reference_path: Optional[Path] = None):
        """
        Initialize the layer drift monitor.

        Args:
            base_dir: Base TRAINING directory
            reference_path: Path to reference checkpoint. If None, uses base model.
        """
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(exist_ok=True)

        # Default reference is base model
        self.reference_path = reference_path or (self.base_dir / "models" / "Qwen3-0.6B")
        self.history: List[DriftAnalysis] = []
        self._load_history()

    def _load_history(self):
        """Load drift history from status file."""
        history_file = self.status_dir / "layer_drift_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    # Keep last 100 entries
                    self.history = data.get('history', [])[-100:]
            except Exception as e:
                logger.warning(f"Could not load history: {e}")

    def _save_history(self):
        """Save drift history."""
        history_file = self.status_dir / "layer_drift_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump({'history': self.history[-100:]}, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

    def _extract_step_from_path(self, path: Path) -> Optional[int]:
        """Extract checkpoint step number from path."""
        name = path.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except (IndexError, ValueError):
                pass
        return None

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in the checkpoints directory."""
        # Try local checkpoints dir first
        checkpoints_dir = self.base_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = sorted(
                checkpoints_dir.glob("checkpoint-*"),
                key=lambda p: self._extract_step_from_path(p) or 0,
                reverse=True
            )
            if checkpoints:
                return checkpoints[0]

        # Try current_model directory
        current_model = self.base_dir / "current_model"
        if current_model.exists():
            checkpoints = sorted(
                current_model.glob("checkpoint-*"),
                key=lambda p: self._extract_step_from_path(p) or 0,
                reverse=True
            )
            if checkpoints:
                return checkpoints[0]

        # Try models on 3090 - look for checkpoint-* directories, NOT "deployed"
        models_dir = Path.home() / "llm" / "models"
        if models_dir.exists():
            checkpoints = sorted(
                models_dir.glob("checkpoint-*"),
                key=lambda p: self._extract_step_from_path(p) or 0,
                reverse=True
            )
            if checkpoints:
                return checkpoints[0]

        return None

    def _load_state_dict(self, path: Path) -> Dict[str, torch.Tensor]:
        """
        Load model state dict from a checkpoint path.

        Args:
            path: Path to model or checkpoint directory

        Returns:
            State dict mapping parameter names to tensors
        """
        # Handle different checkpoint formats
        if (path / "model.safetensors").exists():
            from safetensors.torch import load_file
            return load_file(path / "model.safetensors")
        elif (path / "pytorch_model.bin").exists():
            return torch.load(path / "pytorch_model.bin", map_location="cpu")
        elif path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(path)
        elif path.suffix == ".bin":
            return torch.load(path, map_location="cpu")
        else:
            # Try safetensors shards
            shard_files = list(path.glob("model-*.safetensors"))
            if shard_files:
                from safetensors.torch import load_file
                state_dict = {}
                for shard in shard_files:
                    state_dict.update(load_file(shard))
                return state_dict
            raise ValueError(f"Could not find model weights in {path}")

    def _group_params_by_layer(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Group parameters by transformer layer index.

        Args:
            state_dict: Model state dict

        Returns:
            Dict mapping layer index to params in that layer
        """
        layers: Dict[int, Dict[str, torch.Tensor]] = {}

        for name, tensor in state_dict.items():
            # Parse layer index from parameter name
            # Common patterns: "model.layers.0.self_attn.q_proj.weight"
            #                  "transformer.h.0.attn.c_attn.weight"
            layer_idx = None

            if ".layers." in name:
                try:
                    # Qwen/Llama style: model.layers.N.xxx
                    parts = name.split(".layers.")
                    if len(parts) > 1:
                        layer_idx = int(parts[1].split(".")[0])
                except (ValueError, IndexError):
                    pass
            elif ".h." in name:
                try:
                    # GPT style: transformer.h.N.xxx
                    parts = name.split(".h.")
                    if len(parts) > 1:
                        layer_idx = int(parts[1].split(".")[0])
                except (ValueError, IndexError):
                    pass

            if layer_idx is not None:
                if layer_idx not in layers:
                    layers[layer_idx] = {}
                layers[layer_idx][name] = tensor
            else:
                # Non-layer params (embeddings, final norm, lm_head)
                # Put in layer -1 for "other"
                if -1 not in layers:
                    layers[-1] = {}
                layers[-1][name] = tensor

        return layers

    def _compute_layer_drift(
        self,
        ref_params: Dict[str, torch.Tensor],
        cur_params: Dict[str, torch.Tensor],
        layer_idx: int
    ) -> LayerDrift:
        """
        Compute drift metrics for a single layer.

        Args:
            ref_params: Reference checkpoint parameters for this layer
            cur_params: Current checkpoint parameters for this layer
            layer_idx: Layer index

        Returns:
            LayerDrift with computed metrics
        """
        # Flatten all params in layer and compute norms
        ref_flat = []
        cur_flat = []
        param_count = 0

        for name in ref_params:
            if name in cur_params:
                ref_tensor = ref_params[name].float().flatten()
                cur_tensor = cur_params[name].float().flatten()
                ref_flat.append(ref_tensor)
                cur_flat.append(cur_tensor)
                param_count += ref_tensor.numel()

        if not ref_flat:
            return LayerDrift(
                layer_idx=layer_idx,
                layer_name=f"layer_{layer_idx}" if layer_idx >= 0 else "embeddings_other",
                delta_norm=0.0,
                reference_norm=0.0,
                relative_change=0.0,
                param_count=0
            )

        ref_cat = torch.cat(ref_flat)
        cur_cat = torch.cat(cur_flat)

        delta = cur_cat - ref_cat
        delta_norm = torch.norm(delta, p=2).item()
        ref_norm = torch.norm(ref_cat, p=2).item()

        relative_change = delta_norm / ref_norm if ref_norm > 0 else 0.0

        layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "embeddings_other"

        return LayerDrift(
            layer_idx=layer_idx,
            layer_name=layer_name,
            delta_norm=delta_norm,
            reference_norm=ref_norm,
            relative_change=relative_change,
            param_count=param_count
        )

    def analyze(
        self,
        current_path: Optional[Path] = None,
        reference_path: Optional[Path] = None
    ) -> DriftAnalysis:
        """
        Analyze layer drift between reference and current checkpoint.

        Args:
            current_path: Path to current checkpoint. If None, finds latest.
            reference_path: Path to reference checkpoint. If None, uses default.

        Returns:
            DriftAnalysis with per-layer metrics and summary

        Side Effects:
            - Writes to status/layer_drift.json
            - Updates history
        """
        # Resolve paths
        if current_path is None:
            current_path = self._find_latest_checkpoint()
            if current_path is None:
                raise ValueError("No checkpoint found to analyze")

        if reference_path is None:
            reference_path = self.reference_path

        logger.info(f"Analyzing drift: {reference_path.name} -> {current_path.name}")

        # Load state dicts
        logger.info("Loading reference checkpoint...")
        ref_state = self._load_state_dict(reference_path)

        logger.info("Loading current checkpoint...")
        cur_state = self._load_state_dict(current_path)

        # Group by layer
        ref_layers = self._group_params_by_layer(ref_state)
        cur_layers = self._group_params_by_layer(cur_state)

        # Compute per-layer drift
        layer_drifts: List[LayerDrift] = []
        all_layer_indices = sorted(set(ref_layers.keys()) | set(cur_layers.keys()))

        for layer_idx in all_layer_indices:
            ref_params = ref_layers.get(layer_idx, {})
            cur_params = cur_layers.get(layer_idx, {})

            drift = self._compute_layer_drift(ref_params, cur_params, layer_idx)
            layer_drifts.append(drift)

        # Compute summary statistics
        transformer_drifts = [d for d in layer_drifts if d.layer_idx >= 0]

        if transformer_drifts:
            max_drift_layer = max(transformer_drifts, key=lambda d: d.relative_change)
            min_drift_layer = min(transformer_drifts, key=lambda d: d.relative_change)
            avg_relative_change = sum(d.relative_change for d in transformer_drifts) / len(transformer_drifts)

            # Detect patterns
            num_layers = len(transformer_drifts)
            top_quarter = transformer_drifts[int(num_layers * 0.75):]
            bottom_quarter = transformer_drifts[:int(num_layers * 0.25)]

            top_avg = sum(d.relative_change for d in top_quarter) / len(top_quarter) if top_quarter else 0
            bottom_avg = sum(d.relative_change for d in bottom_quarter) / len(bottom_quarter) if bottom_quarter else 0

            if top_avg > 2 * bottom_avg:
                pattern = "top_heavy"  # Most change in upper layers (normal)
            elif bottom_avg > 2 * top_avg:
                pattern = "bottom_heavy"  # Most change in lower layers (unusual)
            else:
                pattern = "uniform"  # Change spread across layers
        else:
            max_drift_layer = min_drift_layer = None
            avg_relative_change = 0.0
            pattern = "unknown"

        # Total drift across all params
        total_delta_norm = sum(d.delta_norm for d in layer_drifts)
        total_ref_norm = sum(d.reference_norm for d in layer_drifts)
        total_relative_change = total_delta_norm / total_ref_norm if total_ref_norm > 0 else 0
        total_params = sum(d.param_count for d in layer_drifts)

        summary = {
            "pattern": pattern,
            "avg_relative_change": round(avg_relative_change, 6),
            "max_drift_layer": max_drift_layer.layer_idx if max_drift_layer else None,
            "max_drift_value": round(max_drift_layer.relative_change, 6) if max_drift_layer else None,
            "min_drift_layer": min_drift_layer.layer_idx if min_drift_layer else None,
            "min_drift_value": round(min_drift_layer.relative_change, 6) if min_drift_layer else None,
            "top_layers_avg": round(top_avg, 6) if transformer_drifts else None,
            "bottom_layers_avg": round(bottom_avg, 6) if transformer_drifts else None,
            "num_transformer_layers": len(transformer_drifts),
        }

        analysis = DriftAnalysis(
            reference_checkpoint=str(reference_path.name),
            current_checkpoint=str(current_path.name),
            reference_step=self._extract_step_from_path(reference_path),
            current_step=self._extract_step_from_path(current_path),
            timestamp=datetime.now().isoformat(),
            total_params=total_params,
            total_delta_norm=round(total_delta_norm, 4),
            total_relative_change=round(total_relative_change, 6),
            layers=[asdict(d) for d in layer_drifts],
            summary=summary
        )

        # Save current analysis
        self._save_analysis(analysis)

        # Update history
        self.history.append(asdict(analysis))
        self._save_history()

        logger.info(f"Analysis complete. Pattern: {pattern}, Avg change: {avg_relative_change:.4%}")

        return analysis

    def _save_analysis(self, analysis: DriftAnalysis):
        """Save analysis to status file."""
        output_path = self.status_dir / "layer_drift.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        logger.info(f"Saved to {output_path}")

    def run_daemon(self, interval: int = 600):
        """
        Run as a daemon, analyzing drift periodically.

        Args:
            interval: Seconds between analyses
        """
        logger.info(f"Starting layer drift monitor daemon (interval={interval}s)")

        last_checkpoint = None

        while True:
            try:
                current = self._find_latest_checkpoint()

                if current and current != last_checkpoint:
                    logger.info(f"New checkpoint detected: {current.name}")
                    self.analyze(current)
                    last_checkpoint = current
                else:
                    logger.debug("No new checkpoint")

            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Layer Drift Monitor")
    parser.add_argument("--base-dir", type=Path, required=True, help="Base TRAINING directory")
    parser.add_argument("--reference", type=Path, help="Reference checkpoint path")
    parser.add_argument("--current", type=Path, help="Current checkpoint path (default: latest)")
    parser.add_argument("--interval", type=int, default=600, help="Daemon interval in seconds")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--once", action="store_true", help="Run once and exit")

    args = parser.parse_args()

    monitor = LayerDriftMonitor(args.base_dir, args.reference)

    if args.daemon:
        monitor.run_daemon(args.interval)
    else:
        analysis = monitor.analyze(args.current, args.reference)
        print(json.dumps(asdict(analysis), indent=2))


if __name__ == "__main__":
    main()
