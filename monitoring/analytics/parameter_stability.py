#!/usr/bin/env python3
"""
Parameter Stability Monitor - Track weight norms over time.

Catches silent training pathologies:
- Exploding weights (norms growing unbounded)
- Vanishing gradients (weights barely changing)
- Dead layers (frozen parameters)
- Sudden instability (norm spikes)

No inference required - pure weight analysis.

Usage:
    python3 parameter_stability.py --base-dir ~/TRAINING --interval 600

Output:
    status/parameter_stability.json - Current norms + alerts
    status/parameter_stability_history.json - Time series for trending
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LayerNorms:
    """Weight norms for a single layer."""
    layer_idx: int
    layer_name: str
    weight_norm: float       # L2 norm of all weights
    bias_norm: float         # L2 norm of biases (if present)
    max_abs_weight: float    # Max absolute value (detect explosions)
    min_abs_weight: float    # Min absolute value (detect vanishing)
    param_count: int


@dataclass
class StabilitySnapshot:
    """Stability metrics for a single checkpoint."""
    checkpoint: str
    step: Optional[int]
    timestamp: str
    layers: List[Dict[str, Any]]
    summary: Dict[str, Any]
    alerts: List[Dict[str, Any]]


@dataclass
class Alert:
    """Stability alert."""
    severity: str  # "warning" | "critical"
    type: str      # "exploding" | "vanishing" | "dead" | "spike"
    layer: int
    message: str
    value: float
    threshold: float


class ParameterStabilityMonitor:
    """
    Monitor parameter norms across checkpoints to detect pathologies.

    Tracks:
    - Per-layer weight norms over time
    - Max/min absolute values
    - Running statistics for anomaly detection
    - Alerts for concerning patterns

    Attributes:
        base_dir: Base TRAINING directory
        history: Time series of snapshots
        thresholds: Alert thresholds
    """

    # Default thresholds
    THRESHOLDS = {
        "exploding_norm": 1000.0,      # Weight norm above this is concerning
        "exploding_max": 100.0,        # Max weight above this
        "vanishing_norm": 0.001,       # Weight norm below this
        "spike_factor": 3.0,           # Norm change > 3x is a spike
        "dead_change_threshold": 1e-6, # Less than this change = dead
    }

    def __init__(self, base_dir: Optional[Path] = None, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the parameter stability monitor.

        Args:
            base_dir: Base TRAINING directory
            thresholds: Custom alert thresholds (optional)
        """
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(exist_ok=True)

        self.thresholds = {**self.THRESHOLDS, **(thresholds or {})}
        self.history: List[StabilitySnapshot] = []
        self._load_history()

    def _load_history(self):
        """Load history from status file."""
        history_file = self.status_dir / "parameter_stability_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    self.history = [
                        StabilitySnapshot(**h) for h in data.get('history', [])[-200:]
                    ]
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
                self.history = []

    def _save_history(self):
        """Save history to status file."""
        history_file = self.status_dir / "parameter_stability_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'history': [asdict(h) for h in self.history[-200:]]
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

    def _extract_step_from_path(self, path: Path) -> Optional[int]:
        """Extract checkpoint step from path."""
        name = path.name
        if name.startswith("checkpoint-"):
            try:
                return int(name.split("-")[1])
            except (IndexError, ValueError):
                pass
        return None

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recent checkpoint."""
        # Try local checkpoints
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
        """Load model state dict from checkpoint."""
        if (path / "model.safetensors").exists():
            from safetensors.torch import load_file
            return load_file(path / "model.safetensors")
        elif (path / "pytorch_model.bin").exists():
            return torch.load(path / "pytorch_model.bin", map_location="cpu")
        else:
            shard_files = list(path.glob("model-*.safetensors"))
            if shard_files:
                from safetensors.torch import load_file
                state_dict = {}
                for shard in shard_files:
                    state_dict.update(load_file(shard))
                return state_dict
            raise ValueError(f"Could not find model weights in {path}")

    def _compute_layer_norms(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> List[LayerNorms]:
        """
        Compute norms for each transformer layer.

        Args:
            state_dict: Model state dict

        Returns:
            List of LayerNorms for each layer
        """
        # Group params by layer
        layers: Dict[int, Dict[str, torch.Tensor]] = {}

        for name, tensor in state_dict.items():
            layer_idx = None

            if ".layers." in name:
                try:
                    parts = name.split(".layers.")
                    if len(parts) > 1:
                        layer_idx = int(parts[1].split(".")[0])
                except (ValueError, IndexError):
                    pass
            elif ".h." in name:
                try:
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
                if -1 not in layers:
                    layers[-1] = {}
                layers[-1][name] = tensor

        # Compute norms per layer
        results = []
        for layer_idx in sorted(layers.keys()):
            params = layers[layer_idx]

            weights = []
            biases = []
            param_count = 0

            for name, tensor in params.items():
                t = tensor.float().flatten()
                param_count += t.numel()
                if "bias" in name:
                    biases.append(t)
                else:
                    weights.append(t)

            # Concatenate and compute norms
            if weights:
                w_cat = torch.cat(weights)
                weight_norm = torch.norm(w_cat, p=2).item()
                max_abs = torch.max(torch.abs(w_cat)).item()
                min_abs = torch.min(torch.abs(w_cat)).item()
            else:
                weight_norm = max_abs = min_abs = 0.0

            if biases:
                b_cat = torch.cat(biases)
                bias_norm = torch.norm(b_cat, p=2).item()
            else:
                bias_norm = 0.0

            layer_name = f"layer_{layer_idx}" if layer_idx >= 0 else "embeddings_other"

            results.append(LayerNorms(
                layer_idx=layer_idx,
                layer_name=layer_name,
                weight_norm=weight_norm,
                bias_norm=bias_norm,
                max_abs_weight=max_abs,
                min_abs_weight=min_abs,
                param_count=param_count
            ))

        return results

    def _check_alerts(
        self,
        current: List[LayerNorms],
        previous: Optional[List[LayerNorms]] = None
    ) -> List[Alert]:
        """
        Check for stability alerts.

        Args:
            current: Current layer norms
            previous: Previous snapshot norms (for spike detection)

        Returns:
            List of alerts
        """
        alerts = []

        prev_norms = {}
        if previous:
            prev_norms = {n.layer_idx: n for n in previous}

        for layer in current:
            if layer.layer_idx < 0:  # Skip embeddings/other
                continue

            # Check for exploding weights
            if layer.weight_norm > self.thresholds["exploding_norm"]:
                alerts.append(Alert(
                    severity="critical",
                    type="exploding",
                    layer=layer.layer_idx,
                    message=f"Layer {layer.layer_idx} weight norm extremely high",
                    value=layer.weight_norm,
                    threshold=self.thresholds["exploding_norm"]
                ))

            if layer.max_abs_weight > self.thresholds["exploding_max"]:
                alerts.append(Alert(
                    severity="warning",
                    type="exploding",
                    layer=layer.layer_idx,
                    message=f"Layer {layer.layer_idx} has extreme weight values",
                    value=layer.max_abs_weight,
                    threshold=self.thresholds["exploding_max"]
                ))

            # Check for vanishing weights
            if layer.weight_norm < self.thresholds["vanishing_norm"]:
                alerts.append(Alert(
                    severity="warning",
                    type="vanishing",
                    layer=layer.layer_idx,
                    message=f"Layer {layer.layer_idx} weight norm very low",
                    value=layer.weight_norm,
                    threshold=self.thresholds["vanishing_norm"]
                ))

            # Check for spikes (sudden changes)
            if layer.layer_idx in prev_norms:
                prev = prev_norms[layer.layer_idx]
                if prev.weight_norm > 0:
                    ratio = layer.weight_norm / prev.weight_norm
                    if ratio > self.thresholds["spike_factor"]:
                        alerts.append(Alert(
                            severity="warning",
                            type="spike",
                            layer=layer.layer_idx,
                            message=f"Layer {layer.layer_idx} norm spiked {ratio:.1f}x",
                            value=ratio,
                            threshold=self.thresholds["spike_factor"]
                        ))

        return alerts

    def analyze(self, checkpoint_path: Optional[Path] = None) -> StabilitySnapshot:
        """
        Analyze parameter stability for a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. If None, finds latest.

        Returns:
            StabilitySnapshot with norms and alerts

        Side Effects:
            - Writes to status/parameter_stability.json
            - Updates history
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoint found")

        logger.info(f"Analyzing stability: {checkpoint_path.name}")

        # Load and compute norms
        state_dict = self._load_state_dict(checkpoint_path)
        layer_norms = self._compute_layer_norms(state_dict)

        # Get previous for comparison
        previous_norms = None
        if self.history:
            try:
                previous_norms = [
                    LayerNorms(**ln) for ln in self.history[-1].layers
                ]
            except Exception:
                pass

        # Check for alerts
        alerts = self._check_alerts(layer_norms, previous_norms)

        # Compute summary
        transformer_layers = [n for n in layer_norms if n.layer_idx >= 0]
        if transformer_layers:
            avg_norm = sum(n.weight_norm for n in transformer_layers) / len(transformer_layers)
            max_norm = max(n.weight_norm for n in transformer_layers)
            min_norm = min(n.weight_norm for n in transformer_layers)
            std_norm = (sum((n.weight_norm - avg_norm)**2 for n in transformer_layers) / len(transformer_layers)) ** 0.5
        else:
            avg_norm = max_norm = min_norm = std_norm = 0.0

        summary = {
            "avg_weight_norm": round(avg_norm, 4),
            "max_weight_norm": round(max_norm, 4),
            "min_weight_norm": round(min_norm, 4),
            "std_weight_norm": round(std_norm, 4),
            "num_layers": len(transformer_layers),
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in alerts if a.severity == "warning"]),
            "health_status": "critical" if any(a.severity == "critical" for a in alerts)
                           else "warning" if alerts
                           else "healthy"
        }

        snapshot = StabilitySnapshot(
            checkpoint=str(checkpoint_path.name),
            step=self._extract_step_from_path(checkpoint_path),
            timestamp=datetime.now().isoformat(),
            layers=[asdict(n) for n in layer_norms],
            summary=summary,
            alerts=[asdict(a) for a in alerts]
        )

        # Save
        self._save_snapshot(snapshot)
        self.history.append(snapshot)
        self._save_history()

        status = summary["health_status"]
        logger.info(f"Analysis complete. Status: {status}, Alerts: {len(alerts)}")

        return snapshot

    def _save_snapshot(self, snapshot: StabilitySnapshot):
        """Save current snapshot to status file."""
        output_path = self.status_dir / "parameter_stability.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2)
        logger.info(f"Saved to {output_path}")

    def get_trends(self, num_snapshots: int = 20) -> Dict[str, Any]:
        """
        Get trending data for visualization.

        Args:
            num_snapshots: Number of recent snapshots to include

        Returns:
            Dict with time series data per layer
        """
        recent = self.history[-num_snapshots:]
        if not recent:
            return {"error": "No history available"}

        # Build time series per layer
        layers_data: Dict[int, List[Dict]] = {}

        for snapshot in recent:
            step = snapshot.step or 0
            for layer in snapshot.layers:
                layer_idx = layer["layer_idx"]
                if layer_idx not in layers_data:
                    layers_data[layer_idx] = []
                layers_data[layer_idx].append({
                    "step": step,
                    "weight_norm": layer["weight_norm"],
                    "timestamp": snapshot.timestamp
                })

        return {
            "num_snapshots": len(recent),
            "layers": layers_data,
            "latest_summary": recent[-1].summary if recent else None
        }

    def run_daemon(self, interval: int = 600):
        """
        Run as daemon, analyzing periodically.

        Args:
            interval: Seconds between analyses
        """
        logger.info(f"Starting parameter stability monitor (interval={interval}s)")

        last_checkpoint = None

        while True:
            try:
                current = self._find_latest_checkpoint()

                if current and current != last_checkpoint:
                    logger.info(f"New checkpoint: {current.name}")
                    snapshot = self.analyze(current)

                    # Log alerts
                    for alert in snapshot.alerts:
                        log_func = logger.error if alert["severity"] == "critical" else logger.warning
                        log_func(f"ALERT: {alert['message']}")

                    last_checkpoint = current

            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Parameter Stability Monitor")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, help="Specific checkpoint to analyze")
    parser.add_argument("--interval", type=int, default=600, help="Daemon interval")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--trends", action="store_true", help="Show trends")

    args = parser.parse_args()

    monitor = ParameterStabilityMonitor(args.base_dir)

    if args.trends:
        print(json.dumps(monitor.get_trends(), indent=2))
    elif args.daemon:
        monitor.run_daemon(args.interval)
    else:
        snapshot = monitor.analyze(args.checkpoint)
        print(json.dumps(asdict(snapshot), indent=2))


if __name__ == "__main__":
    main()
