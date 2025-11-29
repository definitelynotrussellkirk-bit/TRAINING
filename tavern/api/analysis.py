"""
Analysis API - Model Archaeology

Extracted from tavern/server.py for better organization.
Handles:
- /api/analysis/layer-stats/list/{campaign_id} - List layer stats
- /api/analysis/layer-stats/{campaign_id}/{checkpoint} - Get layer stats detail
- /api/analysis/drift-timeline/{campaign_id} - Drift time series
- /api/analysis/top-movers/{campaign_id} - Most changed layers
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core import paths

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def _get_active_hero_id() -> Optional[str]:
    """Get the active hero_id from the current campaign."""
    try:
        campaign_file = paths.get_control_dir() / "active_campaign.json"
        if campaign_file.exists():
            with open(campaign_file) as f:
                data = json.load(f)
                return data.get("hero_id")
    except Exception as e:
        logger.warning(f"Failed to get active hero_id: {e}")
    return None


def _get_analysis_dir(campaign_id: str, hero_id: Optional[str] = None) -> Path:
    """Get the analysis directory for a campaign."""
    if not hero_id:
        hero_id = _get_active_hero_id()
    if not hero_id:
        logger.warning("No active hero, cannot resolve analysis dir")
        return paths.get_campaigns_dir() / "unknown" / campaign_id / "analysis"
    return paths.get_campaign_analysis_dir(hero_id, campaign_id)


def serve_layer_stats_list(handler: "TavernHandler", campaign_id: str, query: dict):
    """
    GET /api/analysis/layer-stats/list/{campaign_id}

    List available layer stats for a campaign.
    Query params:
    - hero_id: Override hero (defaults to active)
    """
    hero_id = query.get("hero_id", [_get_active_hero_id()])[0]
    analysis_dir = _get_analysis_dir(campaign_id, hero_id) / "layer_stats"

    if not analysis_dir.exists():
        handler._send_json({"stats": [], "count": 0, "campaign_id": campaign_id})
        return

    try:
        stats = []
        for f in sorted(analysis_dir.glob("*.layer_stats.json")):
            # Read summary fields only
            with open(f) as fp:
                data = json.load(fp)

            stats.append({
                "checkpoint_step": data.get("checkpoint_step", 0),
                "created_at": data.get("created_at"),
                "has_drift": bool(data.get("drift_stats")),
                "has_activations": bool(data.get("activation_stats")),
                "num_layers": len(data.get("weight_stats", {})),
                "most_changed_layer": (
                    data.get("global_drift_stats", {}).get("most_changed_layer")
                ),
                "avg_weight_norm": (
                    data.get("global_weight_stats", {}).get("avg_weight_norm")
                ),
                "compute_duration_sec": data.get("compute_duration_sec", 0),
                "filename": f.name,
            })

        handler._send_json({
            "stats": stats,
            "count": len(stats),
            "campaign_id": campaign_id,
            "hero_id": hero_id,
        })

    except Exception as e:
        logger.error(f"Layer stats list error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_layer_stats_detail(
    handler: "TavernHandler",
    campaign_id: str,
    checkpoint_name: str,
    query: dict,
):
    """
    GET /api/analysis/layer-stats/{campaign_id}/{checkpoint}

    Get full layer stats for a specific checkpoint.
    checkpoint can be step number (183000) or filename.
    """
    hero_id = query.get("hero_id", [_get_active_hero_id()])[0]
    analysis_dir = _get_analysis_dir(campaign_id, hero_id) / "layer_stats"

    # Handle both "183000" and "ckpt-183000.layer_stats.json" formats
    if checkpoint_name.isdigit():
        filename = f"ckpt-{int(checkpoint_name):06d}.layer_stats.json"
    else:
        filename = checkpoint_name

    filepath = analysis_dir / filename

    if not filepath.exists():
        handler._send_json({"error": f"Not found: {filename}"}, 404)
        return

    try:
        with open(filepath) as f:
            data = json.load(f)
        handler._send_json(data)

    except Exception as e:
        logger.error(f"Layer stats detail error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_drift_timeline(handler: "TavernHandler", campaign_id: str, query: dict):
    """
    GET /api/analysis/drift-timeline/{campaign_id}

    Get drift time series for visualization.
    Returns per-layer drift over checkpoints for heatmap/chart display.

    Query params:
    - hero_id: Override hero
    - layers: Comma-separated layer name filters
    - max_layers: Limit layers (default 30)
    """
    hero_id = query.get("hero_id", [_get_active_hero_id()])[0]
    layer_filter = query.get("layers", [None])[0]
    max_layers = int(query.get("max_layers", [30])[0])

    analysis_dir = _get_analysis_dir(campaign_id, hero_id) / "layer_stats"

    if not analysis_dir.exists():
        handler._send_json({"error": "No analysis data found"}, 404)
        return

    try:
        timeline = {
            "checkpoints": [],
            "layers": {},
        }

        # Collect all layer stats with drift data
        for f in sorted(analysis_dir.glob("*.layer_stats.json")):
            with open(f) as fp:
                data = json.load(fp)

            if not data.get("drift_stats"):
                continue

            step = data.get("checkpoint_step", 0)
            timeline["checkpoints"].append(step)

            for layer_name, drift in data["drift_stats"].items():
                # Apply layer filter if specified
                if layer_filter:
                    allowed = layer_filter.split(",")
                    if not any(a in layer_name for a in allowed):
                        continue

                if layer_name not in timeline["layers"]:
                    timeline["layers"][layer_name] = {
                        "name": layer_name,
                        "drift_l2": [],
                        "drift_cosine": [],
                    }

                timeline["layers"][layer_name]["drift_l2"].append(
                    drift.get("total_l2", 0)
                )
                timeline["layers"][layer_name]["drift_cosine"].append(
                    drift.get("avg_cosine", 1.0)
                )

        # Limit number of layers (sort by total drift, keep top N)
        if len(timeline["layers"]) > max_layers:
            sorted_layers = sorted(
                timeline["layers"].items(),
                key=lambda x: sum(x[1]["drift_l2"]),
                reverse=True
            )
            timeline["layers"] = dict(sorted_layers[:max_layers])

        handler._send_json(timeline)

    except Exception as e:
        logger.error(f"Drift timeline error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_top_movers(handler: "TavernHandler", campaign_id: str, query: dict):
    """
    GET /api/analysis/top-movers/{campaign_id}

    Get layers that changed the most across training.

    Query params:
    - hero_id: Override hero
    - n: Number of top/bottom layers (default 10)
    """
    hero_id = query.get("hero_id", [_get_active_hero_id()])[0]
    top_n = int(query.get("n", [10])[0])

    analysis_dir = _get_analysis_dir(campaign_id, hero_id) / "layer_stats"

    if not analysis_dir.exists():
        handler._send_json({"error": "No analysis data found"}, 404)
        return

    try:
        # Accumulate total drift per layer
        layer_drift = {}
        checkpoint_count = 0

        for f in sorted(analysis_dir.glob("*.layer_stats.json")):
            with open(f) as fp:
                data = json.load(fp)

            if not data.get("drift_stats"):
                continue

            checkpoint_count += 1

            for layer_name, drift in data["drift_stats"].items():
                if layer_name not in layer_drift:
                    layer_drift[layer_name] = 0
                layer_drift[layer_name] += drift.get("total_l2", 0)

        # Sort by total drift
        sorted_layers = sorted(
            layer_drift.items(),
            key=lambda x: x[1],
            reverse=True
        )

        handler._send_json({
            "top_movers": [
                {"layer": name, "total_drift": drift}
                for name, drift in sorted_layers[:top_n]
            ],
            "most_stable": [
                {"layer": name, "total_drift": drift}
                for name, drift in sorted_layers[-top_n:][::-1]
            ],
            "total_layers": len(layer_drift),
            "checkpoints_analyzed": checkpoint_count,
            "campaign_id": campaign_id,
            "hero_id": hero_id,
        })

    except Exception as e:
        logger.error(f"Top movers error: {e}")
        handler._send_json({"error": str(e)}, 500)
