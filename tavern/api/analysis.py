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


def trigger_layer_stats(handler: "TavernHandler", query: dict, body: dict):
    """
    POST /api/analysis/trigger

    Trigger layer stats analysis for a checkpoint.

    Body params:
    - checkpoint_step: Step number (will find path from ledger)
    - checkpoint_path: Full path (optional, overrides step)
    - campaign_id: Campaign ID
    - hero_id: Hero ID
    - reference_step: Optional reference checkpoint step for drift
    - compute_activations: Whether to compute activation stats (default true)
    """
    import requests
    from core.checkpoint_ledger import get_ledger
    from core.hosts import get_service_url

    campaign_id = body.get("campaign_id")
    hero_id = body.get("hero_id")
    checkpoint_step = body.get("checkpoint_step")
    checkpoint_path = body.get("checkpoint_path")
    reference_step = body.get("reference_step")
    compute_activations = body.get("compute_activations", True)

    if not campaign_id or not hero_id:
        handler._send_json({"error": "campaign_id and hero_id required"}, 400)
        return

    # Resolve checkpoint path from ledger if only step provided
    if not checkpoint_path and checkpoint_step:
        ledger = get_ledger()
        record = ledger.get(checkpoint_step)
        if record and record.get("path"):
            checkpoint_path = record["path"]
        else:
            handler._send_json({"error": f"Checkpoint {checkpoint_step} not found in ledger"}, 404)
            return

    if not checkpoint_path:
        handler._send_json({"error": "checkpoint_step or checkpoint_path required"}, 400)
        return

    # Resolve reference path if provided
    reference_path = None
    if reference_step:
        ledger = get_ledger()
        ref_record = ledger.get(reference_step)
        if ref_record and ref_record.get("path"):
            reference_path = ref_record["path"]

    # Build job payload
    job_payload = {
        "job_type": "layer_stats",
        "payload": {
            "campaign_id": campaign_id,
            "hero_id": hero_id,
            "checkpoint_path": checkpoint_path,
            "compute_activations": compute_activations,
        },
        "priority": "normal",
    }

    if reference_path:
        job_payload["payload"]["reference_checkpoint_path"] = reference_path

    # Submit to job queue
    try:
        vault_url = get_service_url("vault", fallback="http://localhost:8767")
        resp = requests.post(
            f"{vault_url}/api/jobs",
            json=job_payload,
            timeout=10,
        )

        result = resp.json()

        if result.get("accepted"):
            handler._send_json({
                "success": True,
                "job_id": result.get("job_id"),
                "queue_position": result.get("queue_position"),
                "message": f"Layer stats job submitted for step {checkpoint_step or 'custom'}",
            })
        else:
            handler._send_json({
                "success": False,
                "error": result.get("message", "Job submission failed"),
            }, 400)

    except requests.RequestException as e:
        logger.error(f"Failed to submit layer_stats job: {e}")
        handler._send_json({
            "success": False,
            "error": f"Failed to connect to job server: {e}",
        }, 503)


def get_available_checkpoints(handler: "TavernHandler", query: dict):
    """
    GET /api/analysis/checkpoints

    Get list of checkpoints available for analysis.
    Returns checkpoints from ledger with analysis status.
    """
    from core.checkpoint_ledger import get_ledger

    hero_id = query.get("hero_id", [_get_active_hero_id()])[0]
    campaign_id = query.get("campaign_id", [None])[0]
    limit = int(query.get("limit", [20])[0])

    ledger = get_ledger()

    # Get recent checkpoints from ledger
    all_records = ledger.list_all()
    records = sorted(all_records, key=lambda x: x.get("step", 0), reverse=True)[:limit]

    # Check which have analysis
    analysis_dir = None
    if hero_id and campaign_id:
        analysis_dir = _get_analysis_dir(campaign_id, hero_id) / "layer_stats"

    checkpoints = []
    for rec in records:
        step = rec.get("step", 0)
        has_analysis = False

        if analysis_dir and analysis_dir.exists():
            analysis_file = analysis_dir / f"ckpt-{step:06d}.layer_stats.json"
            has_analysis = analysis_file.exists()

        checkpoints.append({
            "step": step,
            "path": rec.get("path"),
            "train_loss": rec.get("train_loss"),
            "created_at": rec.get("created_at"),
            "has_analysis": has_analysis,
        })

    handler._send_json({
        "checkpoints": checkpoints,
        "total": len(all_records),
    })


# =============================================================================
# PRIMITIVE ANALYTICS ENDPOINTS
# =============================================================================

def serve_primitive_weakness_report(handler: "TavernHandler", query: dict):
    """
    GET /api/analysis/primitives/weakness

    Get weakness report showing which cognitive primitives are struggling.

    Query params:
    - campaign_id: Filter to specific campaign
    - hero_id: Filter to specific hero
    - threshold: Accuracy below this is weak (default 0.7)
    """
    try:
        from guild.primitives import get_weakness_report

        campaign_id = query.get("campaign_id", [None])[0]
        hero_id = query.get("hero_id", [None])[0]
        threshold = float(query.get("threshold", [0.7])[0])

        report = get_weakness_report(
            campaign_id=campaign_id,
            hero_id=hero_id,
        )

        handler._send_json(report.to_dict())

    except Exception as e:
        logger.error(f"Primitive weakness report error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_primitive_profile(handler: "TavernHandler", query: dict):
    """
    GET /api/analysis/primitives/profile

    Get complete primitive profile showing stats for all known primitives.

    Query params:
    - campaign_id: Filter to specific campaign
    - hero_id: Filter to specific hero
    """
    try:
        from guild.primitives import get_primitive_profile, PRIMITIVE_CATEGORIES

        campaign_id = query.get("campaign_id", [None])[0]
        hero_id = query.get("hero_id", [None])[0]

        profile = get_primitive_profile(
            campaign_id=campaign_id,
            hero_id=hero_id,
        )

        # Convert to serializable format grouped by category
        by_category = {}
        for prim_id, stats in profile.items():
            cat = stats.category
            if cat not in by_category:
                by_category[cat] = {
                    "name": PRIMITIVE_CATEGORIES.get(cat, cat),
                    "primitives": [],
                }
            by_category[cat]["primitives"].append({
                "id": prim_id,
                "accuracy": stats.accuracy,
                "samples": stats.total_samples,
                "skills": stats.skills_exercised,
                "is_weak": stats.is_weak,
            })

        handler._send_json({
            "profile": by_category,
            "total_primitives": len(profile),
            "campaign_id": campaign_id,
            "hero_id": hero_id,
        })

    except Exception as e:
        logger.error(f"Primitive profile error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_primitive_suggestions(handler: "TavernHandler", query: dict):
    """
    GET /api/analysis/primitives/suggestions

    Get training suggestions to strengthen weak primitives.

    Query params:
    - campaign_id: Filter to specific campaign
    - threshold: Accuracy below this is weak (default 0.7)
    """
    try:
        from guild.primitives import suggest_training_for_weak

        campaign_id = query.get("campaign_id", [None])[0]
        threshold = float(query.get("threshold", [0.7])[0])

        suggestions = suggest_training_for_weak(
            campaign_id=campaign_id,
            threshold=threshold,
        )

        handler._send_json({
            "suggestions": [
                {
                    "primitive": s.primitive_id,
                    "current_accuracy": s.current_accuracy,
                    "suggested_skills": s.suggested_skills,
                    "suggested_levels": s.suggested_levels,
                    "rationale": s.rationale,
                }
                for s in suggestions
            ],
            "count": len(suggestions),
            "campaign_id": campaign_id,
        })

    except Exception as e:
        logger.error(f"Primitive suggestions error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_eval_campaign_summary(handler: "TavernHandler", query: dict):
    """
    GET /api/analysis/eval-summary

    Get comprehensive eval summary for a campaign (uses new campaign_id filtering).

    Query params:
    - campaign_id: Campaign to summarize (required)
    - hero_id: Filter to specific hero
    """
    try:
        from core.evaluation_ledger import get_eval_ledger

        campaign_id = query.get("campaign_id", [None])[0]
        hero_id = query.get("hero_id", [None])[0]

        if not campaign_id:
            handler._send_json({"error": "campaign_id required"}, 400)
            return

        ledger = get_eval_ledger()
        summary = ledger.get_campaign_summary(campaign_id, hero_id)

        handler._send_json(summary)

    except Exception as e:
        logger.error(f"Eval campaign summary error: {e}")
        handler._send_json({"error": str(e)}, 500)
