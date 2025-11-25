"""
Analytics API Plugin - Serves data for the analytics dashboard.

Provides endpoints for:
- Hard example performance
- Error distribution
- Impact tracking
- Correction pipeline status
- Improvement cycle history
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')


def get_base_dir() -> Path:
    """Get the base directory."""
    return Path("/path/to/training")


def read_json_file(filepath: Path) -> Optional[Dict]:
    """Safely read a JSON file."""
    try:
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
    return None


@analytics_bp.route('/overview')
def get_overview():
    """Get complete analytics overview for dashboard."""
    base_dir = get_base_dir()
    status_dir = base_dir / "status"

    result = {
        "timestamp": datetime.now().isoformat(),
        "hard_examples": {},
        "errors": {},
        "training": {},
        "corrections": {},
        "impact": {},
        "cycles": []
    }

    # Hard Example Board
    board_data = read_json_file(status_dir / "hard_example_board.json")
    if board_data:
        entries = board_data.get("entries", [])
        if entries:
            latest = entries[-1]
            result["hard_examples"] = {
                "accuracy": latest.get("accuracy", 0),
                "correct": latest.get("total_correct", 0),
                "total": latest.get("total", 10),
                "step": latest.get("step", 0),
                "error_types": latest.get("error_types", {}),
                "history": [
                    {
                        "step": e.get("step", 0),
                        "accuracy": e.get("accuracy", 0)
                    }
                    for e in entries[-10:]
                ]
            }
            result["errors"] = latest.get("error_types", {})

    # Training Status
    training_data = read_json_file(status_dir / "training_status.json")
    if training_data:
        result["training"] = {
            "step": training_data.get("current_step", 0),
            "train_loss": training_data.get("loss", 0),
            "val_loss": training_data.get("validation_loss", 0),
            "status": training_data.get("status", "unknown"),
            "dataset": training_data.get("current_dataset")
        }

    # Correction Sync Status
    sync_data = read_json_file(status_dir / "correction_sync.json")
    if sync_data:
        result["corrections"]["synced_total"] = sync_data.get("total_synced", 0)
        result["corrections"]["recent"] = [
            r.get("filename") for r in sync_data.get("records", [])[-5:]
        ]

    # Count inbox corrections
    inbox_dir = base_dir / "inbox"
    if inbox_dir.exists():
        result["corrections"]["pending_local"] = len(
            list(inbox_dir.glob("corrections_*.jsonl"))
        )

    # Impact Tracker
    impact_data = read_json_file(status_dir / "impact_tracker.json")
    if impact_data:
        snapshots = impact_data.get("snapshots", [])
        if snapshots:
            first = snapshots[0]
            latest = snapshots[-1]
            result["impact"] = {
                "snapshots_count": len(snapshots),
                "first_accuracy": first.get("hard_example_accuracy", 0),
                "latest_accuracy": latest.get("hard_example_accuracy", 0),
                "change": latest.get("hard_example_accuracy", 0) - first.get("hard_example_accuracy", 0)
            }

    # Auto-Improve Cycles
    auto_improve = read_json_file(status_dir / "auto_improve.json")
    if auto_improve:
        result["cycles"] = [
            {
                "id": c.get("cycle_id"),
                "accuracy": c.get("baseline_accuracy", 0),
                "corrections": c.get("corrections_generated", 0),
                "status": c.get("status")
            }
            for c in auto_improve.get("cycles", [])[-5:]
        ]

    return jsonify(result)


@analytics_bp.route('/hard-examples')
def get_hard_examples():
    """Get hard example evaluation details."""
    base_dir = get_base_dir()
    status_dir = base_dir / "status"

    board_data = read_json_file(status_dir / "hard_example_board.json")
    examples_data = read_json_file(base_dir / "config" / "hard_examples.json")

    result = {
        "examples": examples_data or [],
        "board": []
    }

    if board_data:
        entries = board_data.get("entries", [])
        for entry in entries[-5:]:
            result["board"].append({
                "step": entry.get("step", 0),
                "timestamp": entry.get("timestamp"),
                "accuracy": entry.get("accuracy", 0),
                "results": entry.get("results", {}),
                "error_types": entry.get("error_types", {})
            })

    return jsonify(result)


@analytics_bp.route('/corrections')
def get_corrections():
    """Get correction pipeline details."""
    base_dir = get_base_dir()
    status_dir = base_dir / "status"
    inbox_dir = base_dir / "inbox"

    result = {
        "pipeline": {},
        "recent_files": [],
        "by_type": {}
    }

    # Sync status
    sync_data = read_json_file(status_dir / "correction_sync.json")
    if sync_data:
        result["pipeline"]["total_synced"] = sync_data.get("total_synced", 0)
        result["pipeline"]["last_sync"] = sync_data.get("last_updated")

        for record in sync_data.get("records", [])[-10:]:
            result["recent_files"].append({
                "filename": record.get("filename"),
                "examples": record.get("examples_count", 0),
                "synced_at": record.get("synced_at")
            })

    # Count by type
    if inbox_dir.exists():
        for jsonl in inbox_dir.glob("corrections_*.jsonl"):
            # Extract type from filename
            name = jsonl.stem
            parts = name.split("_")
            if len(parts) >= 2:
                error_type = parts[1]
                result["by_type"][error_type] = result["by_type"].get(error_type, 0) + 1

    return jsonify(result)


@analytics_bp.route('/impact')
def get_impact():
    """Get impact tracking details."""
    base_dir = get_base_dir()
    status_dir = base_dir / "status"

    impact_data = read_json_file(status_dir / "impact_tracker.json")

    result = {
        "snapshots": [],
        "summary": {}
    }

    if impact_data:
        snapshots = impact_data.get("snapshots", [])

        for snap in snapshots:
            result["snapshots"].append({
                "timestamp": snap.get("timestamp"),
                "step": snap.get("step", 0),
                "accuracy": snap.get("hard_example_accuracy", 0),
                "errors": snap.get("error_distribution", {}),
                "notes": snap.get("notes")
            })

        if len(snapshots) >= 2:
            first = snapshots[0]
            latest = snapshots[-1]
            result["summary"] = {
                "first_accuracy": first.get("hard_example_accuracy", 0),
                "latest_accuracy": latest.get("hard_example_accuracy", 0),
                "change": latest.get("hard_example_accuracy", 0) - first.get("hard_example_accuracy", 0),
                "steps_trained": latest.get("step", 0) - first.get("step", 0),
                "improved": latest.get("hard_example_accuracy", 0) > first.get("hard_example_accuracy", 0)
            }

    return jsonify(result)


@analytics_bp.route('/cycles')
def get_cycles():
    """Get improvement cycle history."""
    base_dir = get_base_dir()
    status_dir = base_dir / "status"

    auto_improve = read_json_file(status_dir / "auto_improve.json")

    result = {
        "cycles": [],
        "daemon_status": "unknown",
        "next_cycle_in": None
    }

    if auto_improve:
        result["daemon_status"] = auto_improve.get("daemon_status", "unknown")

        for cycle in auto_improve.get("cycles", []):
            result["cycles"].append({
                "id": cycle.get("cycle_id"),
                "started_at": cycle.get("started_at"),
                "completed_at": cycle.get("completed_at"),
                "baseline_accuracy": cycle.get("baseline_accuracy"),
                "errors_found": cycle.get("errors_found", 0),
                "corrections_generated": cycle.get("corrections_generated", 0),
                "corrections_synced": cycle.get("corrections_synced", 0),
                "status": cycle.get("status")
            })

    return jsonify(result)


def register_plugin(app):
    """Register the analytics plugin with the Flask app."""
    app.register_blueprint(analytics_bp)
    logger.info("Analytics API plugin registered")
