"""
Storage Plugin - Synology NAS monitoring for unified API

Reads status/storage_status.json and provides storage metrics
for the dashboard.
"""

import json
from pathlib import Path
from datetime import datetime


def get_storage_status(base_dir: str = "/path/to/training") -> dict:
    """
    Get storage status for unified API.

    Returns:
        Dict with storage metrics for dashboard
    """
    status_file = Path(base_dir) / "status" / "storage_status.json"

    if not status_file.exists():
        return {
            "available": False,
            "error": "No storage status file found",
            "hint": "Run: python3 monitoring/storage_manager.py --once"
        }

    try:
        with open(status_file) as f:
            data = json.load(f)

        # Check freshness (warn if > 10 minutes old)
        last_updated = data.get("last_updated") or data.get("timestamp")
        stale = False
        if last_updated:
            try:
                updated = datetime.fromisoformat(last_updated)
                age_seconds = (datetime.now() - updated).total_seconds()
                stale = age_seconds > 600  # 10 minutes
            except:
                pass

        return {
            "available": True,
            "connected": data.get("connected", False),
            "host": data.get("host", "unknown"),
            "health": data.get("health", "unknown"),
            "total_tb": data.get("total_capacity_tb", 0),
            "used_tb": data.get("used_capacity_tb", 0),
            "free_tb": data.get("free_capacity_tb", 0),
            "usage_percent": data.get("usage_percent", 0),
            "volumes": len(data.get("volumes", [])),
            "disks": len(data.get("disks", [])),
            "system": data.get("system", {}),
            "stale": stale,
            "last_updated": last_updated
        }

    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def get_storage_details(base_dir: str = "/path/to/training") -> dict:
    """
    Get detailed storage info including disks and volumes.

    Returns:
        Full storage status data
    """
    status_file = Path(base_dir) / "status" / "storage_status.json"

    if not status_file.exists():
        return {"error": "No status file"}

    try:
        with open(status_file) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# Plugin interface for aggregator
def register(aggregator):
    """Register storage endpoints with the API aggregator"""

    @aggregator.app.route('/api/storage')
    def api_storage():
        from flask import jsonify
        return jsonify(get_storage_status(aggregator.base_dir))

    @aggregator.app.route('/api/storage/details')
    def api_storage_details():
        from flask import jsonify
        return jsonify(get_storage_details(aggregator.base_dir))

    # Add to unified sources
    aggregator.add_source("storage", get_storage_status)
