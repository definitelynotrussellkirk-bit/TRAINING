"""
Momentum Engine API

Exposes forward progress state and blockers.
Handles:
- /api/momentum - Get current momentum state with blockers (runs checks)
- /api/momentum/check - Force run all momentum checks
"""

import logging
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from core.momentum import get_momentum_state, run_momentum_checks, get_daemon_status

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_momentum(handler: "TavernHandler", run_checks: bool = True):
    """
    GET /api/momentum - Current momentum state.

    Query params:
        ?check=false - Skip running checks, just return current state

    Returns:
        {
            "status": "go" | "blocked" | "idle",
            "primary_blocker": {...} | null,
            "blockers": {...},
            "blocker_count": int,
            "daemon": {
                "running": bool,
                "pid": int | null,
                "stale": bool
            }
        }
    """
    try:
        # Parse query params
        query = {}
        if "?" in handler.path:
            query = parse_qs(handler.path.split("?")[1])

        # Run checks by default, unless ?check=false
        should_check = query.get("check", ["true"])[0].lower() != "false"

        if should_check and run_checks:
            state = run_momentum_checks()
        else:
            state = get_momentum_state()

        # Build response
        response = state.to_dict()

        # Add daemon status - critical for honest UI
        daemon = get_daemon_status()
        response["daemon"] = {
            "running": daemon["running"],
            "pid": daemon["pid"],
            "stale": daemon["stale"],
        }

        # If daemon not running, override status display
        if not daemon["running"]:
            response["daemon_message"] = "Training daemon is not running"

        handler._send_json(response)
    except Exception as e:
        logger.error(f"Momentum API error: {e}")
        handler._send_json({
            "status": "idle",
            "primary_blocker": None,
            "blockers": {},
            "blocker_count": 0,
            "daemon": {"running": False, "pid": None, "stale": False},
            "error": str(e)
        })
