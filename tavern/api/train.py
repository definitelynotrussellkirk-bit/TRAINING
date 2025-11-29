"""
Training API - Start training sessions from the UI.

Handles:
- POST /api/train - Start a training session for the active campaign

The endpoint writes a train request file that the training daemon picks up.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core.momentum import (
    report_blocker,
    clear_blocker,
    BLOCKER_NO_ACTIVE_CAMPAIGN,
)
from core.paths import get_control_dir
from guild.campaigns.loader import load_active_campaign

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def _get_train_request_file() -> Path:
    """Get path to train request file."""
    return get_control_dir() / "train_request.json"


def serve_train_post(handler: "TavernHandler"):
    """
    POST /api/train - Request a training session.

    Body:
        {"steps": 2000}

    Creates a train_request.json file that the daemon picks up.
    """
    try:
        # Parse request body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length).decode()) if content_length else {}
        steps = int(body.get("steps", 1000))

        # Validate steps
        if steps < 100:
            handler._send_json({
                "ok": False,
                "error": "Steps must be at least 100",
            }, 400)
            return

        if steps > 100000:
            handler._send_json({
                "ok": False,
                "error": "Steps cannot exceed 100,000 per request",
            }, 400)
            return

        # Check for active campaign
        campaign = load_active_campaign()
        if campaign is None:
            report_blocker(
                code=BLOCKER_NO_ACTIVE_CAMPAIGN,
                what=f"Start a {steps:,}-step training session",
                why="There is no active campaign selected.",
                how_to_fix="Go to the Campaign page, create or select a campaign, and mark it active.",
                suggested_action="open_campaign",
            )
            handler._send_json({
                "ok": False,
                "error": "NO_ACTIVE_CAMPAIGN",
                "message": "No active campaign. Create or select one first.",
            }, 400)
            return

        # Clear the blocker since we have a campaign
        clear_blocker(BLOCKER_NO_ACTIVE_CAMPAIGN)

        # Create the train request
        request = {
            "hero_id": campaign.hero_id,
            "campaign_id": campaign.id,
            "steps": steps,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }

        # Write the request file
        request_file = _get_train_request_file()
        request_file.parent.mkdir(parents=True, exist_ok=True)
        request_file.write_text(json.dumps(request, indent=2))

        logger.info(f"[Train API] Created train request: {steps} steps for {campaign.id}")

        handler._send_json({
            "ok": True,
            "message": f"Training request submitted: {steps:,} steps",
            "request": request,
            "request_file": str(request_file),
        })

    except json.JSONDecodeError:
        handler._send_json({
            "ok": False,
            "error": "Invalid JSON body",
        }, 400)
    except Exception as e:
        logger.error(f"Train API error: {e}")
        handler._send_json({
            "ok": False,
            "error": str(e),
        }, 500)


def serve_train_status(handler: "TavernHandler"):
    """
    GET /api/train - Get current train request status.
    """
    try:
        request_file = _get_train_request_file()

        if not request_file.exists():
            handler._send_json({
                "has_request": False,
                "request": None,
            })
            return

        request = json.loads(request_file.read_text())
        handler._send_json({
            "has_request": True,
            "request": request,
        })

    except Exception as e:
        logger.error(f"Train status error: {e}")
        handler._send_json({
            "has_request": False,
            "error": str(e),
        })
