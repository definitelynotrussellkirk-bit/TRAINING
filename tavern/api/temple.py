"""
Temple API - Diagnostic rituals HTTP endpoints.

Extracted from tavern/server.py for better organization.
Handles:
- GET /api/temple/rituals - List available rituals
- POST /api/temple/run - Execute a ritual
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_rituals(handler: "TavernHandler"):
    """
    GET /api/temple/rituals - List available rituals.
    """
    try:
        from temple import list_rituals
        rituals = list_rituals()
        handler._send_json({"ok": True, "rituals": rituals})
    except ImportError as e:
        logger.error(f"Temple module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Temple module not available",
            "rituals": {},
        }, 500)
    except Exception as e:
        logger.error(f"Temple rituals error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def serve_run_ritual(handler: "TavernHandler"):
    """
    POST /api/temple/run - Run a ritual.

    Request body:
        {"ritual_id": "quick"}

    Response:
        {"ok": true, "ritual": {...}}
    """
    try:
        from temple import run_ritual, list_rituals

        # Parse request body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        ritual_id = data.get("ritual_id", "quick")

        # Validate ritual exists
        rituals = list_rituals()
        if ritual_id not in rituals:
            handler._send_json({
                "ok": False,
                "error": f"Unknown ritual: {ritual_id}",
                "available": list(rituals.keys()),
            }, 400)
            return

        # Run the ritual
        result = run_ritual(ritual_id)

        # Serialize and return
        handler._send_json({
            "ok": True,
            "ritual": _serialize_result(result),
        })

    except ImportError as e:
        logger.error(f"Temple module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Temple module not available",
        }, 500)
    except json.JSONDecodeError as e:
        handler._send_json({
            "ok": False,
            "error": f"Invalid JSON: {e}",
        }, 400)
    except Exception as e:
        logger.error(f"Temple run error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def _serialize_result(result):
    """Serialize RitualResult for JSON response."""
    return {
        "ritual_id": result.ritual_id,
        "name": result.name,
        "description": result.description,
        "status": result.status,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "finished_at": result.finished_at.isoformat() if result.finished_at else None,
        "duration_ms": result.duration_ms(),
        "checks": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "status": c.status,
                "details": c.details,
                "duration_ms": c.duration_ms(),
            }
            for c in result.checks
        ],
    }
