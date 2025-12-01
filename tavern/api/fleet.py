"""
Fleet API - Node management and monitoring endpoints.

Extracted from tavern/server.py for better organization.
Handles:
- GET /api/fleet/status - Full fleet status (all nodes)
- GET /api/fleet/node/<host_id> - Specific node status
- POST /api/fleet/retention/<host_id> - Trigger retention on a node
- POST /api/fleet/retention/all - Trigger retention on all critical nodes
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_fleet_status(handler: "TavernHandler"):
    """
    GET /api/fleet/status - Get full fleet status.

    Returns health information for all nodes in the fleet.
    """
    try:
        from fleet.controller import get_fleet_status

        status = get_fleet_status(max_age_seconds=30)
        handler._send_json({
            "ok": True,
            "fleet": status.to_dict(),
        })
    except ImportError as e:
        logger.error(f"Fleet module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Fleet module not available",
            "fleet": None,
        }, 500)
    except Exception as e:
        logger.error(f"Fleet status error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def serve_node_status(handler: "TavernHandler", host_id: str):
    """
    GET /api/fleet/node/<host_id> - Get status for a specific node.

    Args:
        handler: The request handler
        host_id: The host ID (e.g., "4090", "3090")
    """
    try:
        from fleet.controller import get_node_health

        health = get_node_health(host_id)
        if health:
            handler._send_json({
                "ok": True,
                "node": health.to_dict(),
            })
        else:
            handler._send_json({
                "ok": False,
                "error": f"Node not found or offline: {host_id}",
            }, 404)

    except ImportError as e:
        logger.error(f"Fleet module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Fleet module not available",
        }, 500)
    except Exception as e:
        logger.error(f"Node status error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def serve_trigger_retention(handler: "TavernHandler", host_id: str):
    """
    POST /api/fleet/retention/<host_id> - Trigger retention on a node.

    Request body (optional):
        {"dry_run": true}  # If true, only report what would be deleted

    Args:
        handler: The request handler
        host_id: The host ID (e.g., "3090")
    """
    try:
        from fleet.controller import trigger_retention

        # Parse request body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        dry_run = data.get("dry_run", False)

        result = trigger_retention(host_id, dry_run=dry_run)
        if result:
            handler._send_json({
                "ok": True,
                "result": result.to_dict(),
            })
        else:
            handler._send_json({
                "ok": False,
                "error": f"Failed to trigger retention on {host_id}",
            }, 500)

    except ImportError as e:
        logger.error(f"Fleet module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Fleet module not available",
        }, 500)
    except Exception as e:
        logger.error(f"Retention trigger error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def serve_trigger_retention_all(handler: "TavernHandler"):
    """
    POST /api/fleet/retention/all - Trigger retention on all critical nodes.

    Request body (optional):
        {"dry_run": true, "only_critical": true}
    """
    try:
        from fleet.controller import get_controller

        # Parse request body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        dry_run = data.get("dry_run", False)
        only_critical = data.get("only_critical", True)

        controller = get_controller()
        results = controller.trigger_retention_all(
            dry_run=dry_run,
            only_critical=only_critical,
        )

        # Convert results to dict
        results_dict = {
            host_id: r.to_dict() if r else None
            for host_id, r in results.items()
        }

        handler._send_json({
            "ok": True,
            "results": results_dict,
            "triggered_count": len([r for r in results.values() if r]),
        })

    except ImportError as e:
        logger.error(f"Fleet module not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Fleet module not available",
        }, 500)
    except Exception as e:
        logger.error(f"Retention all error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)


def serve_local_agent_status(handler: "TavernHandler"):
    """
    GET /api/fleet/local - Get local node agent status.

    Returns health for the local node without going through the controller.
    Useful for checking the local agent directly.
    """
    try:
        from fleet.agent import NodeAgent

        agent = NodeAgent()
        health = agent.get_health()

        handler._send_json({
            "ok": True,
            "local": health.to_dict(),
        })

    except ImportError as e:
        logger.error(f"Fleet agent not available: {e}")
        handler._send_json({
            "ok": False,
            "error": "Fleet agent not available",
        }, 500)
    except Exception as e:
        logger.error(f"Local agent status error: {e}")
        handler._send_json({"ok": False, "error": str(e)}, 500)
