"""
Run Context API - Single source of truth for current training run.

Handles:
- GET /api/run-context - Get unified run context

This endpoint replaces the need to merge data from multiple sources:
- /config (root config.json)
- /api/hero-model-info
- /api/active-campaign

All consumers should use /api/run-context for consistent model/path info.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_run_context(handler: "TavernHandler"):
    """
    GET /api/run-context - Get the unified run context.

    Returns a single JSON with all training context:
    - hero_id, hero_name, hero_icon
    - campaign_id, campaign_name, campaign_path
    - model_path, current_model_dir, base_model
    - locked: {base_model, architecture, context_length, vocab_size}
    - daemon: {running, pid, last_heartbeat}
    - auto_run, auto_generate
    - is_legacy_mode, is_first_run

    This is the single source of truth for "what are we training?"
    """
    try:
        from core.run_context import get_run_context, validate_run_context

        ctx = get_run_context()
        data = ctx.to_dict()

        # Add validation warnings
        errors = validate_run_context(ctx)
        if errors:
            data["_warnings"] = errors

        handler._send_json(data)

    except ImportError as e:
        logger.error(f"RunContext module not available: {e}")
        handler._send_json({
            "error": "RunContext system not installed",
            "is_legacy_mode": True,
            "is_first_run": True,
        }, 503)
    except Exception as e:
        logger.error(f"Run context error: {e}")
        handler._send_json({
            "error": str(e),
            "is_legacy_mode": True,
            "is_first_run": True,
        }, 500)
