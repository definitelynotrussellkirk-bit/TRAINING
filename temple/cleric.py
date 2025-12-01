"""
Temple Cleric - Ritual orchestration and registry.

The Cleric manages the registry of available rituals and orchestrates
their execution, collecting results and computing aggregate status.
"""

import logging
from datetime import datetime
from typing import Callable, Dict, List

from .schemas import RitualResult, RitualCheckResult, ResultStatus

logger = logging.getLogger(__name__)

# Registry of available rituals
RITUALS: Dict[str, Callable[[], List[RitualCheckResult]]] = {}
RITUAL_META: Dict[str, Dict[str, str]] = {}


def register_ritual(ritual_id: str, name: str, description: str):
    """
    Decorator to register a ritual function.

    Usage:
        @register_ritual("quick", "Ritual of Quick", "Fast sanity checks")
        def run() -> List[RitualCheckResult]:
            return [...]
    """
    def decorator(fn: Callable[[], List[RitualCheckResult]]):
        RITUALS[ritual_id] = fn
        RITUAL_META[ritual_id] = {"name": name, "description": description}
        return fn
    return decorator


def list_rituals() -> Dict[str, Dict[str, str]]:
    """Return metadata for all registered rituals."""
    # Import ritual modules to trigger registration
    _ensure_rituals_loaded()

    return {
        rid: {"id": rid, **meta}
        for rid, meta in RITUAL_META.items()
    }


def run_ritual(ritual_id: str) -> RitualResult:
    """
    Run a ritual and return aggregated results.

    Args:
        ritual_id: ID of the ritual to run (e.g., "quick", "api")

    Returns:
        RitualResult with all check outcomes

    Raises:
        ValueError: If ritual_id is not registered
    """
    # Ensure rituals are loaded
    _ensure_rituals_loaded()

    if ritual_id not in RITUALS:
        raise ValueError(f"Unknown ritual: {ritual_id}")

    meta = RITUAL_META[ritual_id]
    started = datetime.utcnow()

    try:
        checks = RITUALS[ritual_id]()
    except Exception as e:
        logger.error(f"Ritual {ritual_id} failed: {e}")
        checks = [RitualCheckResult(
            id="ritual_error",
            name="Ritual Error",
            description="The ritual failed to execute",
            status="fail",
            details={"error": str(e)},
            started_at=started,
            finished_at=datetime.utcnow(),
        )]

    # Derive overall status from individual checks
    if any(c.status == "fail" for c in checks):
        status: ResultStatus = "fail"
    elif any(c.status == "warn" for c in checks):
        status = "warn"
    elif all(c.status == "skip" for c in checks):
        status = "skip"
    else:
        status = "ok"

    finished = datetime.utcnow()

    return RitualResult(
        ritual_id=ritual_id,
        name=meta["name"],
        description=meta["description"],
        status=status,
        checks=checks,
        started_at=started,
        finished_at=finished,
    )


def _ensure_rituals_loaded():
    """Import ritual modules to ensure they're registered."""
    if not RITUALS:
        # Import ritual modules - they register themselves via decorator
        try:
            from temple.rituals import quick  # noqa: F401
        except ImportError as e:
            logger.warning(f"Failed to load quick ritual: {e}")

        try:
            from temple.rituals import api  # noqa: F401
        except ImportError as e:
            logger.warning(f"Failed to load api ritual: {e}")
