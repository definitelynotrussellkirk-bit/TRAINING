"""
Temple Cleric - Ritual orchestration and registry.

The Cleric manages the registry of available rituals and orchestrates
their execution, collecting results and computing aggregate status.

Supports both:
- Legacy rituals: @register_ritual("id", "name", "desc") with run() -> List[RitualCheckResult]
- Protocol rituals: @register_ritual(RitualMeta(...)) with run(ctx) -> List[RitualCheckResult]
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .schemas import RitualResult, RitualCheckResult, ResultStatus
from .protocols import (
    RitualMeta,
    RitualKind,
    RitualContext,
    RitualFunc,
    RITUAL_REGISTRY,
)

logger = logging.getLogger(__name__)

# Legacy registry (for backward compatibility)
RITUALS: Dict[str, Callable[[], List[RitualCheckResult]]] = {}
RITUAL_META: Dict[str, Dict[str, str]] = {}

# Default metadata for legacy rituals
_LEGACY_RITUAL_DEFAULTS = {
    "quick": {"kind": RitualKind.HEALTHCHECK, "icon": "⚡"},
    "api": {"kind": RitualKind.HEALTHCHECK, "icon": "🌐"},
    "forge": {"kind": RitualKind.TRAINING, "icon": "🔥"},
    "weaver": {"kind": RitualKind.INFRA, "icon": "🕸️"},
    "champion": {"kind": RitualKind.TRAINING, "icon": "🏆"},
    "oracle": {"kind": RitualKind.HEALTHCHECK, "icon": "🔮"},
    "guild": {"kind": RitualKind.INFRA, "icon": "⚔️"},
    "scribe": {"kind": RitualKind.EVAL, "icon": "📜"},
    "deep": {"kind": RitualKind.META, "icon": "🌊"},
}


def register_ritual(
    ritual_id_or_meta: Union[str, RitualMeta],
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Decorator to register a ritual function.

    Supports two styles:

    Legacy style (backward compatible):
        @register_ritual("quick", "Ritual of Quick", "Fast sanity checks")
        def run() -> List[RitualCheckResult]:
            return [...]

    Protocol style (new):
        @register_ritual(RitualMeta(
            id="quick",
            name="Ritual of Quick",
            description="Fast sanity checks",
            kind=RitualKind.HEALTHCHECK,
            icon="⚡"
        ))
        def run(ctx: RitualContext) -> List[RitualCheckResult]:
            return [...]
    """
    # Protocol style: RitualMeta passed directly
    if isinstance(ritual_id_or_meta, RitualMeta):
        meta = ritual_id_or_meta
        return RITUAL_REGISTRY.register(meta)

    # Legacy style: string ID with name and description
    ritual_id = ritual_id_or_meta
    if name is None or description is None:
        raise ValueError("Legacy style requires name and description")

    def decorator(fn: Callable[[], List[RitualCheckResult]]):
        # Register in legacy registry
        RITUALS[ritual_id] = fn
        RITUAL_META[ritual_id] = {"name": name, "description": description}

        # Also register in new registry with wrapper
        defaults = _LEGACY_RITUAL_DEFAULTS.get(ritual_id, {})
        meta = RitualMeta(
            id=ritual_id,
            name=name,
            description=description,
            kind=defaults.get("kind", RitualKind.HEALTHCHECK),
            icon=defaults.get("icon", "🔮"),
        )

        # Wrap legacy function to accept context
        def wrapper(ctx: RitualContext) -> List[RitualCheckResult]:
            return fn()

        RITUAL_REGISTRY._rituals[meta.id] = wrapper
        RITUAL_REGISTRY._meta[meta.id] = meta

        return fn

    return decorator


def create_context(
    dry_run: bool = False,
    campaign_id: Optional[str] = None,
    hero_id: Optional[str] = None,
) -> RitualContext:
    """
    Create a RitualContext for ritual execution.

    Args:
        dry_run: If True, rituals should not make changes
        campaign_id: Active campaign ID
        hero_id: Active hero ID

    Returns:
        RitualContext with dependencies injected
    """
    from core.paths import get_base_dir

    base_dir = get_base_dir()

    # Load active campaign if not specified
    if campaign_id is None or hero_id is None:
        try:
            import json
            campaign_file = base_dir / "control" / "active_campaign.json"
            if campaign_file.exists():
                with open(campaign_file) as f:
                    data = json.load(f)
                campaign_id = campaign_id or data.get("campaign_id")
                hero_id = hero_id or data.get("hero_id")
        except Exception:
            pass

    # Load config
    config = {}
    try:
        import json
        config_file = base_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
    except Exception:
        pass

    return RitualContext(
        run_id=str(uuid.uuid4())[:8],
        base_dir=base_dir,
        dry_run=dry_run,
        logger=logger,
        config=config,
        campaign_id=campaign_id,
        hero_id=hero_id,
    )


def list_rituals() -> Dict[str, Dict[str, str]]:
    """
    Return metadata for all registered rituals.

    Returns dict mapping ritual_id to metadata dict with:
    - id: Ritual ID
    - name: Display name
    - description: One-line description
    - kind: RitualKind value (if available)
    - icon: Icon/emoji (if available)
    """
    _ensure_rituals_loaded()

    result = {}
    for rid, meta in RITUAL_REGISTRY._meta.items():
        result[rid] = {
            "id": rid,
            "name": meta.name,
            "description": meta.description,
            "kind": meta.kind.value,
            "icon": meta.icon,
            "requires": list(meta.requires),
            "tags": list(meta.tags),
        }

    return result


def get_ritual_meta(ritual_id: str) -> Optional[RitualMeta]:
    """Get RitualMeta for a specific ritual."""
    _ensure_rituals_loaded()
    return RITUAL_REGISTRY.get_meta(ritual_id)


def run_ritual(
    ritual_id: str,
    ctx: Optional[RitualContext] = None,
) -> RitualResult:
    """
    Run a ritual and return aggregated results.

    Args:
        ritual_id: ID of the ritual to run (e.g., "quick", "api")
        ctx: Optional RitualContext (created if not provided)

    Returns:
        RitualResult with all check outcomes

    Raises:
        ValueError: If ritual_id is not registered
    """
    _ensure_rituals_loaded()

    # Get ritual function
    ritual_fn = RITUAL_REGISTRY.get(ritual_id)
    if ritual_fn is None:
        # Try legacy registry
        if ritual_id in RITUALS:
            ritual_fn = lambda c: RITUALS[ritual_id]()
        else:
            raise ValueError(f"Unknown ritual: {ritual_id}")

    meta = RITUAL_REGISTRY.get_meta(ritual_id)
    if meta is None:
        # Fall back to legacy meta
        legacy_meta = RITUAL_META.get(ritual_id, {})
        meta_dict = {
            "name": legacy_meta.get("name", ritual_id),
            "description": legacy_meta.get("description", ""),
        }
    else:
        meta_dict = {
            "name": meta.name,
            "description": meta.description,
        }

    # Create context if not provided
    if ctx is None:
        ctx = create_context()

    started = datetime.utcnow()

    try:
        checks = ritual_fn(ctx)
    except Exception as e:
        logger.error(f"Ritual {ritual_id} failed: {e}")
        import traceback
        traceback.print_exc()
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
        name=meta_dict["name"],
        description=meta_dict["description"],
        status=status,
        checks=checks,
        started_at=started,
        finished_at=finished,
    )


def run_ceremony(
    ritual_ids: Optional[List[str]] = None,
    ctx: Optional[RitualContext] = None,
    stop_on_fail: bool = False,
) -> Dict[str, RitualResult]:
    """
    Run multiple rituals in dependency order.

    Args:
        ritual_ids: List of ritual IDs to run (None = all)
        ctx: Optional RitualContext (created if not provided)
        stop_on_fail: If True, stop on first failure

    Returns:
        Dict mapping ritual_id to RitualResult
    """
    _ensure_rituals_loaded()

    if ctx is None:
        ctx = create_context()

    if ritual_ids is None:
        ritual_ids = list(RITUAL_REGISTRY._rituals.keys())

    # Sort by dependencies
    try:
        ordered = RITUAL_REGISTRY.get_dependency_order(ritual_ids)
    except ValueError as e:
        logger.error(f"Dependency error: {e}")
        ordered = ritual_ids

    results = {}
    for rid in ordered:
        result = run_ritual(rid, ctx)
        results[rid] = result

        if stop_on_fail and result.status == "fail":
            logger.warning(f"Stopping ceremony due to {rid} failure")
            break

    return results


def get_ceremony_status(results: Dict[str, RitualResult]) -> ResultStatus:
    """
    Compute aggregate status from ceremony results.

    Returns:
        "ok" if all passed, "warn" if any warned, "fail" if any failed
    """
    if any(r.status == "fail" for r in results.values()):
        return "fail"
    if any(r.status == "warn" for r in results.values()):
        return "warn"
    if all(r.status == "skip" for r in results.values()):
        return "skip"
    return "ok"


def _ensure_rituals_loaded():
    """Import ritual modules to ensure they're registered."""
    if not RITUAL_REGISTRY._rituals and not RITUALS:
        # Import ritual modules - they register themselves via decorator
        ritual_modules = [
            "quick",      # Core service checks
            "api",        # HTTP API validation
            "forge",      # GPU/hardware diagnostics
            "weaver",     # Daemon/process health
            "champion",   # Model/checkpoint health
            "oracle",     # Inference server
            "guild",      # Skills/curriculum
            "scribe",     # Evaluation system
            "vault",      # Vault unification & remote ops
            "deep",       # Comprehensive (runs all)
        ]

        for module_name in ritual_modules:
            try:
                __import__(f"temple.rituals.{module_name}", fromlist=[module_name])
            except ImportError as e:
                logger.warning(f"Failed to load {module_name} ritual: {e}")
