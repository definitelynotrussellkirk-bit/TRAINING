"""
Temple Protocols - Formal interfaces for rituals and temple systems.

This module defines the Protocol interfaces that all rituals must implement,
enabling type-safe ritual registration and execution.

Design Philosophy:
    - Protocols define contracts, not inheritance
    - RitualContext carries all dependencies (no global state)
    - RitualResult is immutable and serializable
    - Rituals are pure functions: (context) -> result
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from .schemas import RitualCheckResult, RitualResult, ResultStatus


class RitualKind(str, Enum):
    """
    Classification of ritual purpose.

    Used for:
    - UI grouping (healthchecks in one section, training in another)
    - Dependency validation (training rituals shouldn't depend on each other)
    - Filtering (run only healthchecks, skip training rituals)
    """
    HEALTHCHECK = "healthcheck"  # Quick diagnostics (Oracle, Quick)
    TRAINING = "training"        # Training-related (Forge, Champion)
    EVAL = "eval"                # Evaluation system (Scribe)
    INFRA = "infra"              # Infrastructure (Weaver, Guild)
    META = "meta"                # Meta-rituals that run others (Deep)


@dataclass(frozen=True)
class RitualMeta:
    """
    Immutable metadata about a ritual.

    This is registered alongside the ritual function and used for:
    - UI display (name, description, icon)
    - Dependency resolution (requires)
    - Categorization (kind)

    Attributes:
        id: Unique identifier (e.g., "quick", "scribe")
        name: Display name (e.g., "Ritual of Quick")
        description: One-line description for UI tooltips
        kind: Classification for grouping/filtering
        icon: Emoji or icon identifier for UI
        requires: List of ritual IDs that must pass before this one
        tags: Additional categorization tags
    """
    id: str
    name: str
    description: str
    kind: RitualKind = RitualKind.HEALTHCHECK
    icon: str = "🔮"
    requires: tuple = ()  # Tuple for immutability
    tags: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "icon": self.icon,
            "requires": list(self.requires),
            "tags": list(self.tags),
        }


@dataclass
class RitualContext:
    """
    Execution context passed to every ritual.

    Contains all dependencies a ritual might need, avoiding global state.
    Rituals should only read from context, never modify external state
    except through explicit side-effect declarations.

    Attributes:
        run_id: Unique identifier for this ritual execution
        base_dir: Project base directory
        dry_run: If True, don't make changes (just validate)
        logger: Optional logger for ritual output
        config: Optional config dict (from config.json)
        campaign_id: Active campaign ID if any
        hero_id: Active hero ID if any
    """
    run_id: str
    base_dir: Path
    dry_run: bool = False
    logger: Optional[Any] = None  # logging.Logger
    config: Dict[str, Any] = field(default_factory=dict)
    campaign_id: Optional[str] = None
    hero_id: Optional[str] = None

    # Injected dependencies (lazy-loaded by cleric)
    _ledger: Optional[Any] = field(default=None, repr=False)
    _eval_ledger: Optional[Any] = field(default=None, repr=False)

    @property
    def ledger(self):
        """Get checkpoint ledger (lazy-loaded)."""
        if self._ledger is None:
            from core.checkpoint_ledger import get_ledger
            self._ledger = get_ledger()
        return self._ledger

    @property
    def eval_ledger(self):
        """Get evaluation ledger (lazy-loaded)."""
        if self._eval_ledger is None:
            from core.evaluation_ledger import get_eval_ledger
            self._eval_ledger = get_eval_ledger(self.base_dir)
        return self._eval_ledger


@runtime_checkable
class Ritual(Protocol):
    """
    Protocol that all rituals must implement.

    A ritual is a diagnostic or operational procedure that:
    1. Receives a RitualContext with dependencies
    2. Performs checks or operations
    3. Returns a list of RitualCheckResult

    Rituals should be:
    - Idempotent (running twice gives same result)
    - Side-effect free for dry_run=True
    - Fast for healthchecks (< 5 seconds)

    Example implementation:
        @register_ritual(RitualMeta(
            id="quick",
            name="Ritual of Quick",
            description="Fast sanity checks",
            kind=RitualKind.HEALTHCHECK,
            icon="⚡"
        ))
        def run(ctx: RitualContext) -> List[RitualCheckResult]:
            results = []
            results.append(check_something(ctx))
            return results
    """

    def __call__(self, ctx: RitualContext) -> List[RitualCheckResult]:
        """Execute the ritual and return check results."""
        ...


# Type alias for ritual functions
RitualFunc = Callable[[RitualContext], List[RitualCheckResult]]


@dataclass
class RitualRegistry:
    """
    Central registry for all rituals.

    Provides:
    - Registration via decorator
    - Lookup by ID
    - Filtering by kind/tags
    - Dependency graph resolution
    """
    _rituals: Dict[str, RitualFunc] = field(default_factory=dict)
    _meta: Dict[str, RitualMeta] = field(default_factory=dict)

    def register(self, meta: RitualMeta):
        """
        Decorator to register a ritual function.

        Usage:
            @registry.register(RitualMeta(
                id="quick",
                name="Ritual of Quick",
                ...
            ))
            def run(ctx: RitualContext) -> List[RitualCheckResult]:
                ...
        """
        def decorator(fn: RitualFunc) -> RitualFunc:
            self._rituals[meta.id] = fn
            self._meta[meta.id] = meta
            return fn
        return decorator

    def get(self, ritual_id: str) -> Optional[RitualFunc]:
        """Get ritual function by ID."""
        return self._rituals.get(ritual_id)

    def get_meta(self, ritual_id: str) -> Optional[RitualMeta]:
        """Get ritual metadata by ID."""
        return self._meta.get(ritual_id)

    def list_all(self) -> List[RitualMeta]:
        """List all registered ritual metadata."""
        return list(self._meta.values())

    def list_by_kind(self, kind: RitualKind) -> List[RitualMeta]:
        """List rituals of a specific kind."""
        return [m for m in self._meta.values() if m.kind == kind]

    def get_dependency_order(self, ritual_ids: Optional[List[str]] = None) -> List[str]:
        """
        Get rituals in dependency order (topological sort).

        Args:
            ritual_ids: Specific rituals to order (None = all)

        Returns:
            List of ritual IDs in execution order

        Raises:
            ValueError: If circular dependency detected
        """
        if ritual_ids is None:
            ritual_ids = list(self._rituals.keys())

        # Build dependency graph
        visited = set()
        result = []
        temp_mark = set()

        def visit(rid: str):
            if rid in temp_mark:
                raise ValueError(f"Circular dependency detected involving {rid}")
            if rid in visited:
                return

            temp_mark.add(rid)
            meta = self._meta.get(rid)
            if meta:
                for dep in meta.requires:
                    if dep in self._rituals:
                        visit(dep)
            temp_mark.remove(rid)
            visited.add(rid)
            result.append(rid)

        for rid in ritual_ids:
            if rid not in visited:
                visit(rid)

        return result

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize registry to JSON-safe dict."""
        return {
            rid: meta.to_dict()
            for rid, meta in self._meta.items()
        }


# Global registry instance
RITUAL_REGISTRY = RitualRegistry()


def register_ritual(meta: RitualMeta):
    """
    Convenience decorator using global registry.

    Usage:
        @register_ritual(RitualMeta(
            id="quick",
            name="Ritual of Quick",
            description="Fast sanity checks",
            kind=RitualKind.HEALTHCHECK
        ))
        def run(ctx: RitualContext) -> List[RitualCheckResult]:
            ...
    """
    return RITUAL_REGISTRY.register(meta)
