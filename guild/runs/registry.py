"""Run registry - central access point for run definitions."""

from pathlib import Path
from typing import Optional, Iterator

from guild.runs.types import RunConfig, RunType
from guild.runs.loader import RunLoader


class RunRegistry:
    """
    Central registry for run configurations.

    Provides:
    - Lazy loading of run configs from YAML
    - Lookup by ID, type, or tag
    - Iteration over all runs
    - Singleton-style global access via module functions

    Example:
        registry = RunRegistry()
        run = registry.get("daily_training")
        eval_runs = registry.by_type(RunType.EVALUATION)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._loader = RunLoader(config_dir)
        self._loaded = False

    def _ensure_loaded(self):
        """Ensure all runs are loaded."""
        if not self._loaded:
            self._loader.load_all()
            self._loaded = True

    def get(self, run_id: str) -> RunConfig:
        """
        Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunConfig

        Raises:
            KeyError: If run not found
        """
        if not self._loader.exists(run_id):
            raise KeyError(f"Unknown run: {run_id}")
        return self._loader.load(run_id)

    def get_or_none(self, run_id: str) -> Optional[RunConfig]:
        """Get a run by ID, returning None if not found."""
        try:
            return self.get(run_id)
        except (KeyError, FileNotFoundError):
            return None

    def exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return self._loader.exists(run_id)

    def list_ids(self) -> list[str]:
        """List all run IDs."""
        return self._loader.discover()

    def all(self) -> dict[str, RunConfig]:
        """Get all runs as a dict."""
        self._ensure_loaded()
        return self._loader.load_all()

    def __iter__(self) -> Iterator[RunConfig]:
        """Iterate over all run configs."""
        self._ensure_loaded()
        for config in self._loader.load_all().values():
            yield config

    def __len__(self) -> int:
        """Number of registered runs."""
        return len(self._loader.discover())

    def __contains__(self, run_id: str) -> bool:
        """Check if run exists."""
        return self.exists(run_id)

    def by_type(self, run_type: RunType) -> list[RunConfig]:
        """Get all runs of a specific type."""
        self._ensure_loaded()
        return [
            run for run in self._loader.load_all().values()
            if run.type == run_type
        ]

    def by_tag(self, tag: str) -> list[RunConfig]:
        """Get all runs with a specific tag."""
        self._ensure_loaded()
        return [
            run for run in self._loader.load_all().values()
            if tag in run.tags
        ]

    def by_facility(self, facility_id: str) -> list[RunConfig]:
        """Get all runs targeting a specific facility."""
        self._ensure_loaded()
        return [
            run for run in self._loader.load_all().values()
            if run.facility_id == facility_id
        ]

    def search(
        self,
        run_type: Optional[RunType] = None,
        tags: Optional[list[str]] = None,
        facility_id: Optional[str] = None,
        name_contains: Optional[str] = None,
    ) -> list[RunConfig]:
        """
        Search runs by multiple criteria.

        Args:
            run_type: Filter by type
            tags: Filter by tags (run must have ALL tags)
            facility_id: Filter by facility
            name_contains: Filter by name substring (case-insensitive)

        Returns:
            List of matching runs
        """
        self._ensure_loaded()
        results = list(self._loader.load_all().values())

        if run_type is not None:
            results = [r for r in results if r.type == run_type]

        if tags:
            results = [r for r in results if all(t in r.tags for t in tags)]

        if facility_id:
            results = [r for r in results if r.facility_id == facility_id]

        if name_contains:
            needle = name_contains.lower()
            results = [r for r in results if needle in r.name.lower()]

        return results

    def refresh(self):
        """Refresh the registry by clearing caches."""
        self._loader.clear_cache()
        self._loaded = False

    def invalidate(self, run_id: str):
        """Invalidate a specific run's cache."""
        self._loader.invalidate(run_id)


# Global registry instance
_registry: Optional[RunRegistry] = None


def init_registry(config_dir: Optional[Path] = None) -> RunRegistry:
    """Initialize the global run registry."""
    global _registry
    _registry = RunRegistry(config_dir)
    return _registry


def get_registry() -> RunRegistry:
    """Get the global run registry, initializing if needed."""
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def reset_registry():
    """Reset the global run registry (useful for testing)."""
    global _registry
    _registry = None


# Convenience functions using global registry

def get_run(run_id: str) -> RunConfig:
    """Get a run by ID from the global registry."""
    return get_registry().get(run_id)


def list_runs() -> list[str]:
    """List all run IDs from the global registry."""
    return get_registry().list_ids()


def runs_by_type(run_type: RunType) -> list[RunConfig]:
    """Get runs by type from the global registry."""
    return get_registry().by_type(run_type)


def runs_by_tag(tag: str) -> list[RunConfig]:
    """Get runs by tag from the global registry."""
    return get_registry().by_tag(tag)
