"""Run configuration loading from YAML files."""

from pathlib import Path
from typing import Optional

from guild.config.loader import get_config_dir, load_yaml
from guild.runs.types import RunConfig, RunType


def load_run_config(run_id: str, config_dir: Optional[Path] = None) -> RunConfig:
    """
    Load a run configuration from YAML.

    Args:
        run_id: Run identifier (e.g., "daily_training")
        config_dir: Optional config directory (defaults to GUILD_CONFIG_DIR)

    Returns:
        RunConfig instance

    Raises:
        FileNotFoundError: If run config file doesn't exist
        ValueError: If config is invalid
    """
    if config_dir is None:
        config_dir = get_config_dir()

    run_path = config_dir / "runs" / f"{run_id}.yaml"

    if not run_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_path}")

    data = load_yaml(run_path)
    return _dict_to_run_config(data)


def _dict_to_run_config(data: dict) -> RunConfig:
    """Convert a dict to RunConfig, handling enum conversion."""
    # Required fields
    run_id = data.get("id")
    if not run_id:
        raise ValueError("Run config missing 'id' field")

    type_str = data.get("type")
    if not type_str:
        raise ValueError("Run config missing 'type' field")

    try:
        run_type = RunType(type_str)
    except ValueError:
        valid_types = [t.value for t in RunType]
        raise ValueError(f"Invalid run type: {type_str}. Valid: {valid_types}")

    # Optional fields with defaults
    return RunConfig(
        id=run_id,
        type=run_type,
        name=data.get("name", ""),
        description=data.get("description", ""),
        facility_id=data.get("facility_id", ""),
        hero_id=data.get("hero_id", ""),
        quest_filters=data.get("quest_filters", {}),
        max_steps=data.get("max_steps"),
        max_quests=data.get("max_quests"),
        max_duration_seconds=data.get("max_duration_seconds"),
        hyperparams=data.get("hyperparams", {}),
        log_level=data.get("log_level", "INFO"),
        log_facility_id=data.get("log_facility_id", ""),
        checkpoint_every_steps=data.get("checkpoint_every_steps", 1000),
        checkpoint_facility_id=data.get("checkpoint_facility_id", ""),
        tags=data.get("tags", []),
    )


def discover_runs(config_dir: Optional[Path] = None) -> list[str]:
    """
    Discover all run IDs from config files.

    Returns:
        List of run IDs (file stems from configs/runs/*.yaml)
    """
    if config_dir is None:
        config_dir = get_config_dir()

    runs_dir = config_dir / "runs"
    if not runs_dir.exists():
        return []

    run_ids = []
    for path in runs_dir.glob("*.yaml"):
        if not path.name.startswith("_"):
            run_ids.append(path.stem)

    return sorted(run_ids)


def load_all_runs(config_dir: Optional[Path] = None) -> dict[str, RunConfig]:
    """
    Load all run configurations.

    Returns:
        Dict mapping run_id to RunConfig
    """
    if config_dir is None:
        config_dir = get_config_dir()

    runs = {}
    for run_id in discover_runs(config_dir):
        try:
            runs[run_id] = load_run_config(run_id, config_dir)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load run '{run_id}': {e}")

    return runs


class RunLoader:
    """
    Cached run configuration loader.

    Provides caching to avoid re-reading YAML files.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, RunConfig] = {}
        self._discovered: Optional[list[str]] = None

    def load(self, run_id: str, use_cache: bool = True) -> RunConfig:
        """Load a run config with optional caching."""
        if use_cache and run_id in self._cache:
            return self._cache[run_id]

        config = load_run_config(run_id, self.config_dir)

        if use_cache:
            self._cache[run_id] = config

        return config

    def load_all(self, use_cache: bool = True) -> dict[str, RunConfig]:
        """Load all run configs with optional caching."""
        if use_cache and self._cache:
            return self._cache.copy()

        runs = load_all_runs(self.config_dir)

        if use_cache:
            self._cache = runs.copy()

        return runs

    def discover(self, use_cache: bool = True) -> list[str]:
        """Discover run IDs with optional caching."""
        if use_cache and self._discovered is not None:
            return self._discovered.copy()

        discovered = discover_runs(self.config_dir)

        if use_cache:
            self._discovered = discovered.copy()

        return discovered

    def exists(self, run_id: str) -> bool:
        """Check if a run config exists."""
        return run_id in self.discover()

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._discovered = None

    def invalidate(self, run_id: str):
        """Invalidate cache for a specific run."""
        self._cache.pop(run_id, None)
