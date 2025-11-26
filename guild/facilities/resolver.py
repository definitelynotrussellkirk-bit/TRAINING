"""Path resolution using facility configurations."""

import os
from pathlib import Path
from typing import Optional

from guild.facilities.types import Facility, FacilityType
from guild.config.loader import load_yaml


class PathResolver:
    """
    Resolves logical paths to physical paths using facility configs.

    Path formats:
    - "facility:arena_4090:checkpoints" -> resolved path
    - "facility:arena_4090:checkpoints/step-1000" -> with subpath
    - "@checkpoints" -> current facility's checkpoints
    - "@checkpoints/step-1000" -> with subpath
    - "~/path" -> home expansion
    - "/absolute" -> unchanged
    - "relative" -> relative to cwd
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self._facilities: dict[str, Facility] = {}
        self._current_facility: Optional[str] = None
        self._default_facility: Optional[str] = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str | Path):
        """Load facility configuration from YAML."""
        data = load_yaml(config_path)

        self._default_facility = data.get("default_facility")

        for fac_id, fac_data in data.get("facilities", {}).items():
            facility = Facility.from_dict({"id": fac_id, **fac_data})
            self._facilities[fac_id] = facility

    def add_facility(self, facility: Facility):
        """Add a facility directly."""
        self._facilities[facility.id] = facility

    def resolve(self, path_spec: str) -> Path:
        """Resolve a path specification to a physical path."""
        if path_spec.startswith("facility:"):
            parts = path_spec.split(":", 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid facility path: {path_spec}")
            _, facility_id, subpath = parts
            return self._resolve_facility_path(facility_id, subpath)

        elif path_spec.startswith("@"):
            key = path_spec[1:]
            facility_id = self._current_facility or self._default_facility
            if not facility_id:
                raise ValueError("No current or default facility set")
            return self._resolve_facility_path(facility_id, key)

        elif path_spec.startswith("~"):
            return Path(path_spec).expanduser()

        else:
            expanded = os.path.expandvars(path_spec)
            return Path(expanded)

    def _resolve_facility_path(self, facility_id: str, path_key: str) -> Path:
        """Resolve a path within a facility."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")

        facility = self._facilities[facility_id]
        base = Path(os.path.expandvars(facility.base_path)).expanduser()

        # Split path_key into alias and subpath
        if "/" in path_key:
            parts = path_key.split("/", 1)
            alias, subpath = parts[0], parts[1]
        else:
            alias, subpath = path_key, ""

        # Resolve alias
        if alias in facility.paths:
            resolved = base / facility.paths[alias]
        else:
            resolved = base / alias

        # Add subpath
        if subpath:
            resolved = resolved / subpath

        return resolved

    def set_current_facility(self, facility_id: str):
        """Set the current facility for @ shortcuts."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")
        self._current_facility = facility_id

    def get_facility(self, facility_id: str) -> Facility:
        """Get a facility by ID."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")
        return self._facilities[facility_id]

    def list_facilities(self, type_filter: Optional[FacilityType] = None) -> list[str]:
        """List facility IDs, optionally filtered by type."""
        if type_filter:
            return [fid for fid, f in self._facilities.items()
                    if f.type == type_filter]
        return list(self._facilities.keys())

    @property
    def current_facility_id(self) -> Optional[str]:
        return self._current_facility or self._default_facility


# Global resolver
_resolver: Optional[PathResolver] = None


def init_resolver(config_path: Optional[str | Path] = None) -> PathResolver:
    """Initialize the global path resolver."""
    global _resolver

    if config_path is None:
        config_path = os.environ.get("GUILD_FACILITIES_CONFIG")

        if not config_path:
            from guild.config.loader import get_config_dir
            local_path = get_config_dir() / "facilities" / "local.yaml"
            example_path = get_config_dir() / "facilities" / "example.yaml"

            if local_path.exists():
                config_path = local_path
            elif example_path.exists():
                config_path = example_path
            else:
                raise FileNotFoundError(
                    "No facility config found. Create configs/facilities/local.yaml"
                )

    _resolver = PathResolver(config_path)
    return _resolver


def get_resolver() -> PathResolver:
    """Get the global resolver, initializing if needed."""
    global _resolver
    if _resolver is None:
        init_resolver()
    return _resolver


def resolve(path_spec: str) -> Path:
    """Resolve a path using the global resolver."""
    return get_resolver().resolve(path_spec)


def get_facility(facility_id: str) -> Facility:
    """Get a facility by ID."""
    return get_resolver().get_facility(facility_id)


def set_current_facility(facility_id: str):
    """Set the current facility."""
    get_resolver().set_current_facility(facility_id)


def reset_resolver():
    """Reset the global resolver (useful for testing)."""
    global _resolver
    _resolver = None
