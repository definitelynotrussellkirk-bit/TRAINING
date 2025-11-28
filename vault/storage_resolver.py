"""
Storage Resolver - Resolve StorageHandles to filesystem paths.

The resolver is the bridge between logical handles and physical paths.
Given a StorageHandle, it returns the actual filesystem path based on
which device the code is running on.

Usage:
    from vault.storage_resolver import ask_storage, get_resolver
    from core.storage_types import StorageKind, StorageZone, StorageHandle

    # Simple: get default path for a checkpoint
    path = ask_storage(StorageKind.CHECKPOINT, "checkpoint-182000")

    # Explicit: specify zone
    resolver = get_resolver()
    handle = StorageHandle(
        kind=StorageKind.CHECKPOINT,
        key="checkpoint-182000",
        zone=StorageZone.WARM  # Get from NAS instead of local
    )
    nas_path = resolver.resolve(handle)

    # Search: find wherever it exists
    found_path = resolver.locate(StorageKind.CHECKPOINT, "checkpoint-182000")

Configuration:
    Storage zones are defined in config/storage_zones.json
    The current device is determined by TRAINING_DEVICE_ID env var

Design:
    - Each device has a "root" path for each zone it hosts
    - Each StorageKind has a subdirectory pattern with {key} placeholder
    - resolve() combines root + pattern to get final path
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.storage_types import StorageZone, StorageKind, StorageHandle

logger = logging.getLogger("vault.storage_resolver")


class StorageResolverError(Exception):
    """Base exception for storage resolver errors."""
    pass


class ZoneNotAvailable(StorageResolverError):
    """Raised when a zone is not available on the current device."""
    pass


class KindNotConfigured(StorageResolverError):
    """Raised when a storage kind is not configured."""
    pass


class StorageResolver:
    """
    Resolves StorageHandles to filesystem paths.

    The resolver loads zone configuration and provides path resolution
    based on the current device context.

    Usage:
        resolver = StorageResolver()

        # Resolve a handle to path
        handle = StorageHandle(StorageKind.CHECKPOINT, "checkpoint-182000", StorageZone.HOT)
        path = resolver.resolve(handle)

        # Create handle with default zone
        handle = resolver.default_handle(StorageKind.CHECKPOINT, "checkpoint-182000")

        # Find where an asset exists
        path = resolver.locate(StorageKind.CHECKPOINT, "checkpoint-182000")
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        device_id: Optional[str] = None,
    ):
        """
        Initialize the storage resolver.

        Args:
            config_path: Path to storage_zones.json (default: auto-detect)
            device_id: Current device ID (default: from TRAINING_DEVICE_ID env)
        """
        # Auto-detect config path
        if config_path:
            self.config_path = Path(config_path)
        else:
            base_dir = Path(__file__).parent.parent
            self.config_path = base_dir / "config" / "storage_zones.json"

        # Get device ID
        self.device_id = device_id or os.environ.get("TRAINING_DEVICE_ID")
        if not self.device_id:
            # Try to infer from base_dir for backward compatibility
            base_dir = Path(__file__).parent.parent
            if str(base_dir) == "/path/to/training":
                self.device_id = "trainer4090"
                logger.debug("Inferred device_id=trainer4090 from base_dir")

        self._config: Dict[str, Any] = {}
        self._zones: Dict[str, Dict] = {}
        self._kind_patterns: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        self._load()

    def _load(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(
                f"Storage zones config not found: {self.config_path}. "
                "Using fallback paths."
            )
            self._setup_fallback()
            return

        try:
            with open(self.config_path) as f:
                self._config = json.load(f)

            self._zones = self._config.get("zones", {})
            self._kind_patterns = self._config.get("kind_patterns", {})

            logger.info(
                f"Loaded storage zones: {list(self._zones.keys())} "
                f"for device: {self.device_id}"
            )

        except json.JSONDecodeError as e:
            raise StorageResolverError(f"Invalid JSON in {self.config_path}: {e}")

    def _setup_fallback(self) -> None:
        """Setup fallback configuration when config file is missing."""
        # Fallback to simple paths based on base_dir
        base_dir = Path(__file__).parent.parent

        self._zones = {
            "hot": {
                "devices": ["trainer4090"],
                "roots": {"trainer4090": str(base_dir)},
            },
        }

        self._kind_patterns = {
            "checkpoint": {"subdir": "models/current_model/{key}"},
            "current_model": {"subdir": "models/current_model"},
            "dataset": {"subdir": "data/{key}"},
            "queue": {"subdir": "queue/{key}"},
            "log": {"subdir": "logs/{key}"},
            "status": {"subdir": "status/{key}"},
        }

    def reload(self) -> None:
        """Reload configuration from disk."""
        with self._lock:
            self._load()

    # =========================================================================
    # RESOLUTION
    # =========================================================================

    def resolve(self, handle: StorageHandle) -> Path:
        """
        Resolve a StorageHandle to a filesystem path.

        Args:
            handle: The storage handle to resolve

        Returns:
            Path object for the resolved location

        Raises:
            ZoneNotAvailable: If zone not available on current device
            KindNotConfigured: If kind pattern not found
        """
        zone_str = handle.zone.value
        kind_str = handle.kind.value

        # Get zone config
        zone_cfg = self._zones.get(zone_str)
        if not zone_cfg:
            raise ZoneNotAvailable(f"Zone '{zone_str}' not configured")

        # Get root for this device
        roots = zone_cfg.get("roots", {})
        root = roots.get(self.device_id)

        if not root:
            # Check if any device has this zone
            available_devices = list(roots.keys())
            raise ZoneNotAvailable(
                f"Device '{self.device_id}' does not have zone '{zone_str}'. "
                f"Available on: {available_devices}"
            )

        # Get kind pattern
        kind_cfg = self._kind_patterns.get(kind_str)
        if not kind_cfg:
            raise KindNotConfigured(f"Storage kind '{kind_str}' not configured")

        # Build path
        subdir_pattern = kind_cfg.get("subdir", "{key}")
        subdir = subdir_pattern.format(key=handle.key)

        return Path(root) / subdir

    def resolve_or_none(self, handle: StorageHandle) -> Optional[Path]:
        """
        Resolve handle, returning None if not available.

        Unlike resolve(), this doesn't raise on unavailable zones.
        """
        try:
            return self.resolve(handle)
        except StorageResolverError:
            return None

    def default_handle(self, kind: StorageKind, key: str) -> StorageHandle:
        """
        Create a handle with the default zone for the kind.

        Args:
            kind: Storage kind
            key: Unique key for the item

        Returns:
            StorageHandle with default zone
        """
        return StorageHandle.for_kind(kind, key)

    def resolve_default(self, kind: StorageKind, key: str) -> Path:
        """
        Resolve using default zone for the kind.

        Shorthand for: resolve(default_handle(kind, key))
        """
        handle = self.default_handle(kind, key)
        return self.resolve(handle)

    # =========================================================================
    # SEARCH / LOCATE
    # =========================================================================

    def locate(
        self,
        kind: StorageKind,
        key: str,
        check_exists: bool = True,
    ) -> Optional[Path]:
        """
        Find where an asset exists across all zones.

        Searches zones in order: hot -> warm -> cold

        Args:
            kind: Storage kind
            key: Unique key
            check_exists: If True, verify path exists on disk

        Returns:
            Path if found, None otherwise
        """
        search_order = [StorageZone.HOT, StorageZone.WARM, StorageZone.COLD]

        for zone in search_order:
            handle = StorageHandle(kind=kind, key=key, zone=zone)
            try:
                path = self.resolve(handle)
                if not check_exists or path.exists():
                    return path
            except StorageResolverError:
                continue

        return None

    def locate_all(
        self,
        kind: StorageKind,
        key: str,
    ) -> List[Path]:
        """
        Find all locations where an asset exists.

        Returns:
            List of paths where the asset exists
        """
        results = []
        for zone in StorageZone:
            handle = StorageHandle(kind=kind, key=key, zone=zone)
            try:
                path = self.resolve(handle)
                if path.exists():
                    results.append(path)
            except StorageResolverError:
                continue
        return results

    def exists(self, handle: StorageHandle) -> bool:
        """Check if a handle's path exists on disk."""
        try:
            path = self.resolve(handle)
            return path.exists()
        except StorageResolverError:
            return False

    # =========================================================================
    # ZONE INFO
    # =========================================================================

    def available_zones(self) -> List[StorageZone]:
        """Get zones available on the current device."""
        available = []
        for zone_str, zone_cfg in self._zones.items():
            roots = zone_cfg.get("roots", {})
            if self.device_id in roots:
                try:
                    available.append(StorageZone(zone_str))
                except ValueError:
                    pass
        return available

    def zone_root(self, zone: StorageZone) -> Optional[Path]:
        """Get the root path for a zone on this device."""
        zone_cfg = self._zones.get(zone.value, {})
        roots = zone_cfg.get("roots", {})
        root = roots.get(self.device_id)
        return Path(root) if root else None

    def get_info(self) -> Dict[str, Any]:
        """Get resolver info for debugging."""
        return {
            "device_id": self.device_id,
            "config_path": str(self.config_path),
            "available_zones": [z.value for z in self.available_zones()],
            "configured_kinds": list(self._kind_patterns.keys()),
        }


# =============================================================================
# SINGLETON + HELPERS
# =============================================================================

_resolver: Optional[StorageResolver] = None
_resolver_lock = threading.Lock()


def get_resolver() -> StorageResolver:
    """Get or create the storage resolver singleton."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                config_path = os.environ.get("STORAGE_ZONES_PATH")
                _resolver = StorageResolver(
                    config_path=Path(config_path) if config_path else None,
                )
    return _resolver


def ask_storage(kind: StorageKind, key: str) -> Path:
    """
    One-liner to get path for a storage item.

    Uses default zone for the kind.

    Args:
        kind: What type of thing (CHECKPOINT, DATASET, etc.)
        key: Unique identifier

    Returns:
        Filesystem path

    Example:
        path = ask_storage(StorageKind.CHECKPOINT, "checkpoint-182000")
        # Returns: /path/to/training/models/current_model/checkpoint-182000
    """
    return get_resolver().resolve_default(kind, key)


def ask_storage_handle(handle: StorageHandle) -> Path:
    """
    Resolve a StorageHandle to a path.

    Args:
        handle: The handle to resolve

    Returns:
        Filesystem path
    """
    return get_resolver().resolve(handle)


def locate_storage(kind: StorageKind, key: str) -> Optional[Path]:
    """
    Find where a storage item exists.

    Searches across all zones.

    Args:
        kind: What type of thing
        key: Unique identifier

    Returns:
        Path if found, None otherwise
    """
    return get_resolver().locate(kind, key)


def storage_exists(kind: StorageKind, key: str) -> bool:
    """Check if a storage item exists anywhere."""
    return get_resolver().locate(kind, key) is not None


# =============================================================================
# COMMON PATHS
# =============================================================================

def get_checkpoint_path(key: str) -> Path:
    """Get path for a checkpoint."""
    return ask_storage(StorageKind.CHECKPOINT, key)


def get_snapshot_path(key: str) -> Path:
    """Get path for a snapshot."""
    return ask_storage(StorageKind.SNAPSHOT, key)


def get_dataset_path(key: str) -> Path:
    """Get path for a dataset."""
    return ask_storage(StorageKind.DATASET, key)


def get_queue_path(priority: str = "normal") -> Path:
    """Get path for a queue directory."""
    return ask_storage(StorageKind.QUEUE, priority)


def get_current_model_path() -> Path:
    """Get path for the current model directory."""
    return ask_storage(StorageKind.CURRENT_MODEL, "")


def get_status_path(name: str) -> Path:
    """Get path for a status file."""
    return ask_storage(StorageKind.STATUS, name)


def get_log_path(name: str) -> Path:
    """Get path for a log file."""
    return ask_storage(StorageKind.LOG, name)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    resolver = get_resolver()
    info = resolver.get_info()

    print("\nStorage Resolver Info")
    print("=" * 50)
    print(f"Device ID: {info['device_id']}")
    print(f"Config: {info['config_path']}")
    print(f"Available zones: {', '.join(info['available_zones'])}")
    print(f"Configured kinds: {', '.join(info['configured_kinds'])}")

    print("\nExample Resolutions:")

    # Show some example resolutions
    examples = [
        (StorageKind.CHECKPOINT, "checkpoint-182000"),
        (StorageKind.CURRENT_MODEL, ""),
        (StorageKind.QUEUE, "high"),
        (StorageKind.DATASET, "binary_l5_v2"),
        (StorageKind.STATUS, "training_status.json"),
    ]

    for kind, key in examples:
        try:
            path = ask_storage(kind, key)
            exists = "exists" if path.exists() else "not found"
            print(f"  {kind.value}/{key or '.'}: {path} [{exists}]")
        except StorageResolverError as e:
            print(f"  {kind.value}/{key}: ERROR - {e}")

    # Show zone roots
    print("\nZone Roots:")
    for zone in StorageZone:
        root = resolver.zone_root(zone)
        if root:
            print(f"  {zone.value}: {root}")
        else:
            print(f"  {zone.value}: not available on this device")

    # Test locate
    print("\nLocate Test:")
    test_key = "checkpoint-182000"
    found = locate_storage(StorageKind.CHECKPOINT, test_key)
    if found:
        print(f"  Found {test_key} at: {found}")
    else:
        print(f"  {test_key} not found in any zone")
