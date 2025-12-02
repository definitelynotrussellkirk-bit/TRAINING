"""
Device Mapping - Bridge between Ledger device_ids and VaultKeeper strongholds.

This module provides the canonical mapping between:
- device_id: Used by CheckpointLedger (trainer4090, inference3090, synology_data)
- stronghold: Used by VaultKeeper (local_vault, nas_archive, inference_cache)

The mapping is configured in config/device_mapping.json.

Usage:
    from vault.device_mapping import get_mapping, device_to_stronghold

    # Get singleton mapping instance
    mapping = get_mapping()

    # Convert device_id to stronghold
    stronghold = mapping.device_to_stronghold("trainer4090")  # "local_vault"

    # Convert stronghold to device_id
    device_id = mapping.stronghold_to_device("local_vault")   # "trainer4090"

    # Get base path for a device
    path = mapping.get_base_path("trainer4090")  # Path resolved from config

    # Convenience function
    stronghold = device_to_stronghold("trainer4090")

RPG Flavor:
    The DeviceMapping is the Rosetta Stone of the realm - translating between
    the language of the Ledger (device_ids) and the language of the VaultKeeper
    (strongholds). Without it, the two systems cannot communicate.
"""

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vault.device_mapping")


@dataclass
class DeviceInfo:
    """Information about a device/stronghold mapping."""

    device_id: str
    stronghold: str
    zone: str
    is_primary: bool
    base_path: Path
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "stronghold": self.stronghold,
            "zone": self.zone,
            "is_primary": self.is_primary,
            "base_path": str(self.base_path),
            "description": self.description,
        }


class DeviceMapping:
    """
    Bridge between Ledger device_ids and VaultKeeper strongholds.

    Loads mapping from config/device_mapping.json and provides
    bidirectional lookup between the two naming conventions.

    Thread-safe singleton pattern.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the device mapping.

        Args:
            config_path: Path to device_mapping.json (default: auto-detect)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Auto-detect from vault module location
            base_dir = Path(__file__).parent.parent
            self.config_path = base_dir / "config" / "device_mapping.json"

        self._mappings: Dict[str, DeviceInfo] = {}  # device_id -> DeviceInfo
        self._reverse: Dict[str, str] = {}  # stronghold -> device_id
        self._stronghold_priority: List[str] = []
        self._zone_priority: List[str] = []
        self._lock = threading.Lock()
        self._local_device_id: Optional[str] = None

        self._load()

    def _load(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.warning(
                f"Device mapping config not found: {self.config_path}. "
                "Using fallback defaults."
            )
            self._setup_fallback()
            return

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            self._stronghold_priority = data.get("stronghold_priority", [])
            self._zone_priority = data.get("zone_priority", ["hot", "warm", "cold"])

            for device_id, info in data.get("mappings", {}).items():
                # Resolve path placeholders
                raw_path = info["base_path"]
                if raw_path == "$TRAINING_BASE_DIR":
                    # Use core.paths if available, otherwise env var
                    try:
                        from core.paths import get_base_dir
                        resolved_path = get_base_dir()
                    except ImportError:
                        resolved_path = Path(os.environ.get("TRAINING_BASE_DIR", "."))
                elif raw_path.startswith("~"):
                    resolved_path = Path(raw_path).expanduser()
                else:
                    resolved_path = Path(raw_path)

                device_info = DeviceInfo(
                    device_id=device_id,
                    stronghold=info["stronghold"],
                    zone=info.get("zone", "hot"),
                    is_primary=info.get("is_primary", False),
                    base_path=resolved_path,
                    description=info.get("description", ""),
                )
                self._mappings[device_id] = device_info
                self._reverse[info["stronghold"]] = device_id

            logger.info(
                f"Loaded device mapping: {len(self._mappings)} devices, "
                f"{len(self._stronghold_priority)} strongholds"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.config_path}: {e}")
            self._setup_fallback()
        except Exception as e:
            logger.error(f"Failed to load device mapping: {e}")
            self._setup_fallback()

    def _setup_fallback(self) -> None:
        """Setup fallback configuration when config file is missing."""
        # Resolve base path dynamically
        try:
            from core.paths import get_base_dir
            base_path = get_base_dir()
        except ImportError:
            base_path = Path(os.environ.get("TRAINING_BASE_DIR", "."))

        # Minimal fallback for trainer
        self._mappings["trainer4090"] = DeviceInfo(
            device_id="trainer4090",
            stronghold="local_vault",
            zone="hot",
            is_primary=True,
            base_path=base_path,
            description="Training server (fallback)",
        )
        self._reverse["local_vault"] = "trainer4090"
        self._stronghold_priority = ["local_vault"]
        self._zone_priority = ["hot", "warm", "cold"]

    def reload(self) -> None:
        """Reload configuration from disk."""
        with self._lock:
            self._mappings.clear()
            self._reverse.clear()
            self._load()

    # =========================================================================
    # LOOKUP OPERATIONS
    # =========================================================================

    def device_to_stronghold(self, device_id: str) -> str:
        """
        Convert device_id to stronghold name.

        Args:
            device_id: Device identifier (e.g., "trainer4090")

        Returns:
            Stronghold name (e.g., "local_vault")

        Raises:
            KeyError: If device_id not found in mapping
        """
        if device_id not in self._mappings:
            raise KeyError(f"Unknown device_id: {device_id}")
        return self._mappings[device_id].stronghold

    def stronghold_to_device(self, stronghold: str) -> str:
        """
        Convert stronghold name to device_id.

        Args:
            stronghold: Stronghold name (e.g., "local_vault")

        Returns:
            Device identifier (e.g., "trainer4090")

        Raises:
            KeyError: If stronghold not found in mapping
        """
        if stronghold not in self._reverse:
            raise KeyError(f"Unknown stronghold: {stronghold}")
        return self._reverse[stronghold]

    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get full device info for a device_id."""
        return self._mappings.get(device_id)

    def get_stronghold_info(self, stronghold: str) -> Optional[DeviceInfo]:
        """Get device info for a stronghold."""
        device_id = self._reverse.get(stronghold)
        if device_id:
            return self._mappings.get(device_id)
        return None

    def get_base_path(self, device_id: str) -> Path:
        """
        Get base path for a device.

        Args:
            device_id: Device identifier

        Returns:
            Base directory path for that device

        Raises:
            KeyError: If device_id not found
        """
        if device_id not in self._mappings:
            raise KeyError(f"Unknown device_id: {device_id}")
        return self._mappings[device_id].base_path

    def get_zone(self, device_id: str) -> str:
        """
        Get zone for a device.

        Args:
            device_id: Device identifier

        Returns:
            Zone name (hot, warm, cold)

        Raises:
            KeyError: If device_id not found
        """
        if device_id not in self._mappings:
            raise KeyError(f"Unknown device_id: {device_id}")
        return self._mappings[device_id].zone

    # =========================================================================
    # LOCAL DEVICE DETECTION
    # =========================================================================

    def get_local_device_id(self) -> str:
        """
        Detect and return the local device_id.

        Uses environment variable TRAINING_DEVICE_ID if set,
        otherwise infers from hostname.

        Returns:
            Local device_id (e.g., "trainer4090")
        """
        if self._local_device_id:
            return self._local_device_id

        # Check environment variable
        env_device = os.environ.get("TRAINING_DEVICE_ID")
        if env_device and env_device in self._mappings:
            self._local_device_id = env_device
            return env_device

        # Infer from hostname
        hostname = socket.gethostname().lower()

        if "4090" in hostname or "trainer" in hostname:
            self._local_device_id = "trainer4090"
        elif "3090" in hostname or "inference" in hostname:
            self._local_device_id = "inference3090"
        else:
            # Default to trainer
            self._local_device_id = "trainer4090"

        logger.debug(f"Detected local device_id: {self._local_device_id}")
        return self._local_device_id

    def get_local_stronghold(self) -> str:
        """Get the stronghold name for the local device."""
        return self.device_to_stronghold(self.get_local_device_id())

    def get_local_base_path(self) -> Path:
        """Get the base path for the local device."""
        return self.get_base_path(self.get_local_device_id())

    # =========================================================================
    # PRIORITY & SELECTION
    # =========================================================================

    def get_stronghold_priority(self) -> List[str]:
        """Get ordered list of strongholds by preference."""
        return self._stronghold_priority.copy()

    def get_zone_priority(self) -> List[str]:
        """Get ordered list of zones by preference."""
        return self._zone_priority.copy()

    def pick_best_device(self, device_ids: List[str]) -> Optional[str]:
        """
        Pick the best device from a list based on priority.

        Args:
            device_ids: List of device identifiers

        Returns:
            Best device_id from the list, or None if empty
        """
        if not device_ids:
            return None

        # Convert to strongholds and sort by priority
        stronghold_to_device = {}
        for device_id in device_ids:
            try:
                stronghold = self.device_to_stronghold(device_id)
                stronghold_to_device[stronghold] = device_id
            except KeyError:
                continue

        # Find first in priority list
        for stronghold in self._stronghold_priority:
            if stronghold in stronghold_to_device:
                return stronghold_to_device[stronghold]

        # Fallback to first in list
        return device_ids[0] if device_ids else None

    def pick_best_stronghold(self, strongholds: List[str]) -> Optional[str]:
        """
        Pick the best stronghold from a list based on priority.

        Args:
            strongholds: List of stronghold names

        Returns:
            Best stronghold from the list, or None if empty
        """
        if not strongholds:
            return None

        for stronghold in self._stronghold_priority:
            if stronghold in strongholds:
                return stronghold

        return strongholds[0]

    # =========================================================================
    # LISTING
    # =========================================================================

    def list_devices(self) -> List[str]:
        """List all known device_ids."""
        return list(self._mappings.keys())

    def list_strongholds(self) -> List[str]:
        """List all known strongholds."""
        return list(self._reverse.keys())

    def list_all(self) -> List[DeviceInfo]:
        """List all device infos."""
        return list(self._mappings.values())


# =============================================================================
# SINGLETON
# =============================================================================

_mapping_instance: Optional[DeviceMapping] = None
_mapping_lock = threading.Lock()


def get_mapping(config_path: Optional[Path] = None) -> DeviceMapping:
    """
    Get the singleton DeviceMapping instance.

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        DeviceMapping instance
    """
    global _mapping_instance

    if _mapping_instance is None:
        with _mapping_lock:
            if _mapping_instance is None:
                _mapping_instance = DeviceMapping(config_path)

    return _mapping_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def device_to_stronghold(device_id: str) -> str:
    """Convert device_id to stronghold name."""
    return get_mapping().device_to_stronghold(device_id)


def stronghold_to_device(stronghold: str) -> str:
    """Convert stronghold name to device_id."""
    return get_mapping().stronghold_to_device(stronghold)


def get_local_device_id() -> str:
    """Get the local device_id."""
    return get_mapping().get_local_device_id()


def get_local_stronghold() -> str:
    """Get the local stronghold name."""
    return get_mapping().get_local_stronghold()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Device Mapping Utility")
    parser.add_argument("--list", action="store_true", help="List all mappings")
    parser.add_argument("--device", type=str, help="Look up stronghold for device")
    parser.add_argument("--stronghold", type=str, help="Look up device for stronghold")
    parser.add_argument("--local", action="store_true", help="Show local device info")

    args = parser.parse_args()

    mapping = get_mapping()

    if args.list:
        print("Device Mappings:")
        print("-" * 60)
        for info in mapping.list_all():
            print(f"  {info.device_id:20} -> {info.stronghold:20} ({info.zone})")
        print()
        print(f"Stronghold Priority: {mapping.get_stronghold_priority()}")

    elif args.device:
        try:
            stronghold = mapping.device_to_stronghold(args.device)
            info = mapping.get_device_info(args.device)
            print(f"{args.device} -> {stronghold}")
            if info:
                print(f"  Zone: {info.zone}")
                print(f"  Base: {info.base_path}")
        except KeyError as e:
            print(f"Error: {e}")

    elif args.stronghold:
        try:
            device_id = mapping.stronghold_to_device(args.stronghold)
            info = mapping.get_stronghold_info(args.stronghold)
            print(f"{args.stronghold} -> {device_id}")
            if info:
                print(f"  Zone: {info.zone}")
                print(f"  Base: {info.base_path}")
        except KeyError as e:
            print(f"Error: {e}")

    elif args.local:
        device_id = mapping.get_local_device_id()
        stronghold = mapping.get_local_stronghold()
        base_path = mapping.get_local_base_path()
        print(f"Local Device: {device_id}")
        print(f"Stronghold:   {stronghold}")
        print(f"Base Path:    {base_path}")

    else:
        parser.print_help()
