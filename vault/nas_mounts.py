"""NAS mount detection and health checking.

Provides utilities for checking Synology NAS mount status
and integrating with the storage zone system.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MountInfo:
    """Information about a mount point."""

    path: str
    mounted: bool
    device_id: str
    zone: str
    total_gb: Optional[float] = None
    used_gb: Optional[float] = None
    free_gb: Optional[float] = None
    percent_used: Optional[float] = None

    @property
    def available(self) -> bool:
        """True if mount is active and has free space."""
        return self.mounted and (self.free_gb or 0) > 1.0


# Default mount configuration
MOUNT_CONFIG = {
    "synology_data": {
        "local_path": "/mnt/synology/data",
        "data_path": "/mnt/synology/data/llm_training",
        "remote_path": "/volume1/data",
        "zone": "warm",
    },
    "synology_backup": {
        "local_path": "/mnt/synology/backup",
        "data_path": "/mnt/synology/backup/llm_training",
        "remote_path": "/volume1/backup",
        "zone": "warm",
    },
    "synology_archive": {
        "local_path": "/mnt/synology/archive",
        "data_path": "/mnt/synology/archive/llm_training",
        "remote_path": "/volume1/archive",
        "zone": "cold",
    },
}


def is_mountpoint(path: str) -> bool:
    """Check if a path is an active mount point."""
    try:
        result = subprocess.run(
            ["mountpoint", "-q", path], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_mount_stats(path: str) -> tuple[float, float, float, float]:
    """Get disk usage stats for a mounted path.

    Returns: (total_gb, used_gb, free_gb, percent_used)
    """
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        percent_used = (usage.used / usage.total) * 100 if usage.total > 0 else 0
        return total_gb, used_gb, free_gb, percent_used
    except (OSError, PermissionError):
        return 0.0, 0.0, 0.0, 0.0


def check_mount(device_id: str) -> MountInfo:
    """Check status of a specific NAS mount."""
    config = MOUNT_CONFIG.get(device_id)
    if not config:
        return MountInfo(
            path="", mounted=False, device_id=device_id, zone="unknown"
        )

    path = config["local_path"]
    zone = config["zone"]

    mounted = is_mountpoint(path)

    if mounted:
        total, used, free, pct = get_mount_stats(path)
        return MountInfo(
            path=path,
            mounted=True,
            device_id=device_id,
            zone=zone,
            total_gb=total,
            used_gb=used,
            free_gb=free,
            percent_used=pct,
        )
    else:
        return MountInfo(
            path=path, mounted=False, device_id=device_id, zone=zone
        )


def check_all_mounts() -> dict[str, MountInfo]:
    """Check status of all configured NAS mounts."""
    return {device_id: check_mount(device_id) for device_id in MOUNT_CONFIG}


def get_available_nas_devices() -> list[str]:
    """Get list of device_ids for mounted NAS volumes."""
    return [
        device_id
        for device_id, info in check_all_mounts().items()
        if info.available
    ]


def resolve_nas_path(device_id: str, subpath: str = "") -> Optional[str]:
    """Resolve a path on a NAS device.

    Args:
        device_id: The device identifier (e.g., 'synology_data')
        subpath: Optional path within the mount

    Returns:
        Full local path if mounted, None otherwise
    """
    info = check_mount(device_id)
    if not info.mounted:
        return None

    if subpath:
        return os.path.join(info.path, subpath)
    return info.path


def get_health_status() -> dict:
    """Get overall NAS health status for monitoring."""
    mounts = check_all_mounts()

    status = {
        "nas_available": any(m.available for m in mounts.values()),
        "warm_zone_available": any(
            m.available for m in mounts.values() if m.zone == "warm"
        ),
        "cold_zone_available": any(
            m.available for m in mounts.values() if m.zone == "cold"
        ),
        "mounts": {},
    }

    for device_id, info in mounts.items():
        status["mounts"][device_id] = {
            "mounted": info.mounted,
            "available": info.available,
            "zone": info.zone,
            "path": info.path,
        }
        if info.mounted:
            status["mounts"][device_id].update(
                {
                    "total_gb": round(info.total_gb or 0, 1),
                    "used_gb": round(info.used_gb or 0, 1),
                    "free_gb": round(info.free_gb or 0, 1),
                    "percent_used": round(info.percent_used or 0, 1),
                }
            )

    return status


def print_status():
    """Print human-readable mount status."""
    status = get_health_status()

    print("\n=== Synology NAS Mount Status ===\n")

    for device_id, info in status["mounts"].items():
        icon = "✓" if info["available"] else "✗"
        state = "MOUNTED" if info["mounted"] else "NOT MOUNTED"

        print(f"{icon} {device_id} ({info['zone']}): {state}")
        print(f"  Path: {info['path']}")

        if info["mounted"]:
            print(
                f"  Usage: {info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB "
                f"({info['percent_used']:.1f}%)"
            )
            print(f"  Free: {info['free_gb']:.1f}GB")
        print()

    print(f"NAS Available: {'Yes' if status['nas_available'] else 'No'}")
    print(
        f"Warm Zone Available: {'Yes' if status['warm_zone_available'] else 'No'}"
    )
    print(
        f"Cold Zone Available: {'Yes' if status['cold_zone_available'] else 'No'}"
    )


if __name__ == "__main__":
    print_status()
