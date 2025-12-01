#!/usr/bin/env python3
"""
Per-Device Checkpoint Retention - Smart cleanup based on device config.

Each device (trainer, inference, NAS) has different storage limits.
This module enforces those limits by deleting least-recently-used checkpoints.

Key Concepts:
    - Vault (NAS): Keeps ALL checkpoints - the canonical archive
    - Edge devices: Keep only X most recently USED checkpoints
    - "Recently used" = last_used timestamp, NOT creation time

Usage:
    # Run retention for local device
    python3 -m management.device_retention

    # Run for specific device
    python3 -m management.device_retention --device inference3090

    # Dry run (show what would be deleted)
    python3 -m management.device_retention --dry-run

    # From code
    from management.device_retention import enforce_device_retention
    deleted = enforce_device_retention("trainer4090", dry_run=False)

Config in hosts.json:
    "checkpoint_retention": {
        "max_checkpoints": 5,
        "max_gb": 20,
        "keep_strategy": "recently_used",
        "is_vault": false
    }
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RetentionConfig:
    """Per-device retention configuration."""
    max_checkpoints: Optional[int]  # None = unlimited
    max_gb: Optional[float]  # None = unlimited
    keep_strategy: str  # "recently_used", "all", "recent"
    is_vault: bool  # If True, never delete (this is the archive)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetentionConfig":
        return cls(
            max_checkpoints=data.get("max_checkpoints"),
            max_gb=data.get("max_gb"),
            keep_strategy=data.get("keep_strategy", "recently_used"),
            is_vault=data.get("is_vault", False),
        )

    @classmethod
    def default(cls) -> "RetentionConfig":
        """Default config: keep 10 checkpoints, 50GB max."""
        return cls(
            max_checkpoints=10,
            max_gb=50,
            keep_strategy="recently_used",
            is_vault=False,
        )


@dataclass
class RetentionResult:
    """Result of a retention run."""
    device_id: str
    checkpoints_before: int
    checkpoints_after: int
    deleted_count: int
    deleted_steps: List[int]
    freed_bytes: int
    dry_run: bool
    errors: List[str]

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "checkpoints_before": self.checkpoints_before,
            "checkpoints_after": self.checkpoints_after,
            "deleted_count": self.deleted_count,
            "deleted_steps": self.deleted_steps,
            "freed_gb": round(self.freed_gb, 2),
            "dry_run": self.dry_run,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat(),
        }


def resolve_device_id(host_id: str) -> str:
    """
    Resolve a host key (e.g., "4090") to its device_id (e.g., "trainer4090").

    Args:
        host_id: Host key from hosts.json (e.g., "4090", "3090", "nas")

    Returns:
        device_id from the host config
    """
    try:
        from core.hosts import get_host
        host = get_host(host_id)
        if host and host.device_id:
            return host.device_id
    except Exception:
        pass

    # Fallback: try direct file read
    try:
        from core.paths import get_base_dir
        hosts_file = get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                data = json.load(f)
            host_data = data.get("hosts", {}).get(host_id, {})
            if host_data.get("device_id"):
                return host_data["device_id"]
    except Exception:
        pass

    return host_id  # Return as-is if no mapping found


def get_retention_config(host_id: str) -> RetentionConfig:
    """
    Get retention config for a host from hosts.json.

    Args:
        host_id: Host ID (e.g., "4090", "3090", "nas")

    Returns:
        RetentionConfig for this host
    """
    try:
        from core.hosts import get_host
        host = get_host(host_id)
        if host and hasattr(host, 'checkpoint_retention'):
            return RetentionConfig.from_dict(host.checkpoint_retention)
    except Exception as e:
        logger.warning(f"Failed to get host config for {host_id}: {e}")

    # Try direct file read as fallback
    try:
        from core.paths import get_base_dir
        hosts_file = get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                data = json.load(f)
            host_data = data.get("hosts", {}).get(host_id, {})
            retention_data = host_data.get("checkpoint_retention")
            if retention_data:
                return RetentionConfig.from_dict(retention_data)
    except Exception as e:
        logger.warning(f"Failed to read hosts.json for {host_id}: {e}")

    return RetentionConfig.default()


def get_local_device_id() -> str:
    """Get the device_id for the local machine."""
    try:
        from core.hosts import get_local_host
        local = get_local_host()
        if local:
            return local.device_id
    except Exception:
        pass

    # Fallback: try to detect from hostname
    import socket
    hostname = socket.gethostname().lower()

    # Common patterns
    if "4090" in hostname or "trainer" in hostname:
        return "trainer4090"
    elif "3090" in hostname or "inference" in hostname:
        return "inference3090"

    return "trainer4090"  # Default


def get_checkpoints_dir(host_id: str) -> Optional[Path]:
    """Get the checkpoints directory for a host."""
    try:
        from core.hosts import get_host
        host = get_host(host_id)
        if host and host.checkpoints_dir:
            return Path(host.checkpoints_dir)
    except Exception:
        pass

    # Fallback for local device
    try:
        from core.paths import get_base_dir
        return get_base_dir() / "models" / "current_model"
    except Exception:
        pass

    return None


def enforce_device_retention(
    host_id: Optional[str] = None,
    dry_run: bool = False,
) -> RetentionResult:
    """
    Enforce checkpoint retention for a device.

    Deletes least-recently-used checkpoints beyond the configured limit.
    Never deletes the only copy of a checkpoint.

    Args:
        host_id: Host key (e.g., "4090") or device_id (default: local device)
        dry_run: If True, only report what would be deleted

    Returns:
        RetentionResult with details of what was done
    """
    from core.checkpoint_ledger import get_ledger

    # Resolve device_id from host_id
    if host_id is None:
        device_id = get_local_device_id()
        # Also need to figure out the host_id for config lookup
        host_id = "4090"  # Default
    else:
        device_id = resolve_device_id(host_id)

    # Get config using host_id
    config = get_retention_config(host_id)

    # Initialize result
    result = RetentionResult(
        device_id=device_id,
        checkpoints_before=0,
        checkpoints_after=0,
        deleted_count=0,
        deleted_steps=[],
        freed_bytes=0,
        dry_run=dry_run,
        errors=[],
    )

    # Vault never deletes
    if config.is_vault:
        logger.info(f"Device {device_id} is vault - skipping retention")
        return result

    # No limit = no retention
    if config.max_checkpoints is None and config.max_gb is None:
        logger.info(f"Device {device_id} has no limits - skipping retention")
        return result

    # Get ledger
    ledger = get_ledger()

    # Get checkpoints on this device
    checkpoints = ledger.list_by_device(device_id, sort_by_usage=True)
    result.checkpoints_before = len(checkpoints)

    if not checkpoints:
        logger.info(f"No checkpoints found for device {device_id}")
        return result

    # Determine how many to keep
    keep_count = config.max_checkpoints or len(checkpoints)

    # Also check size limit
    if config.max_gb:
        total_bytes = sum(c.size_bytes or 0 for c in checkpoints)
        max_bytes = config.max_gb * (1024 ** 3)

        # If over size limit, reduce keep_count
        if total_bytes > max_bytes:
            cumulative = 0
            for i, c in enumerate(checkpoints):
                cumulative += c.size_bytes or 0
                if cumulative > max_bytes:
                    keep_count = min(keep_count, i)
                    break

    # Get candidates for deletion
    candidates = ledger.get_retention_candidates(device_id, keep_count)

    if not candidates:
        logger.info(f"Device {device_id}: {len(checkpoints)} checkpoints, all within limits")
        result.checkpoints_after = len(checkpoints)
        return result

    logger.info(f"Device {device_id}: {len(checkpoints)} checkpoints, {len(candidates)} candidates for deletion")

    # Get checkpoints dir
    checkpoints_dir = get_checkpoints_dir(device_id)

    # Delete candidates
    for record in candidates:
        step = record.step
        checkpoint_path = Path(record.path)

        # Safety: Don't delete if only copy (need NAS backup first)
        # NAS device_id is "synology_data" per hosts.json
        has_nas_copy = "synology_data" in record.locations
        if len(record.locations) <= 1 and not has_nas_copy:
            result.errors.append(f"Step {step}: Only copy, skipping")
            continue

        # Try to find the checkpoint on disk
        if not checkpoint_path.exists() and checkpoints_dir:
            # Try to find it in the checkpoints dir
            for item in checkpoints_dir.iterdir():
                if item.is_dir() and f"checkpoint-{step}" in item.name:
                    checkpoint_path = item
                    break

        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {checkpoint_path}")
            result.deleted_steps.append(step)
            result.deleted_count += 1
            result.freed_bytes += record.size_bytes or 0
        else:
            try:
                if checkpoint_path.exists():
                    size = sum(f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file())
                    shutil.rmtree(checkpoint_path)
                    ledger.remove_location(step, device_id)
                    result.deleted_steps.append(step)
                    result.deleted_count += 1
                    result.freed_bytes += size
                    logger.info(f"Deleted: {checkpoint_path} ({size / 1e9:.2f} GB)")
                else:
                    # Path doesn't exist - just update ledger
                    ledger.remove_location(step, device_id)
                    result.deleted_steps.append(step)
                    logger.info(f"Removed from ledger (not on disk): step {step}")
            except Exception as e:
                result.errors.append(f"Step {step}: {e}")
                logger.error(f"Failed to delete checkpoint {step}: {e}")

    result.checkpoints_after = result.checkpoints_before - result.deleted_count

    # Log summary
    if result.deleted_count > 0:
        logger.info(
            f"Retention complete for {device_id}: "
            f"deleted {result.deleted_count} checkpoints, "
            f"freed {result.freed_gb:.2f} GB"
        )

    return result


def enforce_all_devices(dry_run: bool = False) -> Dict[str, RetentionResult]:
    """
    Enforce retention on all configured devices.

    Returns:
        Dict mapping device_id to RetentionResult
    """
    results = {}

    try:
        from core.paths import get_base_dir
        hosts_file = get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                data = json.load(f)
            for host_id in data.get("hosts", {}):
                result = enforce_device_retention(host_id, dry_run=dry_run)
                results[host_id] = result
    except Exception as e:
        logger.error(f"Failed to enforce all devices: {e}")

    return results


# Convenience function for common use case
def cleanup_local_checkpoints(dry_run: bool = False) -> RetentionResult:
    """Clean up checkpoints on the local device."""
    return enforce_device_retention(host_id=None, dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Per-device checkpoint retention")
    parser.add_argument("--device", type=str, help="Host ID e.g. '4090', '3090', 'nas' (default: local)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--all", action="store_true", help="Run on all devices")
    parser.add_argument("--status", action="store_true", help="Show retention status")

    args = parser.parse_args()

    if args.status:
        # Show status for device
        host_id = args.device or "4090"  # Default host key
        device_id = resolve_device_id(host_id)
        config = get_retention_config(host_id)

        print(f"\nHost: {host_id} (device_id: {device_id})")
        print(f"Config:")
        print(f"  max_checkpoints: {config.max_checkpoints}")
        print(f"  max_gb: {config.max_gb}")
        print(f"  keep_strategy: {config.keep_strategy}")
        print(f"  is_vault: {config.is_vault}")

        from core.checkpoint_ledger import get_ledger
        ledger = get_ledger()
        checkpoints = ledger.list_by_device(device_id)
        total_size = sum(c.size_bytes or 0 for c in checkpoints)

        print(f"\nCheckpoints on device: {len(checkpoints)}")
        print(f"Total size: {total_size / 1e9:.2f} GB")

        if checkpoints:
            print(f"\nMost recent 5:")
            for c in checkpoints[:5]:
                used = c.last_used[:19] if c.last_used else "never"
                print(f"  {c.step}: {c.size_gb:.2f} GB, last_used={used}")

    elif args.all:
        results = enforce_all_devices(dry_run=args.dry_run)
        print("\nRetention Results:")
        for host_id, result in results.items():
            print(f"  {host_id} ({result.device_id}): deleted {result.deleted_count}, freed {result.freed_gb:.2f} GB")

    else:
        result = enforce_device_retention(host_id=args.device, dry_run=args.dry_run)
        print(json.dumps(result.to_dict(), indent=2))
