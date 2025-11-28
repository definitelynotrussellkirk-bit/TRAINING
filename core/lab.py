"""
Lab - Unified interface for the training lab infrastructure.

This module provides a single entry point for:
- Device capabilities (what can each machine do?)
- Host networking (how do I reach each machine?)
- Storage resolution (where do things live?)

Usage:
    from core.lab import Lab, get_lab

    lab = get_lab()

    # Get current device info
    print(f"Running on: {lab.current_device_id}")
    print(f"Has GPU: {lab.current_device.has_gpu()}")

    # Get paths using storage handles
    ckpt_path = lab.get_checkpoint_path("checkpoint-182000")
    queue_path = lab.get_queue_path("high")

    # Find workers for a job
    eval_workers = lab.get_eval_workers()

    # Get host for SSH/rsync
    host = lab.get_host_for_device("inference3090")
    print(f"SSH: {host.ssh_user}@{host.host}")

Design:
    Lab is a facade that wraps:
    - DeviceRegistry (core/devices.py)
    - HostRegistry (core/hosts.py)
    - StorageResolver (vault/storage_resolver.py)

    It provides a unified, easy-to-use API without needing to import from 3 places.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("lab")


class Lab:
    """
    Unified interface for the training lab.

    Combines device registry, host registry, and storage resolver
    into a single coherent API.

    Usage:
        lab = Lab()

        # Device info
        device = lab.current_device
        workers = lab.get_eval_workers()

        # Storage paths
        path = lab.get_checkpoint_path("checkpoint-182000")

        # Host networking
        host = lab.get_host("4090")
        ssh_target = f"{host.ssh_user}@{host.host}"
    """

    def __init__(self, device_id: Optional[str] = None):
        """
        Initialize the lab interface.

        Args:
            device_id: Override device ID (default: from TRAINING_DEVICE_ID env)
        """
        self._device_id = device_id or os.environ.get("TRAINING_DEVICE_ID")

        # Lazy-load registries
        self._device_registry = None
        self._host_registry = None
        self._storage_resolver = None

    # =========================================================================
    # LAZY LOADERS
    # =========================================================================

    @property
    def device_registry(self):
        """Get device registry (lazy loaded)."""
        if self._device_registry is None:
            from core.devices import get_device_registry
            self._device_registry = get_device_registry()
        return self._device_registry

    @property
    def host_registry(self):
        """Get host registry (lazy loaded)."""
        if self._host_registry is None:
            from core.hosts import get_registry
            self._host_registry = get_registry()
        return self._host_registry

    @property
    def storage_resolver(self):
        """Get storage resolver (lazy loaded)."""
        if self._storage_resolver is None:
            from vault.storage_resolver import StorageResolver
            self._storage_resolver = StorageResolver(device_id=self._device_id)
        return self._storage_resolver

    # =========================================================================
    # CURRENT DEVICE
    # =========================================================================

    @property
    def current_device_id(self) -> Optional[str]:
        """Get current device ID."""
        return self._device_id

    @property
    def current_device(self):
        """Get current device info."""
        if not self._device_id:
            return None
        return self.device_registry.get(self._device_id)

    @property
    def current_host(self):
        """Get host config for current device."""
        if not self._device_id:
            return None
        from core.hosts import get_host_for_device
        return get_host_for_device(self._device_id)

    def is_trainer(self) -> bool:
        """Check if current device is a trainer."""
        from core.devices import DeviceRole
        device = self.current_device
        return device.has_role(DeviceRole.TRAINER) if device else False

    def is_inference(self) -> bool:
        """Check if current device is an inference server."""
        from core.devices import DeviceRole
        device = self.current_device
        return device.has_role(DeviceRole.INFERENCE) if device else False

    # =========================================================================
    # DEVICE QUERIES
    # =========================================================================

    def get_device(self, device_id: str):
        """Get device by ID."""
        return self.device_registry.get(device_id)

    def get_trainer_device(self):
        """Get the primary trainer device."""
        return self.device_registry.get_trainer()

    def get_inference_device(self):
        """Get the primary inference device."""
        return self.device_registry.get_inference()

    def get_eval_workers(self) -> List:
        """Get all eval worker devices."""
        return self.device_registry.get_eval_workers()

    def get_data_forges(self) -> List:
        """Get all data forge devices."""
        return self.device_registry.get_data_forges()

    def get_devices_with_gpu(self) -> List:
        """Get all devices with GPU."""
        return self.device_registry.devices_with_gpu()

    def get_devices_with_role(self, role) -> List:
        """Get devices with specific role."""
        return self.device_registry.devices_with_role(role)

    # =========================================================================
    # HOST QUERIES
    # =========================================================================

    def get_host(self, host_id: str):
        """Get host by ID."""
        return self.host_registry.get(host_id)

    def get_host_for_device(self, device_id: str):
        """Get host config for a device."""
        from core.hosts import get_host_for_device
        return get_host_for_device(device_id)

    def get_service_url(self, service: str, host_id: Optional[str] = None) -> Optional[str]:
        """Get URL for a service."""
        return self.host_registry.get_service_url(service, host_id)

    def get_trainer_host(self):
        """Get trainer host config."""
        return self.host_registry.get_trainer()

    def get_inference_host(self):
        """Get inference host config."""
        return self.host_registry.get_inference()

    # =========================================================================
    # STORAGE PATHS
    # =========================================================================

    def get_checkpoint_path(self, key: str) -> Path:
        """Get path for a checkpoint."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.CHECKPOINT, key)

    def get_snapshot_path(self, key: str) -> Path:
        """Get path for a snapshot."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.SNAPSHOT, key)

    def get_dataset_path(self, key: str) -> Path:
        """Get path for a dataset."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.DATASET, key)

    def get_queue_path(self, priority: str = "normal") -> Path:
        """Get path for a queue directory."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.QUEUE, priority)

    def get_current_model_path(self) -> Path:
        """Get path for current model directory."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.CURRENT_MODEL, "")

    def get_status_path(self, name: str) -> Path:
        """Get path for a status file."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.STATUS, name)

    def get_log_path(self, name: str) -> Path:
        """Get path for a log file."""
        from core.storage_types import StorageKind
        return self.storage_resolver.resolve_default(StorageKind.LOG, name)

    def locate(self, kind, key: str) -> Optional[Path]:
        """Find where an asset exists across zones."""
        return self.storage_resolver.locate(kind, key)

    # =========================================================================
    # COMBINED QUERIES
    # =========================================================================

    def get_ssh_target(self, device_id: str) -> Optional[str]:
        """
        Get SSH target string for a device.

        Returns:
            String like "user@xxx.xxx.88.149" or None
        """
        host = self.get_host_for_device(device_id)
        if host and host.ssh_user:
            return f"{host.ssh_user}@{host.host}"
        return None

    def get_rsync_target(self, device_id: str, path: str = "") -> Optional[str]:
        """
        Get rsync target string for a device.

        Returns:
            String like "user@host:/path/to/models" or None
        """
        host = self.get_host_for_device(device_id)
        if host and host.ssh_user:
            target = f"{host.ssh_user}@{host.host}:"
            if path:
                target += path
            elif host.base_dir:
                target += host.base_dir
            return target
        return None

    def get_device_base_dir(self, device_id: str) -> Optional[str]:
        """Get base directory for a device."""
        host = self.get_host_for_device(device_id)
        return host.base_dir if host else None

    # =========================================================================
    # INFO
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the lab setup."""
        device_summary = self.device_registry.get_summary()
        host_summary = self.host_registry.get_summary()
        resolver_info = self.storage_resolver.get_info()

        return {
            "current_device_id": self._device_id,
            "current_device": self.current_device.device_id if self.current_device else None,
            "is_trainer": self.is_trainer(),
            "is_inference": self.is_inference(),
            "devices": device_summary,
            "hosts": host_summary,
            "storage": resolver_info,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_lab: Optional[Lab] = None


def get_lab(device_id: Optional[str] = None) -> Lab:
    """
    Get or create the Lab singleton.

    Args:
        device_id: Override device ID (only used on first call)

    Returns:
        Lab instance
    """
    global _lab
    if _lab is None:
        _lab = Lab(device_id)
    return _lab


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    lab = get_lab()
    summary = lab.get_summary()

    print("\nLab Summary")
    print("=" * 60)
    print(f"Current device: {summary['current_device_id'] or 'NOT SET'}")
    print(f"Is trainer: {summary['is_trainer']}")
    print(f"Is inference: {summary['is_inference']}")

    print(f"\nDevices ({summary['devices']['total_devices']} total):")
    for d in summary['devices']['devices']:
        roles = ', '.join(d['roles'][:3])
        if len(d['roles']) > 3:
            roles += f" +{len(d['roles'])-3} more"
        print(f"  {d['id']}: {d['hostname']} [{roles}]")

    print(f"\nHosts ({summary['hosts']['total_hosts']} total):")
    for host_id, info in summary['hosts']['hosts'].items():
        services = ', '.join(info['services'][:3]) if info['services'] else 'none'
        print(f"  {host_id}: {info['name']} ({info['role']}) - services: {services}")

    print(f"\nStorage:")
    print(f"  Device: {summary['storage']['device_id']}")
    print(f"  Available zones: {', '.join(summary['storage']['available_zones'])}")

    # Show some example paths
    print("\nExample Paths (if device configured):")
    if summary['current_device']:
        try:
            print(f"  current_model: {lab.get_current_model_path()}")
            print(f"  queue/high: {lab.get_queue_path('high')}")
            print(f"  checkpoint-182000: {lab.get_checkpoint_path('checkpoint-182000')}")
        except Exception as e:
            print(f"  (error: {e})")
    else:
        print("  Set TRAINING_DEVICE_ID to see paths")

    # Show SSH targets
    print("\nSSH Targets:")
    for d in summary['devices']['devices']:
        ssh = lab.get_ssh_target(d['id'])
        if ssh:
            print(f"  {d['id']}: {ssh}")
