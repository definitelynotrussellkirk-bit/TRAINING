"""
Device Registry - Register and query devices in the training lab.

Each device has:
- Roles: What jobs it can perform (trainer, eval_worker, data_forge, etc.)
- Capabilities: Hardware specs (GPUs, CPU, memory)
- Storage zones: Which temperature zones it hosts (hot, warm, cold)

Usage:
    from core.devices import get_device_registry, get_current_device, DeviceRole

    # Get registry
    registry = get_device_registry()

    # Find all eval workers
    workers = registry.devices_with_role(DeviceRole.EVAL_WORKER)

    # Get current device (from TRAINING_DEVICE_ID env var)
    device = get_current_device()
    print(f"Running on: {device.device_id}")
    print(f"Has GPU: {device.has_gpu()}")

    # Check capabilities
    if device.has_role(DeviceRole.TRAINER):
        print(f"Can train with {device.total_vram()}GB VRAM")

Configuration:
    Devices are defined in config/devices.json with schema:
    {
      "devices": {
        "trainer4090": {
          "hostname": "192.168.x.x",
          "roles": ["trainer", "eval_worker", "storage_hot"],
          "gpus": [{"name": "RTX 4090", "count": 1, "vram_gb": 24}],
          "cpu": {"cores": 16, "threads": 32},
          "memory_gb": 64,
          "storage_zones": ["hot"],
          "network": {"speed_gbps": 10, "tags": ["lan_core"]}
        }
      }
    }
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("devices")


# =============================================================================
# ENUMS
# =============================================================================

class DeviceRole(str, Enum):
    """
    Roles a device can perform.

    Devices can have multiple roles simultaneously.
    """
    # Compute roles
    TRAINER = "trainer"             # GPU training (primary training node)
    INFERENCE = "inference"         # Model serving for inference
    EVAL_WORKER = "eval_worker"     # Runs skill evaluations (passives)
    DATA_FORGE = "data_forge"       # Generates/filters training data
    VAULT_WORKER = "vault_worker"   # Archive/retention operations
    ANALYTICS = "analytics"         # Dashboards, metrics aggregation

    # Storage roles (which temperature zones this device hosts)
    STORAGE_HOT = "storage_hot"     # Fast local NVMe storage
    STORAGE_WARM = "storage_warm"   # Primary NAS storage
    STORAGE_COLD = "storage_cold"   # Archive NAS storage

    # Control roles
    CONTROL_PLANE = "control_plane"  # Orchestration services (Tavern, VaultKeeper)

    # Future: distributed training
    DISTRIBUTED_TRAINER = "distributed_trainer"  # Part of multi-GPU training cluster


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str           # e.g., "RTX 4090"
    count: int = 1      # Number of this GPU type
    vram_gb: float = 0  # VRAM per GPU in GB

    def total_vram(self) -> float:
        """Total VRAM across all GPUs of this type."""
        return self.vram_gb * self.count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "vram_gb": self.vram_gb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUInfo":
        return cls(
            name=data.get("name", "Unknown GPU"),
            count=data.get("count", 1),
            vram_gb=data.get("vram_gb", 0),
        )


@dataclass
class CPUInfo:
    """CPU hardware information."""
    cores: int = 1      # Physical cores
    threads: int = 1    # Logical threads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cores": self.cores,
            "threads": self.threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CPUInfo":
        return cls(
            cores=data.get("cores", 1),
            threads=data.get("threads", data.get("cores", 1)),
        )


@dataclass
class NetworkInfo:
    """Network connectivity information."""
    speed_gbps: float = 1.0     # Network speed in Gbps
    tags: List[str] = field(default_factory=list)  # Tags like "lan_core", "nas"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed_gbps": self.speed_gbps,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkInfo":
        return cls(
            speed_gbps=data.get("speed_gbps", 1.0),
            tags=data.get("tags", []),
        )


@dataclass
class DeviceInfo:
    """
    Complete information about a device in the lab.

    A device is a physical or virtual machine that can perform
    various roles in the training pipeline.
    """
    device_id: str                          # Unique identifier
    hostname: str                           # IP or hostname
    roles: List[DeviceRole] = field(default_factory=list)
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory_gb: int = 0
    storage_zones: List[str] = field(default_factory=list)  # ["hot", "warm", "cold"]
    network: NetworkInfo = field(default_factory=NetworkInfo)

    # Optional metadata
    description: str = ""
    enabled: bool = True  # Can disable devices without removing from config

    def has_role(self, role: DeviceRole) -> bool:
        """Check if device has a specific role."""
        return role in self.roles

    def has_any_role(self, roles: List[DeviceRole]) -> bool:
        """Check if device has any of the specified roles."""
        return any(role in self.roles for role in roles)

    def has_all_roles(self, roles: List[DeviceRole]) -> bool:
        """Check if device has all of the specified roles."""
        return all(role in self.roles for role in roles)

    def has_gpu(self) -> bool:
        """Check if device has any GPU."""
        return len(self.gpus) > 0

    def total_vram(self) -> float:
        """Total VRAM across all GPUs in GB."""
        return sum(gpu.total_vram() for gpu in self.gpus)

    def gpu_count(self) -> int:
        """Total number of GPUs."""
        return sum(gpu.count for gpu in self.gpus)

    def has_storage_zone(self, zone: str) -> bool:
        """Check if device hosts a specific storage zone."""
        return zone in self.storage_zones

    def is_storage_device(self) -> bool:
        """Check if device has any storage role."""
        storage_roles = {
            DeviceRole.STORAGE_HOT,
            DeviceRole.STORAGE_WARM,
            DeviceRole.STORAGE_COLD,
        }
        return any(role in self.roles for role in storage_roles)

    def is_compute_device(self) -> bool:
        """Check if device has any compute role."""
        compute_roles = {
            DeviceRole.TRAINER,
            DeviceRole.INFERENCE,
            DeviceRole.EVAL_WORKER,
            DeviceRole.DATA_FORGE,
        }
        return any(role in self.roles for role in compute_roles)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "hostname": self.hostname,
            "roles": [r.value for r in self.roles],
            "gpus": [g.to_dict() for g in self.gpus],
            "cpu": self.cpu.to_dict(),
            "memory_gb": self.memory_gb,
            "storage_zones": self.storage_zones,
            "network": self.network.to_dict(),
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, device_id: str, data: Dict[str, Any]) -> "DeviceInfo":
        # Parse roles
        roles = []
        for role_str in data.get("roles", []):
            try:
                roles.append(DeviceRole(role_str))
            except ValueError:
                logger.warning(f"Unknown role '{role_str}' for device {device_id}")

        # Parse GPUs
        gpus = [GPUInfo.from_dict(g) for g in data.get("gpus", [])]

        # Parse CPU
        cpu_data = data.get("cpu", {})
        cpu = CPUInfo.from_dict(cpu_data) if cpu_data else CPUInfo()

        # Parse network
        net_data = data.get("network", {})
        network = NetworkInfo.from_dict(net_data) if net_data else NetworkInfo()

        return cls(
            device_id=device_id,
            hostname=data.get("hostname", "localhost"),
            roles=roles,
            gpus=gpus,
            cpu=cpu,
            memory_gb=data.get("memory_gb", 0),
            storage_zones=data.get("storage_zones", []),
            network=network,
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )


# =============================================================================
# REGISTRY
# =============================================================================

class DeviceRegistry:
    """
    Registry of all devices in the training lab.

    Loads configuration from config/devices.json and provides
    lookup and filtering methods.

    Usage:
        registry = DeviceRegistry()

        # Get specific device
        trainer = registry.get("trainer4090")

        # Find devices by role
        workers = registry.devices_with_role(DeviceRole.EVAL_WORKER)

        # Find devices with GPU
        gpu_devices = registry.devices_with_gpu()
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize device registry.

        Args:
            config_path: Path to devices.json. If None, auto-detects.
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Auto-detect based on this file's location
            base_dir = Path(__file__).parent.parent
            self.config_path = base_dir / "config" / "devices.json"

        self._devices: Dict[str, DeviceInfo] = {}
        self._lock = threading.Lock()
        self._loaded = False

        self._load()

    def _load(self) -> None:
        """Load devices from config file."""
        if not self.config_path.exists():
            logger.warning(
                f"Device config not found: {self.config_path}. "
                "Using empty registry. Create config/devices.json to define devices."
            )
            self._loaded = True
            return

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            devices_data = data.get("devices", {})
            for device_id, device_data in devices_data.items():
                device = DeviceInfo.from_dict(device_id, device_data)
                self._devices[device_id] = device

            logger.info(f"Loaded {len(self._devices)} devices from {self.config_path}")
            self._loaded = True

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load devices: {e}")
            raise

    def reload(self) -> None:
        """Reload configuration from disk."""
        with self._lock:
            self._devices.clear()
            self._loaded = False
            self._load()

    # =========================================================================
    # LOOKUPS
    # =========================================================================

    def get(self, device_id: str) -> Optional[DeviceInfo]:
        """Get device by ID."""
        return self._devices.get(device_id)

    def get_or_raise(self, device_id: str) -> DeviceInfo:
        """Get device by ID, raising if not found."""
        device = self._devices.get(device_id)
        if not device:
            raise KeyError(f"Device not found: {device_id}")
        return device

    def all_devices(self) -> List[DeviceInfo]:
        """Get all registered devices."""
        return list(self._devices.values())

    def enabled_devices(self) -> List[DeviceInfo]:
        """Get all enabled devices."""
        return [d for d in self._devices.values() if d.enabled]

    def device_ids(self) -> List[str]:
        """Get all device IDs."""
        return list(self._devices.keys())

    # =========================================================================
    # FILTERING
    # =========================================================================

    def devices_with_role(
        self,
        role: DeviceRole,
        enabled_only: bool = True,
    ) -> List[DeviceInfo]:
        """Find devices with a specific role."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.has_role(role)]

    def devices_with_any_role(
        self,
        roles: List[DeviceRole],
        enabled_only: bool = True,
    ) -> List[DeviceInfo]:
        """Find devices with any of the specified roles."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.has_any_role(roles)]

    def devices_with_storage_zone(
        self,
        zone: str,
        enabled_only: bool = True,
    ) -> List[DeviceInfo]:
        """Find devices that host a specific storage zone."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.has_storage_zone(zone)]

    def devices_with_gpu(self, enabled_only: bool = True) -> List[DeviceInfo]:
        """Find devices with GPU capability."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.has_gpu()]

    def compute_devices(self, enabled_only: bool = True) -> List[DeviceInfo]:
        """Find devices with compute roles."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.is_compute_device()]

    def storage_devices(self, enabled_only: bool = True) -> List[DeviceInfo]:
        """Find devices with storage roles."""
        devices = self.enabled_devices() if enabled_only else self.all_devices()
        return [d for d in devices if d.is_storage_device()]

    # =========================================================================
    # SPECIAL LOOKUPS
    # =========================================================================

    def get_trainer(self) -> Optional[DeviceInfo]:
        """Get the primary trainer device."""
        trainers = self.devices_with_role(DeviceRole.TRAINER)
        return trainers[0] if trainers else None

    def get_inference(self) -> Optional[DeviceInfo]:
        """Get the primary inference device."""
        inference = self.devices_with_role(DeviceRole.INFERENCE)
        return inference[0] if inference else None

    def get_eval_workers(self) -> List[DeviceInfo]:
        """Get all eval worker devices."""
        return self.devices_with_role(DeviceRole.EVAL_WORKER)

    def get_data_forges(self) -> List[DeviceInfo]:
        """Get all data forge devices."""
        return self.devices_with_role(DeviceRole.DATA_FORGE)

    # =========================================================================
    # INFO
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of registered devices."""
        devices = self.all_devices()
        enabled = [d for d in devices if d.enabled]

        # Count roles
        role_counts = {}
        for device in enabled:
            for role in device.roles:
                role_counts[role.value] = role_counts.get(role.value, 0) + 1

        # Count zones
        zone_counts = {}
        for device in enabled:
            for zone in device.storage_zones:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

        # Total resources
        total_vram = sum(d.total_vram() for d in enabled)
        total_memory = sum(d.memory_gb for d in enabled)
        total_gpus = sum(d.gpu_count() for d in enabled)

        return {
            "total_devices": len(devices),
            "enabled_devices": len(enabled),
            "total_gpus": total_gpus,
            "total_vram_gb": total_vram,
            "total_memory_gb": total_memory,
            "role_counts": role_counts,
            "zone_counts": zone_counts,
            "devices": [
                {
                    "id": d.device_id,
                    "hostname": d.hostname,
                    "roles": [r.value for r in d.roles],
                    "gpus": d.gpu_count(),
                    "vram_gb": d.total_vram(),
                    "enabled": d.enabled,
                }
                for d in devices
            ],
        }


# =============================================================================
# SINGLETON + HELPERS
# =============================================================================

_registry: Optional[DeviceRegistry] = None
_registry_lock = threading.Lock()


def get_device_registry() -> DeviceRegistry:
    """Get or create the device registry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                config_path = os.environ.get("DEVICES_CONFIG_PATH")
                _registry = DeviceRegistry(
                    Path(config_path) if config_path else None
                )
    return _registry


def get_current_device_id() -> Optional[str]:
    """
    Get the current device ID from environment.

    Set TRAINING_DEVICE_ID to identify which device this code is running on.
    """
    return os.environ.get("TRAINING_DEVICE_ID")


def get_current_device() -> Optional[DeviceInfo]:
    """
    Get the DeviceInfo for the current device.

    Returns None if TRAINING_DEVICE_ID is not set or device not found.
    """
    device_id = get_current_device_id()
    if not device_id:
        return None
    return get_device_registry().get(device_id)


def get_current_device_or_raise() -> DeviceInfo:
    """
    Get the DeviceInfo for the current device, raising if not configured.

    Raises:
        RuntimeError: If TRAINING_DEVICE_ID not set
        KeyError: If device not found in registry
    """
    device_id = get_current_device_id()
    if not device_id:
        raise RuntimeError(
            "TRAINING_DEVICE_ID environment variable not set. "
            "Set it to identify which device this code is running on."
        )
    return get_device_registry().get_or_raise(device_id)


def is_current_device(device_id: str) -> bool:
    """Check if the given device ID is the current device."""
    return get_current_device_id() == device_id


def current_device_has_role(role: DeviceRole) -> bool:
    """Check if current device has a specific role."""
    device = get_current_device()
    return device.has_role(role) if device else False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    registry = get_device_registry()
    summary = registry.get_summary()

    print(f"\nDevice Registry Summary")
    print(f"=" * 50)
    print(f"Total devices: {summary['total_devices']}")
    print(f"Enabled: {summary['enabled_devices']}")
    print(f"Total GPUs: {summary['total_gpus']}")
    print(f"Total VRAM: {summary['total_vram_gb']} GB")
    print(f"Total Memory: {summary['total_memory_gb']} GB")

    print(f"\nRoles:")
    for role, count in sorted(summary['role_counts'].items()):
        print(f"  {role}: {count}")

    print(f"\nStorage Zones:")
    for zone, count in sorted(summary['zone_counts'].items()):
        print(f"  {zone}: {count}")

    print(f"\nDevices:")
    for d in summary['devices']:
        status = "✓" if d['enabled'] else "✗"
        gpu_info = f"{d['gpus']} GPU ({d['vram_gb']}GB)" if d['gpus'] else "no GPU"
        print(f"  [{status}] {d['id']}: {d['hostname']} - {gpu_info}")
        print(f"      roles: {', '.join(d['roles'])}")

    # Show current device if set
    current = get_current_device()
    if current:
        print(f"\nCurrent device: {current.device_id}")
    else:
        device_id = get_current_device_id()
        if device_id:
            print(f"\nWarning: TRAINING_DEVICE_ID={device_id} but device not found")
        else:
            print(f"\nNote: TRAINING_DEVICE_ID not set")
