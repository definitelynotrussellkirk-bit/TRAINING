"""
Vault Zones - Zone registration and federation system.

Zones are distributed locations where assets can exist:
    - 4090: Training machine (central Vault)
    - 3090: Inference server
    - NAS: Synology storage

Each zone can run a Branch Officer daemon that tracks local assets
and responds to queries from the central Vault.

Usage:
    from vault.zones import ZoneRegistry, Zone, ZoneClient

    # Register zones
    registry = ZoneRegistry()
    registry.register(Zone(
        zone_id="3090",
        name="Inference Server",
        host="192.168.x.x",
        port=8768,
    ))

    # Check zone status
    client = ZoneClient(zone)
    status = client.get_status()
    assets = client.list_assets()

    # Transfer assets between zones
    from vault.zones import ZoneTransfer
    transfer = ZoneTransfer(source_zone, target_zone)
    result = transfer.push("checkpoint_181000")

RPG Flavor:
    The Realm is divided into Zones, each guarded by a Branch Officer.
    The central Vault coordinates all zones, knowing what treasures
    exist where and orchestrating transfers between them.
"""

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("vault.zones")


class ZoneType(Enum):
    """Types of zones in the realm."""
    CENTRAL = "central"      # Central Vault (4090)
    INFERENCE = "inference"  # Inference server (3090)
    STORAGE = "storage"      # NAS/backup storage
    COMPUTE = "compute"      # Additional compute nodes


class ZoneStatus(Enum):
    """Current status of a zone."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class Zone:
    """
    A zone in the distributed vault system.

    Each zone is a location where assets can exist, typically
    with a Branch Officer daemon running.
    """
    zone_id: str
    name: str
    zone_type: ZoneType = ZoneType.COMPUTE
    host: str = "localhost"
    port: int = 8768
    ssh_user: str = "user"

    # Paths
    base_path: str = ""
    checkpoint_path: str = ""
    model_path: str = ""

    # Status
    status: ZoneStatus = ZoneStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_sync: Optional[datetime] = None

    # Capabilities
    can_train: bool = False
    can_infer: bool = False
    can_store: bool = True

    # Stats (populated from Branch Officer)
    asset_count: int = 0
    total_size_gb: float = 0.0
    disk_free_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "host": self.host,
            "port": self.port,
            "ssh_user": self.ssh_user,
            "base_path": self.base_path,
            "checkpoint_path": self.checkpoint_path,
            "model_path": self.model_path,
            "status": self.status.value,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "can_train": self.can_train,
            "can_infer": self.can_infer,
            "can_store": self.can_store,
            "asset_count": self.asset_count,
            "total_size_gb": self.total_size_gb,
            "disk_free_gb": self.disk_free_gb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        return cls(
            zone_id=data["zone_id"],
            name=data["name"],
            zone_type=ZoneType(data.get("zone_type", "compute")),
            host=data.get("host", "localhost"),
            port=data.get("port", 8768),
            ssh_user=data.get("ssh_user", "user"),
            base_path=data.get("base_path", ""),
            checkpoint_path=data.get("checkpoint_path", ""),
            model_path=data.get("model_path", ""),
            status=ZoneStatus(data.get("status", "unknown")),
            can_train=data.get("can_train", False),
            can_infer=data.get("can_infer", False),
            can_store=data.get("can_store", True),
            asset_count=data.get("asset_count", 0),
            total_size_gb=data.get("total_size_gb", 0.0),
            disk_free_gb=data.get("disk_free_gb", 0.0),
        )

    @property
    def url(self) -> str:
        """Get the base URL for this zone's Branch Officer."""
        return f"http://{self.host}:{self.port}"

    @property
    def rsync_prefix(self) -> str:
        """Get rsync prefix for this zone."""
        return f"{self.ssh_user}@{self.host}:"


class ZoneClient:
    """
    Client for communicating with a Zone's Branch Officer.

    Usage:
        client = ZoneClient(zone)
        status = client.get_status()
        assets = client.list_assets(asset_type="checkpoint")
    """

    def __init__(self, zone: Zone, timeout: int = 10):
        self.zone = zone
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make HTTP request to Branch Officer."""
        url = f"{self.zone.url}{endpoint}"

        try:
            if method == "GET":
                req = Request(url)
            else:
                body = json.dumps(data).encode("utf-8") if data else None
                req = Request(url, data=body, method=method)
                req.add_header("Content-Type", "application/json")

            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            logger.warning(f"HTTP error from {self.zone.zone_id}: {e.code}")
            return None
        except URLError as e:
            logger.warning(f"Connection failed to {self.zone.zone_id}: {e.reason}")
            return None
        except Exception as e:
            logger.warning(f"Request failed to {self.zone.zone_id}: {e}")
            return None

    def is_online(self) -> bool:
        """Check if zone is reachable."""
        result = self._request("/status")
        return result is not None and result.get("status") == "online"

    def get_status(self) -> Optional[Dict]:
        """Get zone status from Branch Officer."""
        return self._request("/status")

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """List assets in this zone."""
        endpoint = "/assets"
        params = []
        if asset_type:
            params.append(f"type={asset_type}")
        if limit != 100:
            params.append(f"limit={limit}")
        if params:
            endpoint += "?" + "&".join(params)

        result = self._request(endpoint)
        return result.get("assets", []) if result else []

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        """Get specific asset details."""
        result = self._request(f"/assets/{asset_id}")
        return result.get("asset") if result else None

    def get_stats(self) -> Optional[Dict]:
        """Get catalog statistics."""
        return self._request("/stats")

    def scan(self, paths: Optional[List[str]] = None) -> Optional[Dict]:
        """Trigger asset scan."""
        return self._request("/scan", method="POST", data={"paths": paths})

    def prepare_receive(
        self,
        asset_id: str,
        asset_type: str = "checkpoint",
        destination: Optional[str] = None,
    ) -> Optional[Dict]:
        """Prepare zone to receive an asset."""
        return self._request(
            "/receive",
            method="POST",
            data={
                "asset_id": asset_id,
                "asset_type": asset_type,
                "destination": destination,
            },
        )

    def register_asset(
        self,
        asset_id: str,
        asset_type: str,
        name: str,
        path: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Register an asset in the zone's catalog."""
        return self._request(
            "/register",
            method="POST",
            data={
                "asset_id": asset_id,
                "asset_type": asset_type,
                "name": name,
                "path": path,
                "metadata": metadata,
            },
        )


class ZoneRegistry:
    """
    Registry of all zones in the realm.

    Persists zone configuration and provides lookup/iteration.
    """

    # Pre-configured zones for our setup
    DEFAULT_ZONES = [
        Zone(
            zone_id="4090",
            name="Training Server",
            zone_type=ZoneType.CENTRAL,
            host="localhost",
            port=8767,  # Central Vault port
            base_path="/path/to/training",
            checkpoint_path="/path/to/training/models",
            model_path="/path/to/training/models",
            can_train=True,
            can_store=True,
        ),
        Zone(
            zone_id="3090",
            name="Inference Server",
            zone_type=ZoneType.INFERENCE,
            host="192.168.x.x",
            port=8768,
            ssh_user="user",
            base_path="/path/to/models",
            checkpoint_path="/path/to/models",
            model_path="/path/to/models",
            can_infer=True,
            can_store=True,
        ),
        Zone(
            zone_id="nas",
            name="Synology NAS",
            zone_type=ZoneType.STORAGE,
            host="192.168.x.x",
            port=8768,
            ssh_user="admin",
            base_path="/volume1/data/llm_training",
            checkpoint_path="/volume1/data/llm_training/checkpoints",
            model_path="/volume1/data/llm_training/models",
            can_store=True,
        ),
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize zone registry.

        Args:
            config_path: Path to zones config file (optional)
        """
        self.config_path = config_path or Path("/path/to/training/config/zones.json")
        self._zones: Dict[str, Zone] = {}
        self._lock = threading.Lock()

        self._load()

    def _load(self):
        """Load zones from config or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                for zone_data in data.get("zones", []):
                    zone = Zone.from_dict(zone_data)
                    self._zones[zone.zone_id] = zone
                logger.info(f"Loaded {len(self._zones)} zones from {self.config_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load zones config: {e}")

        # Use defaults
        for zone in self.DEFAULT_ZONES:
            self._zones[zone.zone_id] = zone
        logger.info(f"Using {len(self._zones)} default zones")

    def _save(self):
        """Save zones to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "zones": [z.to_dict() for z in self._zones.values()],
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, zone: Zone):
        """Register a zone."""
        with self._lock:
            self._zones[zone.zone_id] = zone
            self._save()

    def unregister(self, zone_id: str):
        """Unregister a zone."""
        with self._lock:
            if zone_id in self._zones:
                del self._zones[zone_id]
                self._save()

    def get(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID."""
        return self._zones.get(zone_id)

    def list(self, zone_type: Optional[ZoneType] = None) -> List[Zone]:
        """List all zones, optionally filtered by type."""
        zones = list(self._zones.values())
        if zone_type:
            zones = [z for z in zones if z.zone_type == zone_type]
        return zones

    def get_client(self, zone_id: str) -> Optional[ZoneClient]:
        """Get a client for a zone."""
        zone = self.get(zone_id)
        return ZoneClient(zone) if zone else None

    def check_all_status(self) -> Dict[str, ZoneStatus]:
        """Check status of all zones."""
        results = {}

        for zone_id, zone in self._zones.items():
            # Skip central (we are central)
            if zone.zone_type == ZoneType.CENTRAL:
                zone.status = ZoneStatus.ONLINE
                zone.last_checked = datetime.now()
                results[zone_id] = ZoneStatus.ONLINE
                continue

            client = ZoneClient(zone)
            status_data = client.get_status()

            if status_data and status_data.get("status") == "online":
                zone.status = ZoneStatus.ONLINE
                zone.asset_count = status_data.get("catalog", {}).get("total_assets", 0)
                zone.total_size_gb = status_data.get("catalog", {}).get("total_size_gb", 0)
                zone.disk_free_gb = status_data.get("disk", {}).get("free_gb", 0)
            else:
                zone.status = ZoneStatus.OFFLINE

            zone.last_checked = datetime.now()
            results[zone_id] = zone.status

        self._save()
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all zones."""
        zones = list(self._zones.values())

        online = sum(1 for z in zones if z.status == ZoneStatus.ONLINE)
        total_assets = sum(z.asset_count for z in zones)
        total_storage = sum(z.disk_free_gb for z in zones)

        return {
            "total_zones": len(zones),
            "online_zones": online,
            "offline_zones": len(zones) - online,
            "total_assets": total_assets,
            "total_free_gb": round(total_storage, 1),
            "zones": [z.to_dict() for z in zones],
        }


class ZoneTransfer:
    """
    Transfer assets between zones.

    Uses rsync for efficient file transfer between zones.

    Usage:
        transfer = ZoneTransfer(source_zone, target_zone)
        result = transfer.push(asset_id, source_path)
        result = transfer.pull(asset_id, target_path)
    """

    def __init__(
        self,
        source: Zone,
        target: Zone,
        timeout: int = 3600,  # 1 hour default for large transfers
    ):
        self.source = source
        self.target = target
        self.timeout = timeout

    def push(
        self,
        source_path: str,
        dest_path: Optional[str] = None,
        asset_id: Optional[str] = None,
        delete: bool = False,
    ) -> Dict[str, Any]:
        """
        Push asset from source to target zone.

        Args:
            source_path: Path on source zone
            dest_path: Destination path on target zone
            asset_id: Asset ID (for registration)
            delete: Delete source after successful transfer

        Returns:
            Transfer result dict
        """
        start = datetime.now()

        # Build rsync command
        if self.source.host == "localhost" or self.source.zone_type == ZoneType.CENTRAL:
            # Local to remote
            src = source_path
        else:
            # Remote source
            src = f"{self.source.rsync_prefix}{source_path}"

        if self.target.host == "localhost" or self.target.zone_type == ZoneType.CENTRAL:
            # Remote to local
            dst = dest_path or source_path
        else:
            # Remote target
            dst = f"{self.target.rsync_prefix}{dest_path or self.target.base_path}"

        # Prepare target zone to receive
        if self.target.zone_type != ZoneType.CENTRAL:
            client = ZoneClient(self.target)
            prep = client.prepare_receive(
                asset_id=asset_id or Path(source_path).name,
                destination=dest_path,
            )
            if not prep:
                return {
                    "success": False,
                    "error": f"Target zone {self.target.zone_id} not ready",
                    "source": src,
                    "destination": dst,
                }
            dst = prep.get("rsync_target", dst)

        # Run rsync
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--partial",
            src,
            dst,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            duration = (datetime.now() - start).total_seconds()

            if result.returncode == 0:
                # Register in target zone
                if asset_id and self.target.zone_type != ZoneType.CENTRAL:
                    client = ZoneClient(self.target)
                    client.scan([dest_path] if dest_path else None)

                return {
                    "success": True,
                    "source": src,
                    "destination": dst,
                    "duration_seconds": round(duration, 2),
                    "source_zone": self.source.zone_id,
                    "target_zone": self.target.zone_id,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "source": src,
                    "destination": dst,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Transfer timeout",
                "source": src,
                "destination": dst,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": src,
                "destination": dst,
            }

    def pull(
        self,
        source_path: str,
        dest_path: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pull asset from source zone to local.

        This is essentially push() with swapped source/target perspective.
        """
        # For pull, we swap source and target in the rsync command
        start = datetime.now()

        # Build source spec
        if self.source.host == "localhost" or self.source.zone_type == ZoneType.CENTRAL:
            src = source_path
        else:
            src = f"{self.source.rsync_prefix}{source_path}"

        # Build destination spec (always local for pull)
        dst = dest_path or f"/tmp/{Path(source_path).name}"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--partial",
            src,
            dst,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            duration = (datetime.now() - start).total_seconds()

            if result.returncode == 0:
                return {
                    "success": True,
                    "source": src,
                    "destination": dst,
                    "local_path": dst,
                    "duration_seconds": round(duration, 2),
                    "source_zone": self.source.zone_id,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "source": src,
                    "destination": dst,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Transfer timeout",
                "source": src,
                "destination": dst,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": src,
                "destination": dst,
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_registry: Optional[ZoneRegistry] = None


def get_zone_registry() -> ZoneRegistry:
    """Get or create the zone registry singleton."""
    global _registry
    if _registry is None:
        _registry = ZoneRegistry()
    return _registry


def get_zone(zone_id: str) -> Optional[Zone]:
    """Get a zone by ID."""
    return get_zone_registry().get(zone_id)


def get_zone_client(zone_id: str) -> Optional[ZoneClient]:
    """Get a client for a zone."""
    return get_zone_registry().get_client(zone_id)


def check_zones() -> Dict[str, ZoneStatus]:
    """Check status of all zones."""
    return get_zone_registry().check_all_status()


def push_to_zone(
    asset_path: str,
    target_zone_id: str,
    dest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Push an asset to a zone.

    Args:
        asset_path: Local path to asset
        target_zone_id: Target zone ID
        dest_path: Destination path (optional)

    Returns:
        Transfer result
    """
    registry = get_zone_registry()

    source = registry.get("4090")  # Central vault
    target = registry.get(target_zone_id)

    if not target:
        return {"success": False, "error": f"Unknown zone: {target_zone_id}"}

    transfer = ZoneTransfer(source, target)
    return transfer.push(asset_path, dest_path)


def pull_from_zone(
    asset_path: str,
    source_zone_id: str,
    dest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pull an asset from a zone.

    Args:
        asset_path: Path on source zone
        source_zone_id: Source zone ID
        dest_path: Local destination path

    Returns:
        Transfer result
    """
    registry = get_zone_registry()

    source = registry.get(source_zone_id)
    target = registry.get("4090")  # Central vault

    if not source:
        return {"success": False, "error": f"Unknown zone: {source_zone_id}"}

    transfer = ZoneTransfer(source, target)
    return transfer.pull(asset_path, dest_path)
