"""
Storage Registry - Register and manage storage locations in the Vault.

The Storage Registry allows registering multiple storage locations
(local, NAS, cloud) with the Vault for treasure storage:

    Strongholds    - Registered storage locations
    Profiles       - Configuration for each stronghold
    Allocation     - Space allocation per stronghold

RPG Flavor:
    The Vault has multiple Strongholds across the realm where treasures
    can be stored. The local Vault is the primary stronghold, but the
    Synology Citadel provides remote archival storage.

Stronghold Types:
    LOCAL       - Local filesystem storage
    NAS         - Network Attached Storage (Synology, etc.)
    CLOUD       - Cloud storage (S3, GCS, etc.)
    REMOTE      - Remote server storage

Usage:
    from vault import StorageRegistry

    registry = StorageRegistry(base_dir)

    # Register the Synology NAS
    registry.register_stronghold(
        name="synology_citadel",
        stronghold_type=StrongholdType.NAS,
        host="nas.local",
        profile=StorageProfile(
            allocation_tb=10,
            retention_days=30,
            sync_schedule="daily"
        )
    )

    # List strongholds
    for sh in registry.list_strongholds():
        print(f"{sh.name}: {sh.status}")

    # Get storage recommendations
    recs = registry.recommend_storage(treasure_type="checkpoint", size_gb=5)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class StrongholdType(Enum):
    """Type of storage stronghold."""
    LOCAL = "local"          # Local filesystem
    NAS = "nas"              # Network Attached Storage
    CLOUD = "cloud"          # Cloud storage (S3, etc.)
    REMOTE = "remote"        # Remote server (SSH)


class StrongholdStatus(Enum):
    """Current status of a stronghold."""
    ACTIVE = "active"        # Available and connected
    OFFLINE = "offline"      # Not reachable
    DEGRADED = "degraded"    # Available but issues
    MAINTENANCE = "maintenance"  # Under maintenance
    UNKNOWN = "unknown"


class SyncSchedule(Enum):
    """When to sync to this stronghold."""
    ON_SAVE = "on_save"      # Sync immediately when saved
    ON_DEPLOY = "on_deploy"  # Sync when deployed
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"        # Only on manual trigger


@dataclass
class StorageProfile:
    """
    Profile for a storage stronghold.

    Defines allocation, retention, and sync behavior.
    """
    # Allocation
    allocation_tb: float = 1.0        # Allocated space in TB
    max_size_gb: Optional[float] = None  # Max size per folder

    # Retention
    retention_days: Optional[int] = None  # Days to keep (None = forever)

    # Sync
    sync_schedule: SyncSchedule = SyncSchedule.DAILY

    # Content types to store
    store_checkpoints: bool = True
    store_models: bool = True
    store_backups: bool = True
    store_logs: bool = False
    store_snapshots: bool = True

    # Cleanup
    auto_cleanup: bool = True
    cleanup_threshold_pct: float = 90.0  # Cleanup when above this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation_tb": self.allocation_tb,
            "max_size_gb": self.max_size_gb,
            "retention_days": self.retention_days,
            "sync_schedule": self.sync_schedule.value,
            "store_checkpoints": self.store_checkpoints,
            "store_models": self.store_models,
            "store_backups": self.store_backups,
            "store_logs": self.store_logs,
            "store_snapshots": self.store_snapshots,
            "auto_cleanup": self.auto_cleanup,
            "cleanup_threshold_pct": self.cleanup_threshold_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageProfile":
        schedule = data.get("sync_schedule", "daily")
        if isinstance(schedule, str):
            schedule = SyncSchedule(schedule)

        return cls(
            allocation_tb=data.get("allocation_tb", 1.0),
            max_size_gb=data.get("max_size_gb"),
            retention_days=data.get("retention_days"),
            sync_schedule=schedule,
            store_checkpoints=data.get("store_checkpoints", True),
            store_models=data.get("store_models", True),
            store_backups=data.get("store_backups", True),
            store_logs=data.get("store_logs", False),
            store_snapshots=data.get("store_snapshots", True),
            auto_cleanup=data.get("auto_cleanup", True),
            cleanup_threshold_pct=data.get("cleanup_threshold_pct", 90.0),
        )


@dataclass
class Stronghold:
    """
    A registered storage stronghold.
    """
    name: str                           # Unique identifier
    stronghold_type: StrongholdType
    description: str = ""

    # Location
    host: Optional[str] = None          # Host address (for NAS/remote)
    share: Optional[str] = None         # Share name (for NAS)
    base_path: str = ""                 # Base path within storage
    mount_point: Optional[str] = None   # Local mount point

    # Profile
    profile: StorageProfile = field(default_factory=StorageProfile)

    # Status
    status: StrongholdStatus = StrongholdStatus.UNKNOWN
    last_checked: Optional[datetime] = None
    last_sync: Optional[datetime] = None

    # Credentials (reference, not stored here)
    credentials_key: Optional[str] = None

    # Stats
    used_gb: float = 0.0
    free_gb: float = 0.0
    total_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stronghold_type": self.stronghold_type.value,
            "description": self.description,
            "host": self.host,
            "share": self.share,
            "base_path": self.base_path,
            "mount_point": self.mount_point,
            "profile": self.profile.to_dict(),
            "status": self.status.value,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "credentials_key": self.credentials_key,
            "used_gb": self.used_gb,
            "free_gb": self.free_gb,
            "total_gb": self.total_gb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stronghold":
        return cls(
            name=data["name"],
            stronghold_type=StrongholdType(data.get("stronghold_type", "local")),
            description=data.get("description", ""),
            host=data.get("host"),
            share=data.get("share"),
            base_path=data.get("base_path", ""),
            mount_point=data.get("mount_point"),
            profile=StorageProfile.from_dict(data.get("profile", {})),
            status=StrongholdStatus(data.get("status", "unknown")),
            credentials_key=data.get("credentials_key"),
            used_gb=data.get("used_gb", 0.0),
            free_gb=data.get("free_gb", 0.0),
            total_gb=data.get("total_gb", 0.0),
        )


class StorageRegistry:
    """
    Registry for storage strongholds.

    Manages registration, profiles, and recommendations for
    storing treasures across multiple locations.

    Usage:
        registry = StorageRegistry(base_dir)

        # Register a NAS
        registry.register_stronghold(
            name="synology",
            stronghold_type=StrongholdType.NAS,
            host="nas.local",
            share="data",
            base_path="llm_training",
            profile=StorageProfile(allocation_tb=10)
        )

        # Check status
        for sh in registry.list_strongholds():
            print(f"{sh.name}: {sh.status.value}")

        # Get recommendation for storing a checkpoint
        rec = registry.recommend_storage("checkpoint", size_gb=5)
    """

    def __init__(self, base_dir: Optional[str | Path] = None):
        """
        Initialize the Storage Registry.

        Args:
            base_dir: Base training directory (default: auto-detect)
        """
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.registry_file = self.config_dir / "storage_registry.json"
        self.legacy_config = self.config_dir / "storage.json"

        self._strongholds: Dict[str, Stronghold] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from file or migrate from legacy."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
            for sh_data in data.get("strongholds", []):
                sh = Stronghold.from_dict(sh_data)
                self._strongholds[sh.name] = sh
        elif self.legacy_config.exists():
            # Migrate from legacy storage.json
            self._migrate_legacy_config()
        else:
            # Register local storage by default
            self._register_local()

    def _migrate_legacy_config(self):
        """Migrate from legacy storage.json format."""
        with open(self.legacy_config) as f:
            legacy = json.load(f)

        # Register local storage
        self._register_local()

        # Register NAS from legacy config
        nas = legacy.get("nas", {})
        if nas:
            profile = StorageProfile(
                allocation_tb=nas.get("allocation_tb", 10),
                store_checkpoints=True,
                store_models=True,
                store_backups=True,
                store_snapshots=True,
            )

            self.register_stronghold(
                name="synology_citadel",
                stronghold_type=StrongholdType.NAS,
                description="Synology NAS for remote archival storage",
                host=nas.get("host", ""),
                share=nas.get("share", "data"),
                base_path=nas.get("base_path", "llm_training"),
                profile=profile,
                credentials_key="synology",
            )

        self._save_registry()

    def _register_local(self):
        """Register local storage as the primary stronghold."""
        import shutil
        usage = shutil.disk_usage(self.base_dir)

        local = Stronghold(
            name="local_vault",
            stronghold_type=StrongholdType.LOCAL,
            description="Primary local storage",
            base_path=str(self.base_dir),
            status=StrongholdStatus.ACTIVE,
            profile=StorageProfile(
                allocation_tb=usage.total / (1024 ** 4),
                sync_schedule=SyncSchedule.ON_SAVE,
            ),
            used_gb=usage.used / (1024 ** 3),
            free_gb=usage.free / (1024 ** 3),
            total_gb=usage.total / (1024 ** 3),
            last_checked=datetime.now(),
        )
        self._strongholds["local_vault"] = local

    def _save_registry(self):
        """Save registry to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "strongholds": [sh.to_dict() for sh in self._strongholds.values()],
        }

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_stronghold(
        self,
        name: str,
        stronghold_type: StrongholdType,
        description: str = "",
        host: Optional[str] = None,
        share: Optional[str] = None,
        base_path: str = "",
        mount_point: Optional[str] = None,
        profile: Optional[StorageProfile] = None,
        credentials_key: Optional[str] = None,
    ) -> Stronghold:
        """
        Register a new storage stronghold.

        Args:
            name: Unique identifier for the stronghold
            stronghold_type: Type of storage (LOCAL, NAS, CLOUD, REMOTE)
            description: Human-readable description
            host: Host address (for network storage)
            share: Share name (for NAS)
            base_path: Base path within the storage
            mount_point: Local mount point
            profile: Storage profile configuration
            credentials_key: Key for looking up credentials

        Returns:
            The registered Stronghold
        """
        stronghold = Stronghold(
            name=name,
            stronghold_type=stronghold_type,
            description=description,
            host=host,
            share=share,
            base_path=base_path,
            mount_point=mount_point,
            profile=profile or StorageProfile(),
            credentials_key=credentials_key,
            status=StrongholdStatus.UNKNOWN,
        )

        self._strongholds[name] = stronghold
        self._save_registry()

        return stronghold

    def unregister_stronghold(self, name: str) -> bool:
        """
        Unregister a stronghold.

        Args:
            name: Stronghold name

        Returns:
            True if removed
        """
        if name in self._strongholds:
            del self._strongholds[name]
            self._save_registry()
            return True
        return False

    def update_profile(self, name: str, profile: StorageProfile) -> bool:
        """
        Update a stronghold's profile.

        Args:
            name: Stronghold name
            profile: New profile

        Returns:
            True if updated
        """
        if name in self._strongholds:
            self._strongholds[name].profile = profile
            self._save_registry()
            return True
        return False

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_stronghold(self, name: str) -> Optional[Stronghold]:
        """Get a stronghold by name."""
        return self._strongholds.get(name)

    def list_strongholds(
        self,
        stronghold_type: Optional[StrongholdType] = None,
        status: Optional[StrongholdStatus] = None,
    ) -> List[Stronghold]:
        """
        List registered strongholds.

        Args:
            stronghold_type: Filter by type
            status: Filter by status

        Returns:
            List of Stronghold
        """
        result = list(self._strongholds.values())

        if stronghold_type:
            result = [s for s in result if s.stronghold_type == stronghold_type]

        if status:
            result = [s for s in result if s.status == status]

        return result

    def get_active_strongholds(self) -> List[Stronghold]:
        """Get all active (available) strongholds."""
        return self.list_strongholds(status=StrongholdStatus.ACTIVE)

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def recommend_storage(
        self,
        treasure_type: str,
        size_gb: float,
    ) -> List[Dict[str, Any]]:
        """
        Recommend storage locations for a treasure.

        Args:
            treasure_type: Type of treasure (checkpoint, model, backup, etc.)
            size_gb: Size in GB

        Returns:
            List of recommendations with stronghold and reason
        """
        recommendations = []

        for sh in self.get_active_strongholds():
            # Check if profile allows this type
            profile = sh.profile
            allowed = False

            if treasure_type == "checkpoint" and profile.store_checkpoints:
                allowed = True
            elif treasure_type == "model" and profile.store_models:
                allowed = True
            elif treasure_type == "backup" and profile.store_backups:
                allowed = True
            elif treasure_type == "log" and profile.store_logs:
                allowed = True
            elif treasure_type == "snapshot" and profile.store_snapshots:
                allowed = True

            if not allowed:
                continue

            # Check space
            if sh.free_gb >= size_gb:
                priority = 1 if sh.stronghold_type == StrongholdType.LOCAL else 2

                recommendations.append({
                    "stronghold": sh.name,
                    "priority": priority,
                    "reason": f"Has {sh.free_gb:.1f}GB free",
                    "sync_schedule": sh.profile.sync_schedule.value,
                })

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"])

        return recommendations

    # =========================================================================
    # STATUS
    # =========================================================================

    def check_stronghold_status(self, name: str) -> StrongholdStatus:
        """
        Check and update status of a stronghold.

        Args:
            name: Stronghold name

        Returns:
            Current status
        """
        sh = self._strongholds.get(name)
        if not sh:
            return StrongholdStatus.UNKNOWN

        if sh.stronghold_type == StrongholdType.LOCAL:
            # Local is always active if base_path exists
            if Path(sh.base_path).exists():
                sh.status = StrongholdStatus.ACTIVE
                import shutil
                usage = shutil.disk_usage(sh.base_path)
                sh.used_gb = usage.used / (1024 ** 3)
                sh.free_gb = usage.free / (1024 ** 3)
                sh.total_gb = usage.total / (1024 ** 3)
            else:
                sh.status = StrongholdStatus.OFFLINE

        elif sh.stronghold_type == StrongholdType.NAS:
            # Check if mount point is accessible or host is reachable
            import subprocess
            if sh.mount_point and Path(sh.mount_point).exists():
                sh.status = StrongholdStatus.ACTIVE
            elif sh.host:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", sh.host],
                    capture_output=True
                )
                if result.returncode == 0:
                    sh.status = StrongholdStatus.ACTIVE
                else:
                    sh.status = StrongholdStatus.OFFLINE
            else:
                sh.status = StrongholdStatus.UNKNOWN

        sh.last_checked = datetime.now()
        self._save_registry()

        return sh.status

    def check_all_status(self) -> Dict[str, StrongholdStatus]:
        """Check status of all strongholds."""
        return {
            name: self.check_stronghold_status(name)
            for name in self._strongholds
        }

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of the registry."""
        strongholds = list(self._strongholds.values())

        total_allocated = sum(sh.profile.allocation_tb for sh in strongholds)
        total_free = sum(sh.free_gb for sh in strongholds)
        active_count = len([s for s in strongholds if s.status == StrongholdStatus.ACTIVE])

        return {
            "total_strongholds": len(strongholds),
            "active_strongholds": active_count,
            "total_allocated_tb": round(total_allocated, 2),
            "total_free_gb": round(total_free, 1),
            "strongholds": [
                {
                    "name": sh.name,
                    "type": sh.stronghold_type.value,
                    "status": sh.status.value,
                    "free_gb": round(sh.free_gb, 1),
                }
                for sh in strongholds
            ],
        }

    # =========================================================================
    # SYNC OPERATIONS
    # =========================================================================

    def sync_treasure(
        self,
        source_path: str | Path,
        stronghold_name: str,
        treasure_type: str = "checkpoint",
        delete_source: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync a treasure to a stronghold.

        Args:
            source_path: Path to treasure to sync
            stronghold_name: Target stronghold name
            treasure_type: Type of treasure (checkpoint, model, backup, etc.)
            delete_source: If True, delete source after sync

        Returns:
            Dict with sync result
        """
        import subprocess
        import shutil

        source_path = Path(source_path)
        sh = self._strongholds.get(stronghold_name)

        if not sh:
            return {"success": False, "error": f"Stronghold not found: {stronghold_name}"}

        if not source_path.exists():
            return {"success": False, "error": f"Source not found: {source_path}"}

        # Check stronghold status
        self.check_stronghold_status(stronghold_name)
        if sh.status != StrongholdStatus.ACTIVE:
            return {"success": False, "error": f"Stronghold not active: {sh.status.value}"}

        # Build destination path
        if sh.stronghold_type == StrongholdType.LOCAL:
            dest_base = Path(sh.base_path)
        elif sh.mount_point:
            dest_base = Path(sh.mount_point) / sh.base_path
        else:
            # Need to use rsync over SSH or similar
            dest_base = None

        result = {
            "success": False,
            "source": str(source_path),
            "stronghold": stronghold_name,
            "treasure_type": treasure_type,
        }

        try:
            if sh.stronghold_type == StrongholdType.LOCAL and dest_base:
                # Local copy
                dest_path = dest_base / treasure_type / source_path.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, dest_path)

                result["success"] = True
                result["destination"] = str(dest_path)

            elif sh.stronghold_type == StrongholdType.NAS:
                # Use rsync for NAS
                if sh.mount_point and Path(sh.mount_point).exists():
                    # Mounted NAS - local copy
                    dest_path = Path(sh.mount_point) / sh.base_path / treasure_type / source_path.name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    if source_path.is_dir():
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_path, dest_path)

                    result["success"] = True
                    result["destination"] = str(dest_path)
                else:
                    # Use rsync over network
                    dest = f"{sh.host}:{sh.share}/{sh.base_path}/{treasure_type}/"
                    cmd = ["rsync", "-avz", "--progress", str(source_path), dest]

                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    result["success"] = proc.returncode == 0
                    result["destination"] = dest
                    if not result["success"]:
                        result["error"] = proc.stderr

            # Delete source if requested and sync succeeded
            if result["success"] and delete_source:
                if source_path.is_dir():
                    shutil.rmtree(source_path)
                else:
                    source_path.unlink()
                result["source_deleted"] = True

            # Update last sync time
            sh.last_sync = datetime.now()
            self._save_registry()

        except Exception as e:
            result["error"] = str(e)

        return result

    def sync_to_all(
        self,
        source_path: str | Path,
        treasure_type: str = "checkpoint",
        stronghold_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Sync a treasure to all eligible strongholds.

        Args:
            source_path: Path to treasure to sync
            treasure_type: Type of treasure
            stronghold_filter: Only sync to these strongholds (None = all)

        Returns:
            Dict with results per stronghold
        """
        results = {}

        for sh in self.get_active_strongholds():
            # Skip local (source is already there)
            if sh.stronghold_type == StrongholdType.LOCAL:
                continue

            # Apply filter
            if stronghold_filter and sh.name not in stronghold_filter:
                continue

            # Check if profile allows this type
            profile = sh.profile
            allowed = False
            if treasure_type == "checkpoint" and profile.store_checkpoints:
                allowed = True
            elif treasure_type == "model" and profile.store_models:
                allowed = True
            elif treasure_type == "backup" and profile.store_backups:
                allowed = True

            if allowed:
                results[sh.name] = self.sync_treasure(
                    source_path, sh.name, treasure_type
                )

        return {
            "synced_to": len([r for r in results.values() if r.get("success")]),
            "failed": len([r for r in results.values() if not r.get("success")]),
            "results": results,
        }

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get sync status for all strongholds.

        Returns:
            Dict with last sync times and status
        """
        status = []
        for sh in self._strongholds.values():
            sync_age = None
            if sh.last_sync:
                sync_age = (datetime.now() - sh.last_sync).total_seconds() / 3600

            status.append({
                "name": sh.name,
                "type": sh.stronghold_type.value,
                "status": sh.status.value,
                "last_sync": sh.last_sync.isoformat() if sh.last_sync else None,
                "sync_age_hours": round(sync_age, 1) if sync_age else None,
                "schedule": sh.profile.sync_schedule.value,
            })

        return {"strongholds": status}

    def needs_sync(self, stronghold_name: str) -> bool:
        """
        Check if a stronghold needs syncing based on schedule.

        Args:
            stronghold_name: Stronghold to check

        Returns:
            True if sync is due
        """
        sh = self._strongholds.get(stronghold_name)
        if not sh:
            return False

        if sh.profile.sync_schedule == SyncSchedule.MANUAL:
            return False

        if not sh.last_sync:
            return True

        age_hours = (datetime.now() - sh.last_sync).total_seconds() / 3600

        schedule_hours = {
            SyncSchedule.ON_SAVE: 0,
            SyncSchedule.ON_DEPLOY: 0,
            SyncSchedule.HOURLY: 1,
            SyncSchedule.DAILY: 24,
            SyncSchedule.WEEKLY: 168,
        }

        threshold = schedule_hours.get(sh.profile.sync_schedule, 24)
        return age_hours >= threshold


# Convenience function
def get_storage_registry(base_dir: Optional[str | Path] = None) -> StorageRegistry:
    """Get a StorageRegistry instance."""
    return StorageRegistry(base_dir)
