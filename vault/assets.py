"""
Vault Assets - Asset types and schemas for the VaultKeeper.

Assets are any valuable items tracked by the VaultKeeper:
    - Checkpoints (training checkpoints)
    - Models (full models, base models)
    - Training Data (JSONL files)
    - Configs (configuration files)
    - Snapshots (daily snapshots)
    - Evaluations (eval results)

Each asset has:
    - A unique ID (hash-based or name-based)
    - A type (checkpoint, model, data, etc.)
    - Metadata (size, created, step, etc.)
    - One or more locations (where copies exist)

RPG Flavor:
    Assets are the treasures of the realm - magical artifacts (models),
    ancient scrolls (training data), and powerful relics (checkpoints).
    The VaultKeeper tracks them all across every stronghold.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AssetType(Enum):
    """Types of assets tracked by the VaultKeeper."""
    CHECKPOINT = "checkpoint"      # Training checkpoint
    MODEL = "model"                # Full/consolidated model
    BASE_MODEL = "base_model"      # Original base model (sacred)
    TRAINING_DATA = "training_data"  # JSONL training files
    VALIDATION_DATA = "validation_data"  # Validation datasets
    CONFIG = "config"              # Configuration files
    SNAPSHOT = "snapshot"          # Daily snapshots
    EVALUATION = "evaluation"      # Evaluation results
    BACKUP = "backup"              # Backup archives
    LOG = "log"                    # Log files


class AssetStatus(Enum):
    """Status of an asset."""
    ACTIVE = "active"              # Currently in use
    ARCHIVED = "archived"          # Archived but available
    DEPRECATED = "deprecated"      # Old version, may be deleted
    MISSING = "missing"            # Expected but not found
    CORRUPTED = "corrupted"        # Failed verification


class LocationStatus(Enum):
    """Status of an asset at a specific location."""
    VERIFIED = "verified"          # Exists and verified
    UNVERIFIED = "unverified"      # Exists but not recently verified
    MISSING = "missing"            # Should be here but isn't
    SYNCING = "syncing"            # Currently being synced
    CORRUPTED = "corrupted"        # Verification failed


@dataclass
class AssetLocation:
    """
    A location where an asset exists.

    An asset can exist in multiple locations (strongholds).
    This tracks each copy's status and path.
    """
    stronghold: str                # Stronghold name (e.g., "local_vault", "synology_citadel")
    path: str                      # Path within the stronghold
    status: LocationStatus = LocationStatus.UNVERIFIED

    # Verification
    verified_at: Optional[datetime] = None
    checksum: Optional[str] = None
    size_bytes: int = 0

    # Sync info
    synced_at: Optional[datetime] = None
    is_primary: bool = False       # Is this the primary/authoritative copy?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stronghold": self.stronghold,
            "path": self.path,
            "status": self.status.value,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
            "is_primary": self.is_primary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetLocation":
        return cls(
            stronghold=data["stronghold"],
            path=data["path"],
            status=LocationStatus(data.get("status", "unverified")),
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
            synced_at=datetime.fromisoformat(data["synced_at"]) if data.get("synced_at") else None,
            is_primary=data.get("is_primary", False),
        )


@dataclass
class Asset:
    """
    A tracked asset in the VaultKeeper.

    Assets are identified by a unique ID and can exist in multiple
    locations across the realm's strongholds.
    """
    asset_id: str                  # Unique identifier
    asset_type: AssetType
    name: str                      # Human-readable name

    # Metadata
    description: str = ""
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    # Status
    status: AssetStatus = AssetStatus.ACTIVE

    # Locations (where copies exist)
    locations: List[AssetLocation] = field(default_factory=list)

    # Asset-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For checkpoints
    @property
    def step_number(self) -> Optional[int]:
        return self.metadata.get("step_number")

    @step_number.setter
    def step_number(self, value: int):
        self.metadata["step_number"] = value

    # For models
    @property
    def base_model(self) -> Optional[str]:
        return self.metadata.get("base_model")

    @base_model.setter
    def base_model(self, value: str):
        self.metadata["base_model"] = value

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def primary_location(self) -> Optional[AssetLocation]:
        """Get the primary location for this asset."""
        for loc in self.locations:
            if loc.is_primary:
                return loc
        # Fall back to first verified location
        for loc in self.locations:
            if loc.status == LocationStatus.VERIFIED:
                return loc
        # Fall back to any location
        return self.locations[0] if self.locations else None

    @property
    def available_locations(self) -> List[AssetLocation]:
        """Get all locations where asset is available."""
        return [
            loc for loc in self.locations
            if loc.status in (LocationStatus.VERIFIED, LocationStatus.UNVERIFIED)
        ]

    def add_location(self, location: AssetLocation):
        """Add a location for this asset."""
        # Check if location already exists
        for i, loc in enumerate(self.locations):
            if loc.stronghold == location.stronghold and loc.path == location.path:
                self.locations[i] = location
                return
        self.locations.append(location)

    def remove_location(self, stronghold: str, path: str):
        """Remove a location from this asset."""
        self.locations = [
            loc for loc in self.locations
            if not (loc.stronghold == stronghold and loc.path == path)
        ]

    def get_location(self, stronghold: str) -> Optional[AssetLocation]:
        """Get location in a specific stronghold."""
        for loc in self.locations:
            if loc.stronghold == stronghold:
                return loc
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "name": self.name,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_gb, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "status": self.status.value,
            "locations": [loc.to_dict() for loc in self.locations],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        return cls(
            asset_id=data["asset_id"],
            asset_type=AssetType(data["asset_type"]),
            name=data["name"],
            description=data.get("description", ""),
            size_bytes=data.get("size_bytes", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            modified_at=datetime.fromisoformat(data["modified_at"]) if data.get("modified_at") else None,
            status=AssetStatus(data.get("status", "active")),
            locations=[AssetLocation.from_dict(loc) for loc in data.get("locations", [])],
            metadata=data.get("metadata", {}),
        )


def generate_asset_id(asset_type: AssetType, name: str, path: Optional[str] = None) -> str:
    """
    Generate a unique asset ID.

    For checkpoints: checkpoint_175000
    For models: model_qwen3_0.6b
    For data: data_<hash of path>
    """
    if asset_type == AssetType.CHECKPOINT:
        # Extract step number if present
        if name.startswith("checkpoint-"):
            return f"checkpoint_{name.replace('checkpoint-', '')}"
        return f"checkpoint_{name}"

    elif asset_type in (AssetType.MODEL, AssetType.BASE_MODEL):
        # Clean model name
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        return f"model_{clean_name}"

    elif asset_type in (AssetType.TRAINING_DATA, AssetType.VALIDATION_DATA):
        # Hash the path for uniqueness
        if path:
            path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
            return f"data_{name}_{path_hash}"
        return f"data_{name}"

    elif asset_type == AssetType.CONFIG:
        return f"config_{name}"

    elif asset_type == AssetType.SNAPSHOT:
        return f"snapshot_{name}"

    elif asset_type == AssetType.EVALUATION:
        return f"eval_{name}"

    elif asset_type == AssetType.BACKUP:
        return f"backup_{name}"

    else:
        # Generic
        return f"{asset_type.value}_{name}"


def asset_from_path(
    path: str | Path,
    stronghold: str = "local_vault",
    is_primary: bool = True,
) -> Asset:
    """
    Create an Asset from a filesystem path.

    Infers asset type from path structure and creates appropriate metadata.
    """
    path = Path(path)

    # Determine asset type from path
    path_str = str(path).lower()
    name = path.name

    if "checkpoint-" in name or "checkpoint_" in name:
        asset_type = AssetType.CHECKPOINT
        # Extract step number (handles canonical names like checkpoint-190000-20251127-1430)
        from core.checkpoint_ledger import extract_step
        step = extract_step(name)
        metadata = {"step_number": step} if step else {}

    elif "models/" in path_str or "model" in name.lower():
        if "base" in path_str or "Qwen" in name or "qwen" in name:
            asset_type = AssetType.BASE_MODEL
        else:
            asset_type = AssetType.MODEL
        metadata = {"base_model": name}

    elif path.suffix == ".jsonl":
        if "validation" in path_str or "val" in name.lower():
            asset_type = AssetType.VALIDATION_DATA
        else:
            asset_type = AssetType.TRAINING_DATA
        metadata = {}

    elif path.suffix == ".json" and "config" in name.lower():
        asset_type = AssetType.CONFIG
        metadata = {}

    elif "snapshot" in path_str:
        asset_type = AssetType.SNAPSHOT
        metadata = {}

    elif "backup" in path_str or path.suffix in (".tar", ".tar.gz", ".zip"):
        asset_type = AssetType.BACKUP
        metadata = {}

    elif "eval" in path_str or "evaluation" in path_str:
        asset_type = AssetType.EVALUATION
        metadata = {}

    else:
        # Default to training data for unknown JSONL, config for JSON
        if path.suffix == ".jsonl":
            asset_type = AssetType.TRAINING_DATA
        elif path.suffix == ".json":
            asset_type = AssetType.CONFIG
        else:
            asset_type = AssetType.MODEL
        metadata = {}

    # Get file stats
    size_bytes = 0
    created_at = None
    modified_at = None

    if path.exists():
        stat = path.stat()
        size_bytes = stat.st_size if path.is_file() else sum(
            f.stat().st_size for f in path.rglob("*") if f.is_file()
        )
        created_at = datetime.fromtimestamp(stat.st_ctime)
        modified_at = datetime.fromtimestamp(stat.st_mtime)

    # Generate ID
    asset_id = generate_asset_id(asset_type, name, str(path))

    # Create location
    location = AssetLocation(
        stronghold=stronghold,
        path=str(path),
        status=LocationStatus.VERIFIED if path.exists() else LocationStatus.MISSING,
        verified_at=datetime.now() if path.exists() else None,
        size_bytes=size_bytes,
        is_primary=is_primary,
    )

    return Asset(
        asset_id=asset_id,
        asset_type=asset_type,
        name=name,
        size_bytes=size_bytes,
        created_at=created_at,
        modified_at=modified_at,
        status=AssetStatus.ACTIVE if path.exists() else AssetStatus.MISSING,
        locations=[location],
        metadata=metadata,
    )


# =============================================================================
# QUERY HELPERS
# =============================================================================

@dataclass
class AssetQuery:
    """Query parameters for searching assets."""
    asset_type: Optional[AssetType] = None
    name_pattern: Optional[str] = None
    stronghold: Optional[str] = None
    status: Optional[AssetStatus] = None
    min_size_bytes: Optional[int] = None
    max_size_bytes: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    metadata_filter: Optional[Dict[str, Any]] = None

    def matches(self, asset: Asset) -> bool:
        """Check if an asset matches this query."""
        if self.asset_type and asset.asset_type != self.asset_type:
            return False

        if self.name_pattern:
            import fnmatch
            if not fnmatch.fnmatch(asset.name.lower(), self.name_pattern.lower()):
                return False

        if self.stronghold:
            if not any(loc.stronghold == self.stronghold for loc in asset.locations):
                return False

        if self.status and asset.status != self.status:
            return False

        if self.min_size_bytes and asset.size_bytes < self.min_size_bytes:
            return False

        if self.max_size_bytes and asset.size_bytes > self.max_size_bytes:
            return False

        if self.created_after and asset.created_at:
            if asset.created_at < self.created_after:
                return False

        if self.created_before and asset.created_at:
            if asset.created_at > self.created_before:
                return False

        if self.metadata_filter:
            for key, value in self.metadata_filter.items():
                if asset.metadata.get(key) != value:
                    return False

        return True
