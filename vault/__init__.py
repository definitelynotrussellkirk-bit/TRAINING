"""
Vault - Secure storage for the realm's treasures.

The Vault lies deep beneath the training citadel, a secure repository
for all precious items:

    VaultKeeper     - Central asset registry (knows where everything is)
    Archivist       - Backup management and verification
    Chronicle       - Model version history and lineage
    Treasury        - Resource management and retention
    StorageRegistry - Register and manage storage locations (strongholds)

RPG Mapping:
    VaultKeeper         → The Keeper (knows all, sees all)
    BackupManager       → Archivist (seals backups)
    ModelVersioner      → Chronicle (records history)
    RetentionManager    → Treasury (manages resources)
    StorageConfig       → StorageRegistry (manages strongholds)
    Checkpoint          → Treasure
    Backup              → Archive
    Version             → Chapter
    Storage Location    → Stronghold

NEW: VaultKeeper - Ask Vault First Pattern
==========================================
Before loading ANY asset, ask the VaultKeeper first:

    from vault import ask_vault_first, VaultKeeper
    from core.paths import get_base_dir

    # Ask vault where checkpoint is (may be local, NAS, or 3090)
    base = get_base_dir()
    checkpoint_path = ask_vault_first(
        "checkpoint_175000",
        fallback=str(base / "models" / "checkpoint-175000")
    )

    # The keeper will:
    # 1. Look up the asset in its catalog
    # 2. Find the best available location
    # 3. Fetch it locally if needed
    # 4. Return the local path

    # Or use the VaultKeeper directly:
    keeper = VaultKeeper()

    # Register assets
    keeper.register_from_path("/path/to/checkpoint-175000")

    # Locate
    result = keeper.locate("checkpoint_175000")
    print(f"Found at: {result.best_location.path}")

    # Fetch from best source
    result = keeper.fetch("checkpoint_175000", "/tmp/local_copy")

API Server (for remote devices like 3090):
    python3 vault/server.py --port 8767

    # Query from 3090:
    from core.hosts import get_service_url
    keeper_url = get_service_url("vault")
    curl {keeper_url}/api/locate/checkpoint_175000

Client Library (for remote queries):
    from vault.client import VaultKeeperClient
    from core.hosts import get_service_url

    keeper_url = get_service_url("vault")
    client = VaultKeeperClient(keeper_url)
    result = client.locate("checkpoint_175000")

Discovery Service (scan and register assets):
    from vault import VaultDiscovery

    discovery = VaultDiscovery()
    discovery.scan_all()  # Register all local assets

This module wraps management/ with RPG-themed naming while maintaining
backward compatibility.
"""

__version__ = "0.2.0"

# Types
from vault.types import (
    # Treasure types
    TreasureType,
    ProtectionLevel,
    VaultRecord,
    # Archive types
    ArchiveType,
    ArchiveEntry,
    # Chronicle types
    ChronicleEntry,
    # Treasury types
    RetentionRule,
    TreasuryStatus,
)

# Asset Types (for VaultKeeper)
from vault.assets import (
    Asset,
    AssetType,
    AssetStatus,
    AssetLocation,
    LocationStatus,
    AssetQuery,
    asset_from_path,
    generate_asset_id,
)

# Location Handlers
from vault.handlers import (
    LocationHandler,
    LocalHandler,
    RemoteHandler,
    NASHandler,
    PathInfo,
    TransferResult,
    get_handler,
)

# VaultKeeper (central registry)
from vault.keeper import (
    VaultKeeper,
    LookupResult,
    FetchResult,
    get_vault_keeper,
    locate,
    fetch,
)

# VaultKeeper Client (for remote queries)
from vault.client import (
    VaultKeeperClient,
    LocateResult as ClientLocateResult,
    FetchResult as ClientFetchResult,
    AssetInfo,
    LocationInfo,
)

# Discovery Service
from vault.discovery import (
    VaultDiscovery,
    ask_vault_first,
    ensure_asset,
    register_and_locate,
)

# Storage Registry (strongholds)
from vault.storage_registry import (
    # Types
    StrongholdType,
    StrongholdStatus,
    SyncSchedule,
    StorageProfile,
    Stronghold,
    # Registry
    StorageRegistry,
    get_storage_registry,
)

# Archivist (backup management)
from vault.archivist import (
    Archivist,
    get_archivist,
    # Backward compat
    BackupManager,
)

# Chronicle (version history)
from vault.chronicle import (
    Chronicle,
    get_chronicle,
    # Backward compat
    ModelVersioner,
)

# Treasury (resource management)
from vault.treasury import (
    Treasury,
    get_treasury,
)

# Zone Federation (Branch Officers)
from vault.zones import (
    # Types
    Zone,
    ZoneType,
    ZoneStatus,
    # Client
    ZoneClient,
    ZoneTransfer,
    # Registry
    ZoneRegistry,
    get_zone_registry,
    get_zone,
    get_zone_client,
    check_zones,
    push_to_zone,
    pull_from_zone,
)

# Storage Resolver (NEW - handle to path resolution)
from vault.storage_resolver import (
    # Core
    StorageResolver,
    get_resolver,
    # Convenience functions
    ask_storage,
    ask_storage_handle,
    locate_storage,
    storage_exists,
    # Common paths
    get_checkpoint_path,
    get_snapshot_path,
    get_dataset_path,
    get_queue_path,
    get_current_model_path,
    get_status_path,
    get_log_path,
    # Errors
    StorageResolverError,
    ZoneNotAvailable,
    KindNotConfigured,
)

# Device Mapping (NEW - bridge between Ledger device_ids and VaultKeeper strongholds)
from vault.device_mapping import (
    # Core
    DeviceMapping,
    DeviceInfo,
    get_mapping,
    # Convenience functions
    device_to_stronghold,
    stronghold_to_device,
    get_local_device_id,
    get_local_stronghold,
)


__all__ = [
    # VaultKeeper - Central Registry
    "VaultKeeper",
    "get_vault_keeper",
    "LookupResult",
    "FetchResult",
    "locate",
    "fetch",
    # VaultKeeper Client
    "VaultKeeperClient",
    "ClientLocateResult",
    "ClientFetchResult",
    "AssetInfo",
    "LocationInfo",
    # Asset Types
    "Asset",
    "AssetType",
    "AssetStatus",
    "AssetLocation",
    "LocationStatus",
    "AssetQuery",
    "asset_from_path",
    "generate_asset_id",
    # Location Handlers
    "LocationHandler",
    "LocalHandler",
    "RemoteHandler",
    "NASHandler",
    "PathInfo",
    "TransferResult",
    "get_handler",
    # Discovery
    "VaultDiscovery",
    "ask_vault_first",
    "ensure_asset",
    "register_and_locate",
    # Types - Treasure
    "TreasureType",
    "ProtectionLevel",
    "VaultRecord",
    # Types - Archive
    "ArchiveType",
    "ArchiveEntry",
    # Types - Chronicle
    "ChronicleEntry",
    # Types - Treasury
    "RetentionRule",
    "TreasuryStatus",
    # Storage Registry - Types
    "StrongholdType",
    "StrongholdStatus",
    "SyncSchedule",
    "StorageProfile",
    "Stronghold",
    # Storage Registry
    "StorageRegistry",
    "get_storage_registry",
    # Archivist
    "Archivist",
    "get_archivist",
    "BackupManager",  # Backward compat
    # Chronicle
    "Chronicle",
    "get_chronicle",
    "ModelVersioner",  # Backward compat
    # Treasury
    "Treasury",
    "get_treasury",
    # Zone Federation
    "Zone",
    "ZoneType",
    "ZoneStatus",
    "ZoneClient",
    "ZoneTransfer",
    "ZoneRegistry",
    "get_zone_registry",
    "get_zone",
    "get_zone_client",
    "check_zones",
    "push_to_zone",
    "pull_from_zone",
    # Storage Resolver
    "StorageResolver",
    "get_resolver",
    "ask_storage",
    "ask_storage_handle",
    "locate_storage",
    "storage_exists",
    "get_checkpoint_path",
    "get_snapshot_path",
    "get_dataset_path",
    "get_queue_path",
    "get_current_model_path",
    "get_status_path",
    "get_log_path",
    "StorageResolverError",
    "ZoneNotAvailable",
    "KindNotConfigured",
    # Device Mapping
    "DeviceMapping",
    "DeviceInfo",
    "get_mapping",
    "device_to_stronghold",
    "stronghold_to_device",
    "get_local_device_id",
    "get_local_stronghold",
]


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
VAULT GLOSSARY
==============

VAULTKEEPER TERMS
-----------------
VaultKeeper     = Central asset registry (the Keeper)
Catalog         = Database of all known assets (the Great Ledger)
Locate          = Find where an asset exists
Fetch           = Retrieve an asset from any location
Register        = Add an asset to the catalog
Discovery       = Scan and auto-register assets
Ask Vault First = Check keeper before loading anything

STRONGHOLD TERMS
----------------
Stronghold      = Storage location (local, NAS, cloud)
Register        = Add storage to the registry
Profile         = Configuration for a stronghold
Allocation      = Space allocated to stronghold
Sync Schedule   = When to sync (on_save, daily, etc.)

STRONGHOLD TYPES
----------------
LOCAL           = Local filesystem
NAS             = Network Attached Storage (Synology, etc.)
CLOUD           = Cloud storage (S3, GCS, etc.)
REMOTE          = Remote server (SSH)

ASSET TYPES
-----------
CHECKPOINT      = Training checkpoint
MODEL           = Full/consolidated model
BASE_MODEL      = Original base model (sacred)
TRAINING_DATA   = JSONL training files
VALIDATION_DATA = Validation datasets
CONFIG          = Configuration files
SNAPSHOT        = Daily snapshots
EVALUATION      = Evaluation results
BACKUP          = Backup archives

TREASURE TERMS
--------------
Treasure        = Stored item (checkpoint, backup, model)
Sacred          = Never delete (base model)
Protected       = Keep unless manually removed (latest, best)
Guarded         = Keep for retention period (today, yesterday)
Expendable      = Can be deleted when space needed

ARCHIVIST TERMS
---------------
Archivist       = Backup manager
Seal            = Create verified backup
Archive         = Backup copy
Verify Seal     = Check backup integrity
Restore         = Recover from backup

CHRONICLE TERMS
---------------
Chronicle       = Version history database
Chapter         = Version entry
Record Chapter  = Create new version
Lineage         = Evolution tree
Milestone       = Significant achievement

TREASURY TERMS
--------------
Treasury        = Resource manager
Vault Capacity  = Disk space
Inventory       = List all stored items
Enforce         = Apply retention policy
Emergency       = Critical cleanup mode
"""
