"""
Storage Types - Logical references for storage locations.

This module defines the type system for storage:
- StorageZone: Temperature-based storage tiers (hot/warm/cold)
- StorageKind: What type of thing is being stored (checkpoint, dataset, etc.)
- StorageHandle: A logical reference to a stored item

Usage:
    from core.storage_types import StorageZone, StorageKind, StorageHandle

    # Create a handle for a checkpoint
    handle = StorageHandle(
        kind=StorageKind.CHECKPOINT,
        key="checkpoint-182000",
        zone=StorageZone.HOT
    )

    # Use with StorageResolver to get actual path
    from vault.storage_resolver import get_resolver
    path = get_resolver().resolve(handle)

Design:
    StorageHandle is the "lingua franca" - code talks in handles, not paths.
    The StorageResolver turns handles into actual filesystem paths based on
    which device the code is running on.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# =============================================================================
# STORAGE ZONES (Temperature)
# =============================================================================

class StorageZone(str, Enum):
    """
    Temperature-based storage tiers.

    Each zone has different characteristics:

    HOT:
        - Location: Local NVMe on trainer/inference machines
        - Speed: Fastest (direct disk access)
        - Capacity: Limited (500GB-2TB)
        - Durability: Ephemeral (no redundancy)
        - Use for: current_model, active checkpoints, training queue

    WARM:
        - Location: Primary NAS (Synology)
        - Speed: Fast (10Gbps network)
        - Capacity: Large (10-20TB)
        - Durability: RAID protected
        - Use for: snapshots, canonical datasets, benchmarks

    COLD:
        - Location: Archive NAS or offsite
        - Speed: Slower (may be 1Gbps)
        - Capacity: Very large (20+ TB)
        - Durability: Backed up
        - Use for: compressed archives, old snapshots
    """
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"

    @property
    def description(self) -> str:
        """Human-readable description of this zone."""
        descriptions = {
            StorageZone.HOT: "Fast local storage (NVMe)",
            StorageZone.WARM: "Primary NAS storage",
            StorageZone.COLD: "Archive storage",
        }
        return descriptions.get(self, "Unknown zone")

    @property
    def is_local(self) -> bool:
        """Whether this zone is typically local storage."""
        return self == StorageZone.HOT

    @property
    def is_networked(self) -> bool:
        """Whether this zone is typically networked storage."""
        return self in (StorageZone.WARM, StorageZone.COLD)


# =============================================================================
# STORAGE KINDS (What type of thing)
# =============================================================================

class StorageKind(str, Enum):
    """
    Types of stored items.

    Each kind has a default zone and subdirectory pattern.
    """
    # Models
    BASE_MODEL = "base_model"           # Pretrained models (Qwen3-0.6B)
    CURRENT_MODEL = "current_model"     # Active training directory
    CHECKPOINT = "checkpoint"           # HF checkpoint-XXXX directories
    SNAPSHOT = "snapshot"               # Promoted/blessed checkpoints

    # Data
    DATASET = "dataset"                 # Training data files
    BENCHMARK = "benchmark"             # Evaluation benchmarks
    VALIDATION = "validation"           # Validation sets

    # Queue/Operations
    QUEUE = "queue"                     # Training queue directories
    INBOX = "inbox"                     # Inbox for new training files

    # Metadata
    LOG = "log"                         # Log files
    STATUS = "status"                   # Status JSON files
    META = "meta"                       # Guild states, configs
    LEDGER = "ledger"                   # Checkpoint ledger

    # Archives
    BACKUP = "backup"                   # Backup archives
    ARCHIVE = "archive"                 # Compressed archives

    @property
    def default_zone(self) -> StorageZone:
        """Default storage zone for this kind."""
        zone_map = {
            # HOT: Fast access, current work
            StorageKind.CURRENT_MODEL: StorageZone.HOT,
            StorageKind.CHECKPOINT: StorageZone.HOT,
            StorageKind.QUEUE: StorageZone.HOT,
            StorageKind.INBOX: StorageZone.HOT,
            StorageKind.LOG: StorageZone.HOT,
            StorageKind.STATUS: StorageZone.HOT,
            StorageKind.LEDGER: StorageZone.HOT,

            # WARM: Durable, frequently accessed
            StorageKind.BASE_MODEL: StorageZone.WARM,
            StorageKind.SNAPSHOT: StorageZone.WARM,
            StorageKind.DATASET: StorageZone.WARM,
            StorageKind.BENCHMARK: StorageZone.WARM,
            StorageKind.VALIDATION: StorageZone.WARM,
            StorageKind.META: StorageZone.WARM,

            # COLD: Archive, rarely accessed
            StorageKind.BACKUP: StorageZone.COLD,
            StorageKind.ARCHIVE: StorageZone.COLD,
        }
        return zone_map.get(self, StorageZone.WARM)

    @property
    def is_model_related(self) -> bool:
        """Whether this kind is model-related."""
        return self in {
            StorageKind.BASE_MODEL,
            StorageKind.CURRENT_MODEL,
            StorageKind.CHECKPOINT,
            StorageKind.SNAPSHOT,
        }

    @property
    def is_data_related(self) -> bool:
        """Whether this kind is data-related."""
        return self in {
            StorageKind.DATASET,
            StorageKind.BENCHMARK,
            StorageKind.VALIDATION,
        }


# =============================================================================
# STORAGE HANDLE
# =============================================================================

@dataclass(frozen=True)
class StorageHandle:
    """
    A logical reference to a stored item.

    StorageHandle is immutable and hashable, making it safe to use as
    dictionary keys or in sets.

    Attributes:
        kind: What type of thing (checkpoint, dataset, etc.)
        key: Unique identifier within the kind (e.g., "checkpoint-182000")
        zone: Which storage zone (hot, warm, cold)
        metadata: Optional extra info (not used in equality)

    Usage:
        # Create a handle
        handle = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="checkpoint-182000",
            zone=StorageZone.HOT
        )

        # Get string ID
        print(handle.handle_id)  # "checkpoint:checkpoint-182000@hot"

        # Create with default zone
        handle = StorageHandle.for_kind(StorageKind.CHECKPOINT, "checkpoint-182000")
    """
    kind: StorageKind
    key: str
    zone: StorageZone

    def __post_init__(self):
        """Validate handle on creation."""
        # Some kinds don't need a key (e.g., CURRENT_MODEL, INBOX)
        kinds_without_key = {StorageKind.CURRENT_MODEL, StorageKind.INBOX}
        if not self.key and self.kind not in kinds_without_key:
            raise ValueError(f"StorageHandle key cannot be empty for kind {self.kind.value}")
        if not isinstance(self.kind, StorageKind):
            raise TypeError(f"kind must be StorageKind, got {type(self.kind)}")
        if not isinstance(self.zone, StorageZone):
            raise TypeError(f"zone must be StorageZone, got {type(self.zone)}")

    @property
    def handle_id(self) -> str:
        """
        Unique string identifier for this handle.

        Format: "{kind}:{key}@{zone}"
        Example: "checkpoint:checkpoint-182000@hot"
        """
        return f"{self.kind.value}:{self.key}@{self.zone.value}"

    @classmethod
    def for_kind(cls, kind: StorageKind, key: str) -> "StorageHandle":
        """
        Create handle with default zone for the kind.

        Args:
            kind: Storage kind
            key: Unique key

        Returns:
            StorageHandle with appropriate default zone
        """
        return cls(kind=kind, key=key, zone=kind.default_zone)

    @classmethod
    def parse(cls, handle_id: str) -> "StorageHandle":
        """
        Parse a handle ID string back into a StorageHandle.

        Args:
            handle_id: String like "checkpoint:checkpoint-182000@hot"

        Returns:
            StorageHandle instance

        Raises:
            ValueError: If format is invalid
        """
        try:
            # Split into kind:key@zone
            kind_key, zone_str = handle_id.rsplit("@", 1)
            kind_str, key = kind_key.split(":", 1)

            kind = StorageKind(kind_str)
            zone = StorageZone(zone_str)

            return cls(kind=kind, key=key, zone=zone)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid handle ID format: {handle_id}") from e

    def with_zone(self, zone: StorageZone) -> "StorageHandle":
        """
        Create new handle with different zone.

        Useful for moving items between zones:
            warm_handle = hot_handle.with_zone(StorageZone.WARM)
        """
        return StorageHandle(kind=self.kind, key=self.key, zone=zone)

    def __str__(self) -> str:
        return self.handle_id

    def __repr__(self) -> str:
        return f"StorageHandle({self.kind.value!r}, {self.key!r}, {self.zone.value!r})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def checkpoint_handle(key: str, zone: Optional[StorageZone] = None) -> StorageHandle:
    """Create a handle for a checkpoint."""
    return StorageHandle(
        kind=StorageKind.CHECKPOINT,
        key=key,
        zone=zone or StorageZone.HOT,
    )


def snapshot_handle(key: str, zone: Optional[StorageZone] = None) -> StorageHandle:
    """Create a handle for a snapshot."""
    return StorageHandle(
        kind=StorageKind.SNAPSHOT,
        key=key,
        zone=zone or StorageZone.WARM,
    )


def dataset_handle(key: str, zone: Optional[StorageZone] = None) -> StorageHandle:
    """Create a handle for a dataset."""
    return StorageHandle(
        kind=StorageKind.DATASET,
        key=key,
        zone=zone or StorageZone.WARM,
    )


def queue_handle(priority: str = "normal") -> StorageHandle:
    """Create a handle for a queue directory."""
    return StorageHandle(
        kind=StorageKind.QUEUE,
        key=priority,
        zone=StorageZone.HOT,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("Storage Types Demo")
    print("=" * 50)

    # Show zones
    print("\nStorage Zones:")
    for zone in StorageZone:
        print(f"  {zone.value}: {zone.description}")
        print(f"    local={zone.is_local}, networked={zone.is_networked}")

    # Show kinds
    print("\nStorage Kinds:")
    for kind in StorageKind:
        print(f"  {kind.value}: default_zone={kind.default_zone.value}")

    # Demo handles
    print("\nHandle Examples:")

    h1 = StorageHandle.for_kind(StorageKind.CHECKPOINT, "checkpoint-182000")
    print(f"  {h1}")
    print(f"    handle_id: {h1.handle_id}")

    h2 = snapshot_handle("checkpoint-180000-20251128-1200")
    print(f"  {h2}")

    h3 = dataset_handle("binary_l5_v2")
    print(f"  {h3}")

    # Parse demo
    print("\nParse Demo:")
    parsed = StorageHandle.parse("checkpoint:checkpoint-182000@hot")
    print(f"  Parsed: {parsed}")
    print(f"  kind={parsed.kind}, key={parsed.key}, zone={parsed.zone}")

    # Zone change demo
    print("\nZone Change Demo:")
    cold_handle = h1.with_zone(StorageZone.COLD)
    print(f"  Original: {h1}")
    print(f"  To cold:  {cold_handle}")
