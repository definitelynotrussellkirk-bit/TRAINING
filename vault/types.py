"""
Vault Types - Core types for the secure storage and management system.

The Vault is the secure heart of the training citadel where treasures
(models, checkpoints, backups) are stored and protected:

    TreasureType    - Type of stored item
    VaultRecord     - Record of a stored item
    ArchiveEntry    - Backup archive entry
    ChronicleEntry  - Version history entry
    RetentionPolicy - Rules for keeping/discarding

RPG Flavor:
    The Vault lies deep beneath the citadel. The Archivist guards the
    backup archives. The Chronicle records all model evolution. The
    Treasury manages precious resources (disk space).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# TREASURE TYPES
# =============================================================================

class TreasureType(Enum):
    """Types of treasures stored in the Vault."""
    CHECKPOINT = "checkpoint"      # Training checkpoint
    SNAPSHOT = "snapshot"          # Daily snapshot
    ARCHIVE = "archive"            # Backup archive
    CONSOLIDATED = "consolidated"  # Consolidated model
    BASE_MODEL = "base_model"      # Original base model


class ProtectionLevel(Enum):
    """Protection level for vault items."""
    SACRED = "sacred"          # Never delete (base model, critical)
    PROTECTED = "protected"    # Keep unless manually removed
    GUARDED = "guarded"        # Keep for retention period
    EXPENDABLE = "expendable"  # Can be deleted when space needed


# =============================================================================
# VAULT RECORDS
# =============================================================================

@dataclass
class VaultRecord:
    """
    Record of an item stored in the Vault.

    Every checkpoint, snapshot, and archive has a VaultRecord
    tracking its location, size, and protection status.
    """
    name: str
    path: str
    treasure_type: TreasureType
    protection: ProtectionLevel = ProtectionLevel.GUARDED

    # Size
    size_bytes: int = 0
    size_human: str = ""

    # Timestamps
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

    # Metadata
    step_number: Optional[int] = None  # For checkpoints
    version_id: Optional[str] = None   # For versioned items
    description: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "treasure_type": self.treasure_type.value,
            "protection": self.protection.value,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_gb, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "step_number": self.step_number,
            "version_id": self.version_id,
            "description": self.description,
        }


# =============================================================================
# ARCHIVIST (Backups)
# =============================================================================

class ArchiveType(Enum):
    """Types of backup archives."""
    PRE_CONSOLIDATION = "pre_consolidation"  # Before model merge
    PRE_DELETION = "pre_deletion"            # Before cleanup
    EMERGENCY = "emergency"                  # Emergency backup
    SCHEDULED = "scheduled"                  # Regular scheduled backup


@dataclass
class ArchiveEntry:
    """
    A backup archive created by the Archivist.
    """
    archive_id: str
    archive_type: ArchiveType
    source_path: str
    archive_path: str

    # Verification
    verified: bool = False
    checksum: Optional[str] = None
    file_count: int = 0

    # Size
    size_bytes: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Metadata
    reason: str = ""
    source_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_id": self.archive_id,
            "archive_type": self.archive_type.value,
            "source_path": self.source_path,
            "archive_path": self.archive_path,
            "verified": self.verified,
            "checksum": self.checksum,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
        }


# =============================================================================
# CHRONICLE (Version History)
# =============================================================================

@dataclass
class ChronicleEntry:
    """
    A version entry in the Chronicle (model history).

    The Chronicle tracks the evolution of the model through training,
    recording each significant version with its metrics and lineage.
    """
    version_id: str              # e.g., "v001", "v002"
    version_number: int
    checkpoint_step: int

    # Source
    source_checkpoint: str
    base_model: str = ""

    # Description
    description: str = ""
    training_data: List[str] = field(default_factory=list)

    # Metrics at this version
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None

    # Timestamps
    created_at: Optional[datetime] = None

    # Lineage
    parent_version: Optional[str] = None
    evolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "checkpoint_step": self.checkpoint_step,
            "source_checkpoint": self.source_checkpoint,
            "base_model": self.base_model,
            "description": self.description,
            "training_data": self.training_data,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "parent_version": self.parent_version,
            "evolution_notes": self.evolution_notes,
        }


# =============================================================================
# TREASURY (Resource Management)
# =============================================================================

@dataclass
class RetentionRule:
    """
    A rule for the Treasury's retention policy.
    """
    name: str
    description: str

    # Conditions
    min_age_hours: float = 0       # Minimum age before eligible
    max_count: Optional[int] = None  # Maximum items to keep
    max_size_gb: Optional[float] = None  # Maximum total size

    # Protection
    protect_latest: bool = True
    protect_best: bool = True
    protect_today: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "min_age_hours": self.min_age_hours,
            "max_count": self.max_count,
            "max_size_gb": self.max_size_gb,
            "protect_latest": self.protect_latest,
            "protect_best": self.protect_best,
            "protect_today": self.protect_today,
        }


@dataclass
class TreasuryStatus:
    """
    Current status of the Treasury (disk/resource usage).
    """
    # Disk usage
    total_disk_gb: float = 0.0
    used_disk_gb: float = 0.0
    free_disk_gb: float = 0.0
    usage_percent: float = 0.0

    # Vault contents
    checkpoint_count: int = 0
    checkpoint_size_gb: float = 0.0
    archive_count: int = 0
    archive_size_gb: float = 0.0

    # Status
    health: str = "healthy"  # healthy, warning, critical
    last_cleanup: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_disk_gb": round(self.total_disk_gb, 2),
            "used_disk_gb": round(self.used_disk_gb, 2),
            "free_disk_gb": round(self.free_disk_gb, 2),
            "usage_percent": round(self.usage_percent, 1),
            "checkpoint_count": self.checkpoint_count,
            "checkpoint_size_gb": round(self.checkpoint_size_gb, 2),
            "archive_count": self.archive_count,
            "archive_size_gb": round(self.archive_size_gb, 2),
            "health": self.health,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
        }
