"""
Archivist - Guardian of the backup archives.

The Archivist is responsible for creating, verifying, and managing
backup archives in the Vault. No treasure is deleted without the
Archivist's seal of approval (verified backup).

RPG Flavor:
    The Archivist is an ancient keeper who guards the backup scrolls.
    Before any treasure is discarded, the Archivist creates a verified
    copy and stamps it with their seal. Their motto: "Never lose what
    cannot be recreated."

Archive Types:
    PRE_CONSOLIDATION - Before merging checkpoints
    PRE_DELETION      - Before cleanup operations
    EMERGENCY         - Urgent safety backup
    SCHEDULED         - Regular scheduled backups

This module wraps management/backup_manager.py with RPG-themed naming.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vault.types import ArchiveEntry, ArchiveType, ProtectionLevel

# Import the underlying manager
from management.backup_manager import BackupManager as _BackupManager


class Archivist(_BackupManager):
    """
    The Archivist - guardian of backup archives.

    RPG wrapper around BackupManager with themed method names.

    Usage:
        archivist = Archivist(base_dir)

        # Create archive before deletion
        archive = archivist.seal_for_safekeeping(
            treasure_path="/path/to/checkpoint",
            reason="Cleanup operation"
        )

        # Verify an archive
        if archivist.verify_seal(archive.archive_path):
            print("Archive verified, safe to proceed")

        # List archives
        archives = archivist.list_archives()

        # Restore from archive
        archivist.restore_from_archive(archive_path, restore_to)
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the Archivist.

        Args:
            base_dir: Base training directory (default: auto-detect)
        """
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = str(get_base_dir())
        super().__init__(base_dir)

        # Rename directories with RPG theme (but keep underlying paths)
        self.consolidation_vault = self.pre_consolidation_dir
        self.deletion_vault = self.pre_deletion_dir
        self.emergency_vault = self.emergency_dir

    # =========================================================================
    # ARCHIVE CREATION
    # =========================================================================

    def seal_for_safekeeping(
        self,
        treasure_path: str | Path,
        archive_type: ArchiveType = ArchiveType.PRE_DELETION,
        reason: str = "",
    ) -> Optional[ArchiveEntry]:
        """
        Create a sealed archive of treasure before operations.

        The Archivist creates a verified copy before any deletion.

        Args:
            treasure_path: Path to item to archive
            archive_type: Type of archive to create
            reason: Why this archive is being created

        Returns:
            ArchiveEntry if successful, None if failed
        """
        treasure_path = Path(treasure_path)

        if not treasure_path.exists():
            return None

        # Map archive type to method
        if archive_type == ArchiveType.PRE_CONSOLIDATION:
            success = self.backup_before_consolidation(str(treasure_path))
            archive_dir = self.consolidation_vault
        elif archive_type == ArchiveType.EMERGENCY:
            success = self.create_emergency_backup(str(treasure_path))
            archive_dir = self.emergency_vault
        else:  # PRE_DELETION or SCHEDULED
            success = self.backup_before_deletion(str(treasure_path))
            archive_dir = self.deletion_vault

        if not success:
            return None

        # Find the created archive
        archive_name = treasure_path.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{archive_name}_{timestamp}"

        # Create entry record
        entry = ArchiveEntry(
            archive_id=f"{archive_type.value}_{timestamp}",
            archive_type=archive_type,
            source_path=str(treasure_path),
            archive_path=str(archive_path),
            verified=True,  # backup_manager verifies automatically
            created_at=datetime.now(),
            reason=reason,
        )

        return entry

    def seal_checkpoint(
        self,
        checkpoint_path: str | Path,
        reason: str = "Pre-deletion backup",
    ) -> Optional[ArchiveEntry]:
        """
        Convenience method to archive a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            reason: Why archiving

        Returns:
            ArchiveEntry if successful
        """
        return self.seal_for_safekeeping(
            treasure_path=checkpoint_path,
            archive_type=ArchiveType.PRE_DELETION,
            reason=reason,
        )

    def emergency_seal(
        self,
        treasure_path: str | Path,
        reason: str = "Emergency backup",
    ) -> Optional[ArchiveEntry]:
        """
        Create an emergency archive immediately.

        Args:
            treasure_path: Path to item
            reason: Emergency reason

        Returns:
            ArchiveEntry if successful
        """
        return self.seal_for_safekeeping(
            treasure_path=treasure_path,
            archive_type=ArchiveType.EMERGENCY,
            reason=reason,
        )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify_seal(self, archive_path: str | Path) -> bool:
        """
        Verify the integrity of an archive's seal.

        Checks that the archive is complete and uncorrupted.

        Args:
            archive_path: Path to archive to verify

        Returns:
            True if archive is verified intact
        """
        return self.verify_backup(str(archive_path))

    def inspect_archive(self, archive_path: str | Path) -> Dict[str, Any]:
        """
        Inspect an archive and return detailed info.

        Args:
            archive_path: Path to archive

        Returns:
            Dict with archive details
        """
        archive_path = Path(archive_path)

        if not archive_path.exists():
            return {"error": "Archive not found"}

        info = self.get_backup_info(str(archive_path))
        info["verified"] = self.verify_seal(archive_path)

        return info

    # =========================================================================
    # ARCHIVE MANAGEMENT
    # =========================================================================

    def list_archives(
        self,
        archive_type: Optional[ArchiveType] = None,
    ) -> List[ArchiveEntry]:
        """
        List all archives in the Vault.

        Args:
            archive_type: Filter by type, or None for all

        Returns:
            List of ArchiveEntry
        """
        archives = []

        type_dirs = {
            ArchiveType.PRE_CONSOLIDATION: self.consolidation_vault,
            ArchiveType.PRE_DELETION: self.deletion_vault,
            ArchiveType.EMERGENCY: self.emergency_vault,
        }

        dirs_to_check = (
            [type_dirs[archive_type]] if archive_type
            else type_dirs.values()
        )

        for vault_dir in dirs_to_check:
            if not vault_dir.exists():
                continue

            for item in vault_dir.iterdir():
                if item.is_dir():
                    # Determine type from directory
                    for atype, adir in type_dirs.items():
                        if vault_dir == adir:
                            entry = ArchiveEntry(
                                archive_id=item.name,
                                archive_type=atype,
                                source_path="",  # Unknown without metadata
                                archive_path=str(item),
                                created_at=datetime.fromtimestamp(item.stat().st_mtime),
                            )
                            archives.append(entry)
                            break

        return sorted(archives, key=lambda x: x.created_at or datetime.min, reverse=True)

    def restore_from_archive(
        self,
        archive_path: str | Path,
        restore_to: str | Path,
    ) -> bool:
        """
        Restore treasure from an archive.

        Args:
            archive_path: Path to archive
            restore_to: Where to restore

        Returns:
            True if successful
        """
        return self.restore_backup(str(archive_path), str(restore_to))

    def purge_old_archives(
        self,
        max_age_days: int = 30,
        archive_type: Optional[ArchiveType] = None,
    ) -> int:
        """
        Purge archives older than specified age.

        Args:
            max_age_days: Maximum age in days
            archive_type: Type to purge, or None for all

        Returns:
            Number of archives purged
        """
        return self.cleanup_old_backups(max_age_days)

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_vault_status(self) -> Dict[str, Any]:
        """
        Get status of archive vaults.

        Returns:
            Dict with vault statistics
        """
        status = {
            "consolidation_vault": {
                "count": 0,
                "size_gb": 0.0,
            },
            "deletion_vault": {
                "count": 0,
                "size_gb": 0.0,
            },
            "emergency_vault": {
                "count": 0,
                "size_gb": 0.0,
            },
            "total_archives": 0,
            "total_size_gb": 0.0,
        }

        vaults = {
            "consolidation_vault": self.consolidation_vault,
            "deletion_vault": self.deletion_vault,
            "emergency_vault": self.emergency_vault,
        }

        for name, vault_dir in vaults.items():
            if vault_dir.exists():
                count = len(list(vault_dir.iterdir()))
                size = sum(
                    f.stat().st_size
                    for f in vault_dir.rglob("*")
                    if f.is_file()
                ) / (1024 ** 3)

                status[name]["count"] = count
                status[name]["size_gb"] = round(size, 2)
                status["total_archives"] += count
                status["total_size_gb"] += size

        status["total_size_gb"] = round(status["total_size_gb"], 2)

        return status


# Convenience function
def get_archivist(base_dir: Optional[str] = None) -> Archivist:
    """Get an Archivist instance for the given base directory."""
    return Archivist(base_dir)


# Re-export original for backward compatibility
BackupManager = _BackupManager
