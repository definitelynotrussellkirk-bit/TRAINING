"""
Treasury - Resource management for the Vault.

The Treasury manages the realm's precious resources (disk space) and
enforces retention policies for checkpoints and archives. The Treasurer
ensures we never run out of space while keeping important treasures safe.

RPG Flavor:
    The Treasurer oversees the Vault's capacity, deciding which treasures
    to keep and which to recycle. Protected items (sacred, latest, best)
    are never touched. Expendable items are recycled when space runs low.

Retention Policies:
    - Minimum age: 36 hours before eligible for cleanup
    - Size limit: 150GB total checkpoint storage
    - Protected: Latest checkpoint, best performer, today/yesterday

This module wraps management/retention_manager.py and auto_disk_manager.py.
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from vault.types import (
    TreasureType,
    ProtectionLevel,
    VaultRecord,
    RetentionRule,
    TreasuryStatus,
)


class Treasury:
    """
    The Treasury - manages Vault resources and retention.

    Tracks disk usage, enforces retention policies, and manages
    checkpoint lifecycle.

    Usage:
        treasury = Treasury(base_dir)

        # Check vault status
        status = treasury.get_status()
        print(f"Free space: {status.free_disk_gb} GB")

        # List treasures by protection level
        expendable = treasury.list_expendable_treasures()

        # Run cleanup if needed
        if status.health == "warning":
            freed = treasury.cleanup_expendable(target_free_gb=50)
    """

    def __init__(
        self,
        base_dir: str | Path = "/path/to/training",
        checkpoint_limit_gb: float = 150.0,
        min_age_hours: float = 36.0,
    ):
        """
        Initialize the Treasury.

        Args:
            base_dir: Base training directory
            checkpoint_limit_gb: Maximum checkpoint storage in GB
            min_age_hours: Minimum age before cleanup eligible
        """
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = self.base_dir / "models" / "current_model"
        self.backup_dir = self.base_dir / "models" / "backups"

        # Retention settings
        self.checkpoint_limit_gb = checkpoint_limit_gb
        self.min_age_hours = min_age_hours

        # Default retention rule
        self.default_rule = RetentionRule(
            name="default",
            description="Standard retention policy",
            min_age_hours=min_age_hours,
            max_size_gb=checkpoint_limit_gb,
            protect_latest=True,
            protect_best=True,
            protect_today=True,
        )

    # =========================================================================
    # STATUS & INVENTORY
    # =========================================================================

    def get_status(self) -> TreasuryStatus:
        """
        Get current Treasury status.

        Returns:
            TreasuryStatus with disk and vault info
        """
        # Get disk stats
        disk = shutil.disk_usage(self.base_dir)
        total_gb = disk.total / (1024 ** 3)
        used_gb = disk.used / (1024 ** 3)
        free_gb = disk.free / (1024 ** 3)
        usage_pct = (disk.used / disk.total) * 100

        # Count checkpoints
        checkpoints = self._list_checkpoints()
        checkpoint_size = sum(c.size_bytes for c in checkpoints) / (1024 ** 3)

        # Count archives
        archives = list(self.backup_dir.rglob("*")) if self.backup_dir.exists() else []
        archive_size = sum(
            f.stat().st_size for f in archives if f.is_file()
        ) / (1024 ** 3)

        # Determine health
        if usage_pct > 90:
            health = "critical"
        elif usage_pct > 80:
            health = "warning"
        else:
            health = "healthy"

        return TreasuryStatus(
            total_disk_gb=total_gb,
            used_disk_gb=used_gb,
            free_disk_gb=free_gb,
            usage_percent=usage_pct,
            checkpoint_count=len(checkpoints),
            checkpoint_size_gb=checkpoint_size,
            archive_count=len([a for a in archives if a.is_dir()]),
            archive_size_gb=archive_size,
            health=health,
        )

    def inventory_treasures(self) -> List[VaultRecord]:
        """
        Get full inventory of all treasures in the Vault.

        Returns:
            List of VaultRecord for all stored items
        """
        treasures = []

        # Add checkpoints
        for ckpt in self._list_checkpoints():
            treasures.append(ckpt)

        # Add base model if exists
        base_model = self.base_dir / "models" / "Qwen3-0.6B"
        if base_model.exists():
            size = sum(f.stat().st_size for f in base_model.rglob("*") if f.is_file())
            treasures.append(VaultRecord(
                name="Qwen3-0.6B",
                path=str(base_model),
                treasure_type=TreasureType.BASE_MODEL,
                protection=ProtectionLevel.SACRED,
                size_bytes=size,
            ))

        return treasures

    def _list_checkpoints(self) -> List[VaultRecord]:
        """List all checkpoints as VaultRecords."""
        checkpoints = []

        if not self.checkpoint_dir.exists():
            return checkpoints

        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                # Extract step number
                try:
                    step = int(item.name.split("-")[1])
                except (IndexError, ValueError):
                    step = 0

                # Calculate size
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                mtime = datetime.fromtimestamp(item.stat().st_mtime)

                # Determine protection level
                protection = self._get_protection_level(item, step, mtime)

                checkpoints.append(VaultRecord(
                    name=item.name,
                    path=str(item),
                    treasure_type=TreasureType.CHECKPOINT,
                    protection=protection,
                    size_bytes=size,
                    created_at=mtime,
                    step_number=step,
                ))

        return sorted(checkpoints, key=lambda x: x.step_number or 0, reverse=True)

    def _get_protection_level(
        self,
        path: Path,
        step: int,
        mtime: datetime,
    ) -> ProtectionLevel:
        """Determine protection level for a checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if not checkpoints:
            return ProtectionLevel.GUARDED

        # Latest checkpoint is protected
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        if path == latest:
            return ProtectionLevel.PROTECTED

        # Today's checkpoints are protected
        today = datetime.now().date()
        if mtime.date() == today:
            return ProtectionLevel.GUARDED

        # Yesterday's checkpoints are guarded
        yesterday = today - timedelta(days=1)
        if mtime.date() == yesterday:
            return ProtectionLevel.GUARDED

        # Check minimum age
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours < self.min_age_hours:
            return ProtectionLevel.GUARDED

        return ProtectionLevel.EXPENDABLE

    # =========================================================================
    # PROTECTION & CLEANUP
    # =========================================================================

    def list_expendable_treasures(self) -> List[VaultRecord]:
        """
        List treasures that can be safely removed.

        Returns:
            List of expendable VaultRecords
        """
        return [
            t for t in self.inventory_treasures()
            if t.protection == ProtectionLevel.EXPENDABLE
        ]

    def list_protected_treasures(self) -> List[VaultRecord]:
        """
        List treasures that are protected from cleanup.

        Returns:
            List of protected VaultRecords
        """
        return [
            t for t in self.inventory_treasures()
            if t.protection in (ProtectionLevel.SACRED, ProtectionLevel.PROTECTED)
        ]

    def cleanup_expendable(
        self,
        target_free_gb: Optional[float] = None,
        max_to_remove: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Clean up expendable treasures to free space.

        Args:
            target_free_gb: Target free space in GB
            max_to_remove: Maximum items to remove
            dry_run: If True, only simulate cleanup

        Returns:
            Dict with cleanup results
        """
        expendable = self.list_expendable_treasures()

        # Sort by age (oldest first)
        expendable.sort(key=lambda x: x.created_at or datetime.max)

        removed = []
        freed_bytes = 0

        for treasure in expendable:
            # Check limits
            if max_to_remove and len(removed) >= max_to_remove:
                break

            if target_free_gb:
                current_free = shutil.disk_usage(self.base_dir).free / (1024 ** 3)
                if current_free >= target_free_gb:
                    break

            # Remove treasure
            if not dry_run:
                try:
                    shutil.rmtree(treasure.path)
                    removed.append(treasure.name)
                    freed_bytes += treasure.size_bytes
                except Exception as e:
                    continue
            else:
                removed.append(treasure.name)
                freed_bytes += treasure.size_bytes

        return {
            "removed": removed,
            "removed_count": len(removed),
            "freed_gb": round(freed_bytes / (1024 ** 3), 2),
            "dry_run": dry_run,
        }

    def enforce_retention(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Enforce the Treasury's retention policy.

        Removes checkpoints exceeding size limit, respecting protections.

        Args:
            dry_run: If True, only simulate

        Returns:
            Dict with enforcement results
        """
        status = self.get_status()

        if status.checkpoint_size_gb <= self.checkpoint_limit_gb:
            return {
                "action": "none",
                "reason": f"Within limit ({status.checkpoint_size_gb:.1f}/{self.checkpoint_limit_gb} GB)",
            }

        excess_gb = status.checkpoint_size_gb - self.checkpoint_limit_gb

        return self.cleanup_expendable(
            target_free_gb=status.free_disk_gb + excess_gb,
            dry_run=dry_run,
        )

    # =========================================================================
    # DISK MANAGEMENT
    # =========================================================================

    def get_disk_usage_by_category(self) -> Dict[str, float]:
        """
        Get disk usage breakdown by category.

        Returns:
            Dict mapping category to size in GB
        """
        categories = {
            "checkpoints": self.checkpoint_dir,
            "backups": self.backup_dir,
            "base_model": self.base_dir / "models" / "Qwen3-0.6B",
            "logs": self.base_dir / "logs",
            "data": self.base_dir / "data",
            "queue": self.base_dir / "queue",
        }

        usage = {}
        for name, path in categories.items():
            if path.exists():
                size = sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                ) / (1024 ** 3)
                usage[name] = round(size, 2)
            else:
                usage[name] = 0.0

        return usage

    def emergency_cleanup(self, target_free_gb: float = 50.0) -> Dict[str, Any]:
        """
        Emergency cleanup when disk is critically low.

        More aggressive than normal cleanup, but still respects
        SACRED protection level.

        Args:
            target_free_gb: Target free space

        Returns:
            Dict with cleanup results
        """
        # First try normal cleanup
        result = self.cleanup_expendable(target_free_gb=target_free_gb)

        status = self.get_status()
        if status.free_disk_gb >= target_free_gb:
            return result

        # If still not enough, try removing GUARDED items (except today)
        guarded = [
            t for t in self.inventory_treasures()
            if t.protection == ProtectionLevel.GUARDED
            and t.created_at
            and t.created_at.date() != datetime.now().date()
        ]

        guarded.sort(key=lambda x: x.created_at or datetime.max)

        for treasure in guarded:
            if shutil.disk_usage(self.base_dir).free / (1024 ** 3) >= target_free_gb:
                break

            try:
                shutil.rmtree(treasure.path)
                result["removed"].append(treasure.name)
                result["removed_count"] += 1
                result["freed_gb"] += treasure.size_gb
            except Exception:
                continue

        return result


# Convenience function
def get_treasury(base_dir: str | Path = "/path/to/training") -> Treasury:
    """Get a Treasury instance for the given base directory."""
    return Treasury(base_dir)
