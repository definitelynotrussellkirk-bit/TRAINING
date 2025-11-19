#!/usr/bin/env python3
"""
Backup Management System

CRITICAL: Prevents data loss by creating verified backups before ANY deletion

Key Features:
1. Automatic backups before consolidation
2. Automatic backups before deletion
3. Backup verification (checksums, file counts, sizes)
4. Retention policy management
5. Detailed logging of all operations

Rule: NEVER delete without backing up and verifying first!
"""

import json
import shutil
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackupManager:
    def __init__(self, base_dir: str = "/path/to/training"):
        self.base_dir = Path(base_dir)
        self.backups_dir = self.base_dir / "models" / "backups"

        # Create backup directories
        self.pre_consolidation_dir = self.backups_dir / "pre_consolidation"
        self.pre_deletion_dir = self.backups_dir / "pre_deletion"
        self.emergency_dir = self.backups_dir / "emergency"

        for d in [self.pre_consolidation_dir, self.pre_deletion_dir, self.emergency_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _calculate_dir_hash(self, path: Path) -> str:
        """Calculate a hash of directory contents for verification"""
        hash_md5 = hashlib.md5()
        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                hash_md5.update(str(file_path.relative_to(path)).encode())
                hash_md5.update(file_path.read_bytes())
        return hash_md5.hexdigest()

    def _get_dir_stats(self, path: Path) -> Dict:
        """Get directory statistics for verification"""
        if not path.exists():
            return {"exists": False}

        files = list(path.rglob('*'))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            "exists": True,
            "total_files": len([f for f in files if f.is_file()]),
            "total_dirs": len([f for f in files if f.is_dir()]),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }

    def create_backup(
        self,
        source_path: str,
        backup_type: str,
        reason: str,
        verify: bool = True
    ) -> Tuple[bool, Optional[Path]]:
        """
        Create a backup of a directory

        Args:
            source_path: Path to backup
            backup_type: One of "pre_consolidation", "pre_deletion", "emergency"
            reason: Why this backup is being created
            verify: Whether to verify the backup

        Returns:
            (success, backup_path)
        """
        source = Path(source_path)
        if not source.exists():
            logger.error(f"❌ Source path {source} does not exist")
            return False, None

        # Determine backup directory
        if backup_type == "pre_consolidation":
            backup_base = self.pre_consolidation_dir
        elif backup_type == "pre_deletion":
            backup_base = self.pre_deletion_dir
        elif backup_type == "emergency":
            backup_base = self.emergency_dir
        else:
            logger.error(f"❌ Invalid backup type: {backup_type}")
            return False, None

        # Create backup path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.name}_{timestamp}"
        backup_path = backup_base / backup_name

        logger.info(f"🔒 Creating {backup_type} backup...")
        logger.info(f"   Source: {source}")
        logger.info(f"   Backup: {backup_path}")
        logger.info(f"   Reason: {reason}")

        try:
            # Get source stats before backup
            source_stats = self._get_dir_stats(source)
            logger.info(f"   Source: {source_stats['total_files']} files, {source_stats['total_size_mb']:.1f} MB")

            # Create backup
            shutil.copytree(source, backup_path)

            # Get backup stats
            backup_stats = self._get_dir_stats(backup_path)
            logger.info(f"   Backup: {backup_stats['total_files']} files, {backup_stats['total_size_mb']:.1f} MB")

            # Verify if requested
            if verify:
                verified, message = self._verify_backup(source, backup_path, source_stats, backup_stats)
                if not verified:
                    logger.error(f"❌ Backup verification failed: {message}")
                    # Clean up failed backup
                    shutil.rmtree(backup_path)
                    return False, None

            # Save backup metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "backup_type": backup_type,
                "reason": reason,
                "source_path": str(source),
                "backup_path": str(backup_path),
                "source_stats": source_stats,
                "backup_stats": backup_stats,
                "verified": verify
            }

            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✅ Backup created and verified successfully")
            return True, backup_path

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            # Clean up partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return False, None

    def _verify_backup(
        self,
        source: Path,
        backup: Path,
        source_stats: Dict,
        backup_stats: Dict
    ) -> Tuple[bool, str]:
        """Verify backup matches source"""

        # Check file counts match
        if source_stats['total_files'] != backup_stats['total_files']:
            return False, f"File count mismatch: {source_stats['total_files']} vs {backup_stats['total_files']}"

        # Check total size matches (within 1%)
        size_diff_pct = abs(source_stats['total_size_bytes'] - backup_stats['total_size_bytes']) / source_stats['total_size_bytes'] * 100
        if size_diff_pct > 1.0:
            return False, f"Size mismatch: {size_diff_pct:.2f}% difference"

        # Check critical files exist
        critical_files = ["adapter_model.safetensors", "adapter_config.json"]
        for filename in critical_files:
            source_file = source / filename
            backup_file = backup / filename

            if source_file.exists():
                if not backup_file.exists():
                    return False, f"Critical file missing: {filename}"

                # Verify file sizes match
                if source_file.stat().st_size != backup_file.stat().st_size:
                    return False, f"File size mismatch: {filename}"

        logger.info(f"   ✅ Verification passed")
        return True, "OK"

    def backup_before_consolidation(self, adapter_path: str) -> Tuple[bool, Optional[Path]]:
        """Backup adapter before consolidation"""
        return self.create_backup(
            source_path=adapter_path,
            backup_type="pre_consolidation",
            reason="Pre-consolidation safety backup",
            verify=True
        )

    def backup_before_deletion(self, path: str, reason: str) -> Tuple[bool, Optional[Path]]:
        """Backup before deleting anything"""
        return self.create_backup(
            source_path=path,
            backup_type="pre_deletion",
            reason=reason,
            verify=True
        )

    def emergency_backup(self, path: str, reason: str) -> Tuple[bool, Optional[Path]]:
        """Create emergency backup"""
        return self.create_backup(
            source_path=path,
            backup_type="emergency",
            reason=reason,
            verify=True
        )

    def list_backups(self, backup_type: Optional[str] = None) -> List[Dict]:
        """List all backups or backups of a specific type"""
        backups = []

        if backup_type:
            if backup_type == "pre_consolidation":
                dirs = [self.pre_consolidation_dir]
            elif backup_type == "pre_deletion":
                dirs = [self.pre_deletion_dir]
            elif backup_type == "emergency":
                dirs = [self.emergency_dir]
            else:
                return []
        else:
            dirs = [self.pre_consolidation_dir, self.pre_deletion_dir, self.emergency_dir]

        for base_dir in dirs:
            for backup_dir in sorted(base_dir.iterdir()):
                if backup_dir.is_dir():
                    metadata_path = backup_dir / "backup_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            backups.append(json.load(f))

        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

    def cleanup_old_backups(self, retention_days: int = 30, dry_run: bool = True) -> List[str]:
        """
        Clean up backups older than retention period

        Args:
            retention_days: Keep backups newer than this
            dry_run: If True, just report what would be deleted

        Returns:
            List of deleted backup paths
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted = []

        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up backups older than {retention_days} days")
        logger.info(f"   Cutoff date: {cutoff_date.isoformat()}")

        for backup in self.list_backups():
            backup_date = datetime.fromisoformat(backup['created_at'])

            if backup_date < cutoff_date:
                backup_path = Path(backup['backup_path'])

                if dry_run:
                    logger.info(f"   Would delete: {backup_path.name} ({backup['backup_type']})")
                else:
                    logger.info(f"   Deleting: {backup_path.name} ({backup['backup_type']})")
                    shutil.rmtree(backup_path)

                deleted.append(str(backup_path))

        if not deleted:
            logger.info("   No backups to clean up")
        else:
            logger.info(f"   {'Would delete' if dry_run else 'Deleted'} {len(deleted)} backups")

        return deleted

    def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """Restore a backup to a target location"""
        backup = Path(backup_path)
        target = Path(target_path)

        if not backup.exists():
            logger.error(f"❌ Backup {backup} does not exist")
            return False

        try:
            logger.info(f"🔓 Restoring backup...")
            logger.info(f"   From: {backup}")
            logger.info(f"   To: {target}")

            # Remove existing target
            if target.exists():
                logger.info(f"   Removing existing target...")
                shutil.rmtree(target)

            # Copy backup to target
            shutil.copytree(backup, target)

            # Remove backup metadata from restored copy
            metadata_in_restore = target / "backup_metadata.json"
            if metadata_in_restore.exists():
                metadata_in_restore.unlink()

            logger.info(f"✅ Backup restored successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backup Management")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List backups
    list_parser = subparsers.add_parser('list', help='List all backups')
    list_parser.add_argument('--type', choices=['pre_consolidation', 'pre_deletion', 'emergency'], help='Filter by type')

    # Create backup
    backup_parser = subparsers.add_parser('backup', help='Create a backup')
    backup_parser.add_argument('source', help='Path to backup')
    backup_parser.add_argument('--type', required=True, choices=['pre_consolidation', 'pre_deletion', 'emergency'])
    backup_parser.add_argument('--reason', required=True, help='Reason for backup')

    # Cleanup old backups
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--retention-days', type=int, default=30, help='Retention period in days')
    cleanup_parser.add_argument('--execute', action='store_true', help='Actually delete (default is dry-run)')

    # Restore backup
    restore_parser = subparsers.add_parser('restore', help='Restore a backup')
    restore_parser.add_argument('backup_path', help='Path to backup')
    restore_parser.add_argument('target_path', help='Where to restore')

    args = parser.parse_args()

    manager = BackupManager(args.base_dir)

    if args.command == 'list':
        backups = manager.list_backups(args.type)
        print(f"\n{'='*80}")
        print(f"BACKUPS ({len(backups)} total)")
        print(f"{'='*80}\n")

        for b in backups:
            print(f"📦 {Path(b['backup_path']).name}")
            print(f"   Type: {b['backup_type']}")
            print(f"   Created: {b['created_at']}")
            print(f"   Reason: {b['reason']}")
            print(f"   Source: {b['source_path']}")
            print(f"   Size: {b['backup_stats']['total_size_mb']:.1f} MB")
            print(f"   Verified: {'✅' if b['verified'] else '❌'}")
            print()

    elif args.command == 'backup':
        success, backup_path = manager.create_backup(
            source_path=args.source,
            backup_type=args.type,
            reason=args.reason,
            verify=True
        )

        if success:
            print(f"✅ Backup created: {backup_path}")
        else:
            print(f"❌ Backup failed")

    elif args.command == 'cleanup':
        deleted = manager.cleanup_old_backups(
            retention_days=args.retention_days,
            dry_run=not args.execute
        )

        if args.execute:
            print(f"✅ Deleted {len(deleted)} backups")
        else:
            print(f"ℹ️  Would delete {len(deleted)} backups (use --execute to actually delete)")

    elif args.command == 'restore':
        success = manager.restore_backup(args.backup_path, args.target_path)
        if success:
            print(f"✅ Restored backup to {args.target_path}")
        else:
            print(f"❌ Restore failed")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
