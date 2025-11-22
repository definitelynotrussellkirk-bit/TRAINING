#!/usr/bin/env python3
"""
Automatic Disk Space Manager

Continuously monitors disk space and automatically cleans up when running low.
Prevents training crashes from "No space left on device" errors.

Runs as a daemon, checks every 5 minutes.
"""

import os
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/disk_manager.log'),
        logging.StreamHandler()
    ]
)

class DiskSpaceManager:
    """Automatically manage disk space to prevent training crashes."""

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        min_free_gb: int = 50,  # Minimum free space in GB
        min_free_percent: int = 10,  # Minimum free space percentage
        keep_versions: int = 2,  # Keep latest N model versions
        check_interval: int = 300  # Check every 5 minutes
    ):
        self.base_dir = Path(base_dir)
        self.min_free_gb = min_free_gb
        self.min_free_percent = min_free_percent
        self.keep_versions = keep_versions
        self.check_interval = check_interval

        # Create logs directory if needed
        (self.base_dir / 'logs').mkdir(exist_ok=True)

    def get_disk_usage(self) -> Tuple[int, int, int]:
        """Get disk usage stats (total, used, free) in GB."""
        stat = shutil.disk_usage(self.base_dir)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        return total_gb, used_gb, free_gb

    def needs_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        total_gb, used_gb, free_gb = self.get_disk_usage()
        free_percent = (free_gb / total_gb) * 100

        if free_gb < self.min_free_gb:
            logging.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free (< {self.min_free_gb}GB threshold)")
            return True

        if free_percent < self.min_free_percent:
            logging.warning(f"⚠️  Low disk space: {free_percent:.1f}% free (< {self.min_free_percent}% threshold)")
            return True

        return False

    def get_old_versions(self) -> List[Path]:
        """Get old model versions to delete (keep only latest N)."""
        versions_dir = self.base_dir / 'models' / 'versions'
        if not versions_dir.exists():
            return []

        # Get all version directories sorted by modification time (newest first)
        versions = sorted(
            [d for d in versions_dir.iterdir() if d.is_dir() and d.name.startswith('v')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Return all except the latest N
        to_delete = versions[self.keep_versions:]
        return to_delete

    def get_old_backups(self) -> List[Path]:
        """Get old backups to delete."""
        backups_dir = self.base_dir / 'models' / 'backups'
        if not backups_dir.exists():
            return []

        to_delete = []

        # Get all backup subdirectories
        for backup_type_dir in backups_dir.iterdir():
            if not backup_type_dir.is_dir():
                continue

            # Get backups sorted by modification time (newest first)
            backups = sorted(
                [d for d in backup_type_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Keep only latest 1 of each type
            to_delete.extend(backups[1:])

        return to_delete

    def get_old_logs(self) -> List[Path]:
        """Get old log files to delete (keep last 7 days)."""
        logs_dir = self.base_dir / 'logs'
        if not logs_dir.exists():
            return []

        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago

        to_delete = []
        for log_file in logs_dir.glob('*.log'):
            if log_file.stat().st_mtime < cutoff_time:
                to_delete.append(log_file)

        return to_delete

    def cleanup_git(self) -> int:
        """Clean up git repository to save space. Returns GB freed."""
        git_dir = self.base_dir / '.git'
        if not git_dir.exists():
            return 0

        size_before = sum(f.stat().st_size for f in git_dir.rglob('*') if f.is_file()) / (1024**3)

        # Run git garbage collection
        try:
            os.chdir(self.base_dir)
            os.system('git gc --aggressive --prune=now > /dev/null 2>&1')

            size_after = sum(f.stat().st_size for f in git_dir.rglob('*') if f.is_file()) / (1024**3)
            freed_gb = size_before - size_after

            if freed_gb > 0.1:
                logging.info(f"🗑️  Git cleanup freed {freed_gb:.1f}GB")

            return freed_gb
        except Exception as e:
            logging.error(f"Git cleanup failed: {e}")
            return 0

    def perform_cleanup(self) -> float:
        """Perform cleanup and return GB freed."""
        total_freed = 0.0

        logging.info("🧹 Starting automatic cleanup...")

        # 1. Delete old model versions
        old_versions = self.get_old_versions()
        for version_dir in old_versions:
            size_gb = sum(f.stat().st_size for f in version_dir.rglob('*') if f.is_file()) / (1024**3)
            try:
                shutil.rmtree(version_dir)
                logging.info(f"🗑️  Deleted old version: {version_dir.name} ({size_gb:.1f}GB)")
                total_freed += size_gb
            except Exception as e:
                logging.error(f"Failed to delete {version_dir}: {e}")

        # 2. Delete old backups
        old_backups = self.get_old_backups()
        for backup_dir in old_backups:
            size_gb = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()) / (1024**3)
            try:
                shutil.rmtree(backup_dir)
                logging.info(f"🗑️  Deleted old backup: {backup_dir.parent.name}/{backup_dir.name} ({size_gb:.1f}GB)")
                total_freed += size_gb
            except Exception as e:
                logging.error(f"Failed to delete {backup_dir}: {e}")

        # 3. Delete old logs
        old_logs = self.get_old_logs()
        for log_file in old_logs:
            size_mb = log_file.stat().st_size / (1024**2)
            try:
                log_file.unlink()
                if size_mb > 10:  # Only log if > 10MB
                    logging.info(f"🗑️  Deleted old log: {log_file.name} ({size_mb:.0f}MB)")
                total_freed += size_mb / 1024
            except Exception as e:
                logging.error(f"Failed to delete {log_file}: {e}")

        # 4. Clean up git if we still need more space
        _, _, free_gb = self.get_disk_usage()
        if free_gb < self.min_free_gb:
            total_freed += self.cleanup_git()

        logging.info(f"✅ Cleanup complete: freed {total_freed:.1f}GB")

        return total_freed

    def run(self):
        """Main daemon loop."""
        logging.info("🚀 Automatic Disk Space Manager started")
        logging.info(f"   Min free space: {self.min_free_gb}GB or {self.min_free_percent}%")
        logging.info(f"   Keep latest {self.keep_versions} model versions")
        logging.info(f"   Check interval: {self.check_interval}s")

        while True:
            try:
                total_gb, used_gb, free_gb = self.get_disk_usage()
                free_percent = (free_gb / total_gb) * 100

                logging.info(f"💾 Disk: {used_gb:.0f}GB used, {free_gb:.0f}GB free ({free_percent:.1f}%)")

                if self.needs_cleanup():
                    freed_gb = self.perform_cleanup()

                    # Check result
                    _, _, new_free_gb = self.get_disk_usage()
                    if new_free_gb >= self.min_free_gb:
                        logging.info(f"✅ Cleanup successful: {new_free_gb:.1f}GB free")
                    else:
                        logging.warning(f"⚠️  Still low on space: {new_free_gb:.1f}GB free")
                        logging.warning("   Consider manual cleanup or increasing disk size")

            except Exception as e:
                logging.error(f"Error in cleanup cycle: {e}")

            # Sleep until next check
            time.sleep(self.check_interval)


if __name__ == '__main__':
    manager = DiskSpaceManager(
        base_dir="/path/to/training",
        min_free_gb=50,  # Cleanup when < 50GB free
        min_free_percent=10,  # Or when < 10% free
        keep_versions=2,  # Keep latest 2 versions
        check_interval=300  # Check every 5 minutes
    )

    manager.run()
