#!/usr/bin/env python3
"""
Daily Snapshot Manager

Creates daily snapshots of the current model and manages retention.

Usage:
    python3 daily_snapshot.py create              # Create today's snapshot
    python3 daily_snapshot.py cleanup --days 7    # Keep only last 7 days
    python3 daily_snapshot.py list                # List all snapshots
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import argparse

class DailySnapshotManager:
    def __init__(self, base_dir="/path/to/training"):
        self.base_dir = Path(base_dir)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.current_model = self.base_dir / "current_model"

    def create_snapshot(self):
        """Create a snapshot for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        snapshot_path = self.snapshots_dir / today

        if snapshot_path.exists():
            print(f"Snapshot for {today} already exists at: {snapshot_path}")
            return snapshot_path

        if not self.current_model.exists():
            print("ERROR: current_model/ does not exist")
            return None

        print(f"Creating snapshot for {today}...")
        self.snapshots_dir.mkdir(exist_ok=True)

        try:
            shutil.copytree(self.current_model, snapshot_path)
            print(f"✓ Snapshot created: {snapshot_path}")

            # Get size
            size = self._get_dir_size(snapshot_path)
            print(f"  Size: {size:.2f} GB")

            return snapshot_path
        except Exception as e:
            print(f"ERROR creating snapshot: {e}")
            return None

    def cleanup_old_snapshots(self, retention_days=7):
        """Remove snapshots older than retention_days"""
        if not self.snapshots_dir.exists():
            print("No snapshots directory found")
            return

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        print(f"Cleaning up snapshots older than {cutoff_str}...")

        removed = []
        kept = []

        for snapshot in sorted(self.snapshots_dir.iterdir()):
            if not snapshot.is_dir():
                continue

            snapshot_date = snapshot.name

            try:
                # Check if snapshot is older than cutoff
                if snapshot_date < cutoff_str:
                    size = self._get_dir_size(snapshot)
                    shutil.rmtree(snapshot)
                    removed.append((snapshot_date, size))
                    print(f"  ✓ Removed: {snapshot_date} ({size:.2f} GB)")
                else:
                    kept.append(snapshot_date)
            except Exception as e:
                print(f"  ERROR removing {snapshot_date}: {e}")

        if removed:
            total_freed = sum(size for _, size in removed)
            print(f"\n✓ Removed {len(removed)} snapshots, freed {total_freed:.2f} GB")
        else:
            print(f"\n✓ No snapshots older than {retention_days} days")

        if kept:
            print(f"✓ Kept {len(kept)} snapshots: {', '.join(kept)}")

    def list_snapshots(self):
        """List all snapshots with sizes"""
        if not self.snapshots_dir.exists():
            print("No snapshots directory found")
            return

        snapshots = []
        for snapshot in sorted(self.snapshots_dir.iterdir()):
            if snapshot.is_dir():
                size = self._get_dir_size(snapshot)
                snapshots.append((snapshot.name, size))

        if not snapshots:
            print("No snapshots found")
            return

        print(f"\nDaily Snapshots ({len(snapshots)} total):")
        print(f"{'Date':<15} {'Size (GB)':>10}")
        print("-" * 27)

        total_size = 0
        for date, size in snapshots:
            print(f"{date:<15} {size:>10.2f}")
            total_size += size

        print("-" * 27)
        print(f"{'TOTAL':<15} {total_size:>10.2f}")

    def _get_dir_size(self, path):
        """Get directory size in GB"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except:
                    pass
        return total / (1024**3)  # Convert to GB

def main():
    parser = argparse.ArgumentParser(description="Daily Snapshot Manager")
    parser.add_argument('command', choices=['create', 'cleanup', 'list'],
                       help="Command to execute")
    parser.add_argument('--days', type=int, default=7,
                       help="Retention days for cleanup (default: 7)")

    args = parser.parse_args()

    manager = DailySnapshotManager()

    if args.command == 'create':
        manager.create_snapshot()
    elif args.command == 'cleanup':
        manager.cleanup_old_snapshots(args.days)
    elif args.command == 'list':
        manager.list_snapshots()

if __name__ == "__main__":
    main()
