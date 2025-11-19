#!/usr/bin/env python3
"""
Safe Checkpoint Cleanup - Removes old checkpoints while keeping latest

This prevents:
1. Accidentally deleting ALL checkpoints
2. Deleting checkpoints while training is active
3. Disk space issues from too many checkpoints

Usage:
    python3 safe_checkpoint_cleanup.py [--keep 5] [--dry-run] [--force]
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/path/to/training")
MODEL_DIR = BASE_DIR / "current_model"
STATUS_FILE = BASE_DIR / "status" / "training_status.json"

DEFAULT_KEEP = 5

class SafeCheckpointCleanup:
    def __init__(self, keep=DEFAULT_KEEP, dry_run=False, force=False):
        self.keep = keep
        self.dry_run = dry_run
        self.force = force

    def is_training_active(self):
        """Check if training is currently active"""
        try:
            # Check if status file was updated recently (within last 5 minutes)
            if STATUS_FILE.exists():
                mtime = STATUS_FILE.stat().st_mtime
                age = time.time() - mtime
                return age < 300  # 5 minutes
        except:
            pass
        return False

    def get_checkpoints(self):
        """Get all checkpoints sorted by step number"""
        if not MODEL_DIR.exists():
            return []

        checkpoints = []
        for ckpt_dir in MODEL_DIR.glob("checkpoint-*"):
            try:
                # Extract step number
                step = int(ckpt_dir.name.split('-')[1])
                checkpoints.append((step, ckpt_dir))
            except:
                pass

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def get_checkpoint_size(self, ckpt_dir):
        """Get checkpoint size in GB"""
        total = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
        return total / (1024**3)

    def cleanup(self):
        """Perform safe cleanup"""
        print("\n" + "="*80)
        print("🧹 SAFE CHECKPOINT CLEANUP")
        print("="*80)

        # Safety check: Is training active?
        if not self.force and self.is_training_active():
            print("\n❌ ABORT: Training appears to be active!")
            print("   Status file was updated within last 5 minutes")
            print("   Cleaning checkpoints during training could cause:")
            print("   1. Training to fail if it tries to load deleted checkpoint")
            print("   2. Loss of recent progress")
            print("\n   Wait for training to finish or use --force")
            return False

        # Get all checkpoints
        checkpoints = self.get_checkpoints()
        if not checkpoints:
            print("\n✅ No checkpoints found - nothing to clean")
            return True

        print(f"\n📦 Found {len(checkpoints)} checkpoints")
        print(f"   Range: checkpoint-{checkpoints[0][0]} to checkpoint-{checkpoints[-1][0]}")

        # Calculate what to keep vs delete
        if len(checkpoints) <= self.keep:
            print(f"\n✅ Only {len(checkpoints)} checkpoints - keeping all")
            print(f"   (Threshold: keep latest {self.keep})")
            return True

        to_keep = checkpoints[-self.keep:]
        to_delete = checkpoints[:-self.keep]

        print(f"\n🎯 Plan:")
        print(f"   Keep:   {len(to_keep)} latest checkpoints")
        print(f"   Delete: {len(to_delete)} old checkpoints")

        # Calculate space savings
        delete_size = sum(self.get_checkpoint_size(d) for _, d in to_delete)
        keep_size = sum(self.get_checkpoint_size(d) for _, d in to_keep)

        print(f"\n💾 Disk space:")
        print(f"   Current total: {delete_size + keep_size:.1f} GB")
        print(f"   After cleanup: {keep_size:.1f} GB")
        print(f"   Space freed:   {delete_size:.1f} GB")

        print(f"\n📝 Checkpoints to KEEP:")
        for step, ckpt_dir in to_keep:
            size = self.get_checkpoint_size(ckpt_dir)
            print(f"   ✅ checkpoint-{step:5d} ({size:.2f} GB)")

        print(f"\n🗑️  Checkpoints to DELETE:")
        for step, ckpt_dir in to_delete:
            size = self.get_checkpoint_size(ckpt_dir)
            print(f"   ❌ checkpoint-{step:5d} ({size:.2f} GB)")

        if self.dry_run:
            print("\n[DRY RUN - No changes made]")
            return True

        # Confirm deletion
        if not self.force:
            print(f"\n⚠️  About to delete {len(to_delete)} checkpoints ({delete_size:.1f} GB)")
            response = input("   Continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("   Cancelled")
                return False

        # Perform deletion
        print("\n🗑️  Deleting old checkpoints...")
        deleted = 0
        for step, ckpt_dir in to_delete:
            try:
                import shutil
                shutil.rmtree(ckpt_dir)
                print(f"   ✓ Deleted checkpoint-{step}")
                deleted += 1
            except Exception as e:
                print(f"   ✗ Failed to delete checkpoint-{step}: {e}")

        print(f"\n✅ Cleanup complete:")
        print(f"   Deleted: {deleted}/{len(to_delete)} checkpoints")
        print(f"   Freed:   ~{delete_size:.1f} GB")
        print(f"   Kept:    {len(to_keep)} latest checkpoints")

        return True

def main():
    parser = argparse.ArgumentParser(description="Safely cleanup old checkpoints")
    parser.add_argument('--keep', type=int, default=DEFAULT_KEEP,
                        help=f"Number of latest checkpoints to keep (default: {DEFAULT_KEEP})")
    parser.add_argument('--dry-run', action='store_true',
                        help="Show what would be deleted without deleting")
    parser.add_argument('--force', action='store_true',
                        help="Skip safety checks (use with caution!)")
    args = parser.parse_args()

    cleaner = SafeCheckpointCleanup(
        keep=args.keep,
        dry_run=args.dry_run,
        force=args.force
    )

    success = cleaner.cleanup()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
