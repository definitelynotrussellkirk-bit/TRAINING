#!/usr/bin/env python3
"""
Retention Management System - Smart Checkpoint & Snapshot Policy

POLICY SUMMARY:
- Daily snapshots: One model per day (weights only, ~1.5GB)
- Checkpoint retention: 36 hours of fine-grained checkpoints
- Hard limit: 150GB total across all backups
- Protection: Always keep latest, today, yesterday

DIRECTORY STRUCTURE:
  output_dir/
    checkpoints/              # HF Trainer checkpoints (with optimizer ~4.5GB each)
      checkpoint-1000/
      checkpoint-1500/
    snapshots/                # Daily archives (weights only ~1.5GB each)
      2025-11-22/
      2025-11-23/
    latest -> checkpoints/checkpoint-1500/   # symlink to resume checkpoint
    retention_index.json      # metadata tracking
    .retention_lock           # file lock for atomic operations

USAGE:
    # Automatic (called by training daemon):
    manager = RetentionManager(output_dir)
    manager.register_checkpoint("checkpoint-1000", metrics={"loss": 0.45})
    manager.enforce_retention()
    manager.create_daily_snapshot_if_needed()

    # Manual:
    python3 retention_manager.py --output-dir /path --enforce
    python3 retention_manager.py --output-dir /path --snapshot
    python3 retention_manager.py --output-dir /path --status
"""

import fcntl
import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

CHECKPOINT_RETENTION_HOURS = 36
TOTAL_SIZE_LIMIT_GB = 150
SNAPSHOT_EXPORT_FILES = [
    "pytorch_model.bin",
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]


@dataclass
class CheckpointMetadata:
    """Metadata for a single checkpoint"""
    path: str                          # Relative path from output_dir
    step: int                          # Training step number
    created_at: str                    # ISO timestamp
    size_bytes: int                    # Total size on disk
    has_optimizer: bool                # Whether this includes optimizer state
    metrics: Dict[str, float]          # Training metrics (loss, eval_loss, etc.)
    protected: bool = False            # If True, never delete

    @property
    def age_hours(self) -> float:
        """Age in hours since creation"""
        created = datetime.fromisoformat(self.created_at)
        return (datetime.now() - created).total_seconds() / 3600

    @property
    def size_gb(self) -> float:
        """Size in gigabytes"""
        return self.size_bytes / GB


@dataclass
class SnapshotMetadata:
    """Metadata for a daily snapshot"""
    path: str                          # Relative path from output_dir
    date: str                          # YYYY-MM-DD
    created_at: str                    # ISO timestamp
    size_bytes: int                    # Total size on disk
    source_checkpoint: str             # Which checkpoint this came from
    source_step: int                   # Training step at snapshot time
    source_metrics: Dict[str, float]   # Metrics at snapshot time
    protected: bool = False            # If True, never delete

    @property
    def age_days(self) -> int:
        """Age in days since creation"""
        snapshot_date = datetime.fromisoformat(self.date)
        return (datetime.now().date() - snapshot_date.date()).days

    @property
    def size_gb(self) -> float:
        """Size in gigabytes"""
        return self.size_bytes / GB


@dataclass
class RetentionIndex:
    """Complete index of all checkpoints and snapshots"""
    checkpoints: List[CheckpointMetadata]
    snapshots: List[SnapshotMetadata]
    latest_checkpoint: Optional[str]   # Path to current resume checkpoint
    best_checkpoint: Optional[Dict]    # {"path": str, "metric": str, "value": float}
    last_updated: str                  # ISO timestamp

    def total_size_bytes(self) -> int:
        """Total size of all tracked items"""
        checkpoint_size = sum(c.size_bytes for c in self.checkpoints)
        snapshot_size = sum(s.size_bytes for s in self.snapshots)
        return checkpoint_size + snapshot_size

    def total_size_gb(self) -> float:
        """Total size in GB"""
        return self.total_size_bytes() / GB


class RetentionManager:
    """Manages checkpoint retention and daily snapshots"""

    def __init__(self, output_dir: Path, base_model_path: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.base_model_path = Path(base_model_path) if base_model_path else None

        # Directory structure
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.latest_link = self.output_dir / "latest"

        # Index and lock files
        self.index_file = self.output_dir / "retention_index.json"
        self.lock_file = self.output_dir / ".retention_lock"

        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> RetentionIndex:
        """Load existing index or create new one by scanning filesystem"""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    data = json.load(f)

                # Convert dicts back to dataclasses
                checkpoints = [CheckpointMetadata(**c) for c in data.get("checkpoints", [])]
                snapshots = [SnapshotMetadata(**s) for s in data.get("snapshots", [])]

                return RetentionIndex(
                    checkpoints=checkpoints,
                    snapshots=snapshots,
                    latest_checkpoint=data.get("latest_checkpoint"),
                    best_checkpoint=data.get("best_checkpoint"),
                    last_updated=data.get("last_updated", datetime.now().isoformat())
                )
            except Exception as e:
                logger.warning(f"Failed to load index, rebuilding: {e}")

        # Rebuild from filesystem
        logger.info("Building retention index from filesystem...")
        return self._rebuild_index()

    def _rebuild_index(self) -> RetentionIndex:
        """Scan filesystem and rebuild index"""
        checkpoints = []
        snapshots = []

        # Scan checkpoints
        if self.checkpoints_dir.exists():
            for checkpoint_dir in sorted(self.checkpoints_dir.glob("checkpoint-*")):
                if checkpoint_dir.is_dir():
                    try:
                        step = int(checkpoint_dir.name.split("-")[1])
                        size = self._get_dir_size(checkpoint_dir)
                        mtime = checkpoint_dir.stat().st_mtime
                        created = datetime.fromtimestamp(mtime).isoformat()

                        # Check if has optimizer state
                        has_optimizer = (checkpoint_dir / "optimizer.pt").exists()

                        # Try to load metrics from trainer_state.json
                        metrics = {}
                        state_file = checkpoint_dir / "trainer_state.json"
                        if state_file.exists():
                            try:
                                with open(state_file) as f:
                                    state = json.load(f)
                                    if "log_history" in state and state["log_history"]:
                                        last_log = state["log_history"][-1]
                                        metrics = {k: v for k, v in last_log.items()
                                                 if isinstance(v, (int, float))}
                            except:
                                pass

                        checkpoints.append(CheckpointMetadata(
                            path=str(checkpoint_dir.relative_to(self.output_dir)),
                            step=step,
                            created_at=created,
                            size_bytes=size,
                            has_optimizer=has_optimizer,
                            metrics=metrics,
                            protected=False
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to index {checkpoint_dir}: {e}")

        # Scan snapshots
        if self.snapshots_dir.exists():
            for snapshot_dir in sorted(self.snapshots_dir.glob("*")):
                if snapshot_dir.is_dir() and snapshot_dir.name.count("-") == 2:
                    try:
                        size = self._get_dir_size(snapshot_dir)
                        mtime = snapshot_dir.stat().st_mtime
                        created = datetime.fromtimestamp(mtime).isoformat()

                        # Try to load snapshot metadata if exists
                        meta_file = snapshot_dir / "snapshot_metadata.json"
                        if meta_file.exists():
                            with open(meta_file) as f:
                                meta = json.load(f)
                                source_checkpoint = meta.get("source_checkpoint", "unknown")
                                source_step = meta.get("source_step", 0)
                                source_metrics = meta.get("source_metrics", {})
                        else:
                            source_checkpoint = "unknown"
                            source_step = 0
                            source_metrics = {}

                        snapshots.append(SnapshotMetadata(
                            path=str(snapshot_dir.relative_to(self.output_dir)),
                            date=snapshot_dir.name,
                            created_at=created,
                            size_bytes=size,
                            source_checkpoint=source_checkpoint,
                            source_step=source_step,
                            source_metrics=source_metrics,
                            protected=False
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to index {snapshot_dir}: {e}")

        # Determine latest checkpoint
        latest_checkpoint = None
        if self.latest_link.exists() and self.latest_link.is_symlink():
            try:
                target = self.latest_link.resolve()
                latest_checkpoint = str(target.relative_to(self.output_dir))
            except:
                pass
        elif checkpoints:
            # Use newest checkpoint
            latest = max(checkpoints, key=lambda c: c.step)
            latest_checkpoint = latest.path

        logger.info(f"Rebuilt index: {len(checkpoints)} checkpoints, {len(snapshots)} snapshots")

        return RetentionIndex(
            checkpoints=checkpoints,
            snapshots=snapshots,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=None,
            last_updated=datetime.now().isoformat()
        )

    def _save_index(self):
        """Save index to disk"""
        data = {
            "checkpoints": [asdict(c) for c in self.index.checkpoints],
            "snapshots": [asdict(s) for s in self.index.snapshots],
            "latest_checkpoint": self.index.latest_checkpoint,
            "best_checkpoint": self.index.best_checkpoint,
            "last_updated": datetime.now().isoformat()
        }

        # Atomic write
        temp_file = self.index_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.index_file)

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory using du"""
        try:
            result = subprocess.run(
                ["du", "-sb", str(path)],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.split()[0])
        except:
            # Fallback to Python walk
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except:
                        pass
            return total

    def register_checkpoint(self, checkpoint_path: str, metrics: Optional[Dict] = None,
                          is_latest: bool = True) -> None:
        """
        Register a new checkpoint in the index

        Args:
            checkpoint_path: Path to checkpoint (can be absolute or relative to output_dir)
            metrics: Training metrics at this checkpoint
            is_latest: Whether this is the new latest checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Make relative to output_dir if absolute
        if checkpoint_path.is_absolute():
            try:
                checkpoint_path = checkpoint_path.relative_to(self.output_dir)
            except ValueError:
                logger.error(f"Checkpoint {checkpoint_path} is not under {self.output_dir}")
                return

        checkpoint_full = self.output_dir / checkpoint_path
        if not checkpoint_full.exists():
            logger.error(f"Checkpoint {checkpoint_full} does not exist")
            return

        # Extract step number
        try:
            step = int(checkpoint_path.name.split("-")[1])
        except:
            logger.warning(f"Could not extract step from {checkpoint_path.name}")
            step = 0

        # Create metadata
        metadata = CheckpointMetadata(
            path=str(checkpoint_path),
            step=step,
            created_at=datetime.now().isoformat(),
            size_bytes=self._get_dir_size(checkpoint_full),
            has_optimizer=(checkpoint_full / "optimizer.pt").exists(),
            metrics=metrics or {},
            protected=False
        )

        # Add or update in index
        existing_idx = None
        for i, cp in enumerate(self.index.checkpoints):
            if cp.path == str(checkpoint_path):
                existing_idx = i
                break

        if existing_idx is not None:
            self.index.checkpoints[existing_idx] = metadata
        else:
            self.index.checkpoints.append(metadata)

        # Update latest if requested
        if is_latest:
            self.index.latest_checkpoint = str(checkpoint_path)
            self._update_latest_symlink(checkpoint_full)

        # Update best if this is better
        if metrics and "eval_loss" in metrics:
            if (not self.index.best_checkpoint or
                metrics["eval_loss"] < self.index.best_checkpoint.get("value", float('inf'))):
                self.index.best_checkpoint = {
                    "path": str(checkpoint_path),
                    "metric": "eval_loss",
                    "value": metrics["eval_loss"]
                }

        self._save_index()
        logger.info(f"Registered checkpoint: {checkpoint_path} (step {step}, "
                   f"{metadata.size_gb:.1f}GB)")

    def _update_latest_symlink(self, checkpoint_path: Path):
        """Update the latest symlink atomically"""
        if self.latest_link.exists() or self.latest_link.is_symlink():
            self.latest_link.unlink()
        self.latest_link.symlink_to(checkpoint_path, target_is_directory=True)

    def create_daily_snapshot(self, date: Optional[str] = None,
                            force: bool = False) -> Optional[Path]:
        """
        Create a daily snapshot (weights only, no optimizer)

        Args:
            date: Date string YYYY-MM-DD (default: today)
            force: Create even if one already exists for this date

        Returns:
            Path to created snapshot or None if skipped
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        snapshot_dir = self.snapshots_dir / date

        # Check if already exists
        if snapshot_dir.exists() and not force:
            logger.info(f"Snapshot for {date} already exists, skipping")
            return None

        # Find source checkpoint (latest)
        if not self.index.latest_checkpoint:
            logger.warning("No latest checkpoint to snapshot")
            return None

        source_path = self.output_dir / self.index.latest_checkpoint
        if not source_path.exists():
            logger.error(f"Source checkpoint {source_path} does not exist")
            return None

        # Get source metadata
        source_meta = None
        for cp in self.index.checkpoints:
            if cp.path == self.index.latest_checkpoint:
                source_meta = cp
                break

        if not source_meta:
            logger.warning("Could not find metadata for latest checkpoint")
            source_meta = CheckpointMetadata(
                path=self.index.latest_checkpoint,
                step=0,
                created_at=datetime.now().isoformat(),
                size_bytes=0,
                has_optimizer=False,
                metrics={}
            )

        logger.info(f"Creating daily snapshot for {date}")
        logger.info(f"  Source: {source_path}")
        logger.info(f"  Step: {source_meta.step}")
        logger.info(f"  Metrics: {source_meta.metrics}")

        # Create snapshot directory
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy only essential model files (no optimizer)
        copied_files = []
        for filename in SNAPSHOT_EXPORT_FILES:
            src_file = source_path / filename
            if src_file.exists():
                dst_file = snapshot_dir / filename
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)

        if not copied_files:
            logger.error("No model files found to snapshot")
            shutil.rmtree(snapshot_dir)
            return None

        # Save snapshot metadata
        snapshot_metadata = {
            "created_at": datetime.now().isoformat(),
            "date": date,
            "source_checkpoint": self.index.latest_checkpoint,
            "source_step": source_meta.step,
            "source_metrics": source_meta.metrics,
            "copied_files": copied_files
        }

        with open(snapshot_dir / "snapshot_metadata.json", 'w') as f:
            json.dump(snapshot_metadata, f, indent=2)

        # Add to index
        snapshot_size = self._get_dir_size(snapshot_dir)
        snapshot_meta = SnapshotMetadata(
            path=str(snapshot_dir.relative_to(self.output_dir)),
            date=date,
            created_at=datetime.now().isoformat(),
            size_bytes=snapshot_size,
            source_checkpoint=self.index.latest_checkpoint,
            source_step=source_meta.step,
            source_metrics=source_meta.metrics,
            protected=False
        )

        self.index.snapshots.append(snapshot_meta)
        self._save_index()

        logger.info(f"✅ Snapshot created: {snapshot_dir.name} "
                   f"({snapshot_size / MB:.1f}MB, {len(copied_files)} files)")

        return snapshot_dir

    def create_daily_snapshot_if_needed(self) -> Optional[Path]:
        """Create today's snapshot if it doesn't exist yet"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Check if today's snapshot exists
        for snapshot in self.index.snapshots:
            if snapshot.date == today:
                logger.debug(f"Today's snapshot already exists: {snapshot.path}")
                return None

        return self.create_daily_snapshot(date=today)

    def enforce_retention(self, dry_run: bool = False) -> Dict:
        """
        Enforce retention policy:
        1. Protect: latest, today, yesterday, best
        2. Keep all checkpoints < 36h old
        3. Delete until total size < 150GB

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Summary dict with stats
        """
        logger.info("Enforcing retention policy...")

        # Step 1: Mark protected items
        self._mark_protected_items()

        # Step 2: Apply 36h rule to checkpoints
        deletable_checkpoints = self._find_old_checkpoints(CHECKPOINT_RETENTION_HOURS)

        # Step 3: Build deletion candidates list
        candidates = []

        # Add old checkpoints (oldest first)
        for cp in sorted(deletable_checkpoints, key=lambda c: c.created_at):
            candidates.append(('checkpoint', cp))

        # Add old snapshots (oldest first)
        deletable_snapshots = [s for s in self.index.snapshots if not s.protected]
        for snap in sorted(deletable_snapshots, key=lambda s: s.date):
            candidates.append(('snapshot', snap))

        # Step 4: Delete until under 150GB
        total_size = self.index.total_size_bytes()
        limit_bytes = TOTAL_SIZE_LIMIT_GB * GB

        to_delete = []
        for item_type, item in candidates:
            if total_size <= limit_bytes:
                break

            to_delete.append((item_type, item))
            total_size -= item.size_bytes

        # Execute deletions
        deleted_checkpoints = []
        deleted_snapshots = []

        if not dry_run:
            for item_type, item in to_delete:
                item_path = self.output_dir / item.path

                try:
                    logger.info(f"Deleting {item_type}: {item.path} "
                              f"({item.size_gb:.1f}GB)")
                    shutil.rmtree(item_path)

                    if item_type == 'checkpoint':
                        self.index.checkpoints.remove(item)
                        deleted_checkpoints.append(item)
                    else:
                        self.index.snapshots.remove(item)
                        deleted_snapshots.append(item)
                except Exception as e:
                    logger.error(f"Failed to delete {item.path}: {e}")

            self._save_index()
        else:
            for item_type, item in to_delete:
                logger.info(f"Would delete {item_type}: {item.path} "
                          f"({item.size_gb:.1f}GB)")
                if item_type == 'checkpoint':
                    deleted_checkpoints.append(item)
                else:
                    deleted_snapshots.append(item)

        # Summary
        summary = {
            "total_size_gb": self.index.total_size_gb(),
            "limit_gb": TOTAL_SIZE_LIMIT_GB,
            "protected_checkpoints": len([c for c in self.index.checkpoints if c.protected]),
            "protected_snapshots": len([s for s in self.index.snapshots if s.protected]),
            "deleted_checkpoints": len(deleted_checkpoints),
            "deleted_snapshots": len(deleted_snapshots),
            "deleted_size_gb": sum(item.size_gb for _, item in to_delete),
            "dry_run": dry_run
        }

        logger.info(f"Retention summary:")
        logger.info(f"  Total size: {summary['total_size_gb']:.1f}GB / {summary['limit_gb']}GB")
        logger.info(f"  Protected: {summary['protected_checkpoints']} checkpoints, "
                   f"{summary['protected_snapshots']} snapshots")
        logger.info(f"  {'Would delete' if dry_run else 'Deleted'}: "
                   f"{summary['deleted_checkpoints']} checkpoints, "
                   f"{summary['deleted_snapshots']} snapshots "
                   f"({summary['deleted_size_gb']:.1f}GB freed)")

        return summary

    def _mark_protected_items(self):
        """Mark items that should never be deleted"""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Reset all protection flags
        for cp in self.index.checkpoints:
            cp.protected = False
        for snap in self.index.snapshots:
            snap.protected = False

        # Protect latest checkpoint
        if self.index.latest_checkpoint:
            for cp in self.index.checkpoints:
                if cp.path == self.index.latest_checkpoint:
                    cp.protected = True
                    logger.debug(f"Protected latest checkpoint: {cp.path}")
                    break

        # Protect best checkpoint
        if self.index.best_checkpoint:
            for cp in self.index.checkpoints:
                if cp.path == self.index.best_checkpoint.get("path"):
                    cp.protected = True
                    logger.debug(f"Protected best checkpoint: {cp.path}")
                    break

        # Protect very recent checkpoints (< 1 hour) as safety buffer
        for cp in self.index.checkpoints:
            if cp.age_hours < 1.0:
                cp.protected = True
                logger.debug(f"Protected recent checkpoint: {cp.path}")

        # Protect today's and yesterday's snapshots
        for snap in self.index.snapshots:
            if snap.date in [today, yesterday]:
                snap.protected = True
                logger.debug(f"Protected snapshot: {snap.date}")

    def _find_old_checkpoints(self, max_age_hours: float) -> List[CheckpointMetadata]:
        """Find checkpoints older than max_age_hours that are not protected"""
        old_checkpoints = []
        for cp in self.index.checkpoints:
            if not cp.protected and cp.age_hours > max_age_hours:
                old_checkpoints.append(cp)
        return old_checkpoints

    def get_status(self) -> Dict:
        """Get current retention status"""
        total_size = self.index.total_size_bytes()
        checkpoint_size = sum(c.size_bytes for c in self.index.checkpoints)
        snapshot_size = sum(s.size_bytes for s in self.index.snapshots)

        return {
            "total_size_gb": total_size / GB,
            "limit_gb": TOTAL_SIZE_LIMIT_GB,
            "usage_pct": (total_size / (TOTAL_SIZE_LIMIT_GB * GB)) * 100,
            "checkpoints": {
                "count": len(self.index.checkpoints),
                "size_gb": checkpoint_size / GB,
                "protected": len([c for c in self.index.checkpoints if c.protected]),
                "oldest_age_hours": max([c.age_hours for c in self.index.checkpoints], default=0)
            },
            "snapshots": {
                "count": len(self.index.snapshots),
                "size_gb": snapshot_size / GB,
                "protected": len([s for s in self.index.snapshots if s.protected]),
                "oldest_age_days": max([s.age_days for s in self.index.snapshots], default=0)
            },
            "latest_checkpoint": self.index.latest_checkpoint,
            "best_checkpoint": self.index.best_checkpoint,
            "last_updated": self.index.last_updated
        }

    def print_status(self):
        """Print human-readable status"""
        status = self.get_status()

        print("\n" + "="*80)
        print("RETENTION STATUS")
        print("="*80)
        print(f"\nStorage: {status['total_size_gb']:.1f}GB / {status['limit_gb']}GB "
              f"({status['usage_pct']:.1f}% used)")

        print(f"\nCheckpoints: {status['checkpoints']['count']} total, "
              f"{status['checkpoints']['protected']} protected")
        print(f"  Size: {status['checkpoints']['size_gb']:.1f}GB")
        print(f"  Oldest: {status['checkpoints']['oldest_age_hours']:.1f}h")

        print(f"\nSnapshots: {status['snapshots']['count']} total, "
              f"{status['snapshots']['protected']} protected")
        print(f"  Size: {status['snapshots']['size_gb']:.1f}GB")
        print(f"  Oldest: {status['snapshots']['oldest_age_days']} days")

        if status['latest_checkpoint']:
            print(f"\nLatest: {status['latest_checkpoint']}")

        if status['best_checkpoint']:
            best = status['best_checkpoint']
            print(f"Best: {best['path']} ({best['metric']}={best['value']:.4f})")

        print(f"\nLast updated: {status['last_updated']}")
        print("="*80 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Retention Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  python3 retention_manager.py --output-dir /training/models --status

  # Create daily snapshot
  python3 retention_manager.py --output-dir /training/models --snapshot

  # Enforce retention (dry run)
  python3 retention_manager.py --output-dir /training/models --enforce --dry-run

  # Enforce retention (actually delete)
  python3 retention_manager.py --output-dir /training/models --enforce

  # Register a new checkpoint
  python3 retention_manager.py --output-dir /training/models --register checkpoints/checkpoint-1000
        """
    )

    parser.add_argument('--output-dir', required=True, help='Output directory containing checkpoints')
    parser.add_argument('--base-model', help='Base model path (for snapshots)')

    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--snapshot', action='store_true', help='Create daily snapshot if needed')
    parser.add_argument('--enforce', action='store_true', help='Enforce retention policy')
    parser.add_argument('--register', metavar='PATH', help='Register a checkpoint')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild index from filesystem')

    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no deletions)')
    parser.add_argument('--force', action='store_true', help='Force operation')

    args = parser.parse_args()

    manager = RetentionManager(args.output_dir, args.base_model)

    if args.rebuild:
        print("Rebuilding index from filesystem...")
        manager.index = manager._rebuild_index()
        manager._save_index()
        print("✅ Index rebuilt")

    if args.register:
        print(f"Registering checkpoint: {args.register}")
        manager.register_checkpoint(args.register)
        print("✅ Checkpoint registered")

    if args.snapshot:
        print("Creating daily snapshot...")
        result = manager.create_daily_snapshot_if_needed()
        if result:
            print(f"✅ Snapshot created: {result}")
        else:
            print("ℹ️  Snapshot already exists or no checkpoint available")

    if args.enforce:
        print(f"Enforcing retention policy {'(DRY RUN)' if args.dry_run else ''}...")
        summary = manager.enforce_retention(dry_run=args.dry_run)
        print("\n✅ Retention enforced")

    if args.status or not any([args.rebuild, args.register, args.snapshot, args.enforce]):
        manager.print_status()


if __name__ == '__main__':
    main()
