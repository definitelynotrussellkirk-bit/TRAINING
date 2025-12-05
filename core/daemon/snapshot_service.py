#!/usr/bin/env python3
"""
Snapshot Service - Create and verify model snapshots.

This module handles checkpoint snapshots:
- Daily snapshot creation
- Snapshot verification (integrity checks)
- Latest checkpoint copying
- Essential file preservation

Usage:
    from daemon.snapshot_service import SnapshotService, SnapshotConfig

    service = SnapshotService(SnapshotConfig(
        checkpoints_dir=Path("models/current_model"),
        snapshots_dir=Path("snapshots"),
        snapshot_time="02:00"
    ))

    if service.should_create_snapshot():
        service.create_snapshot()
"""

import json
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class SnapshotConfig:
    """
    Configuration for snapshot service.

    Attributes:
        checkpoints_dir: Directory containing model checkpoints
        snapshots_dir: Directory to store snapshots
        snapshot_time: Time to create daily snapshot (HH:MM format)
        essential_files: List of essential files to copy
    """
    checkpoints_dir: Path
    snapshots_dir: Path
    snapshot_time: str = "02:00"
    essential_files: List[str] = field(default_factory=lambda: [
        "config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "added_tokens.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "pytorch_model.bin",
    ])


@dataclass
class SnapshotResult:
    """
    Result of snapshot creation.

    Attributes:
        success: Whether snapshot was created successfully
        snapshot_path: Path to created snapshot
        size_bytes: Size of snapshot in bytes
        checkpoint_name: Name of checkpoint that was snapshotted
        error: Error message if failed
    """
    success: bool
    snapshot_path: Optional[Path] = None
    size_bytes: int = 0
    checkpoint_name: Optional[str] = None
    error: Optional[str] = None


class SnapshotService:
    """
    Creates and verifies model snapshots.

    Handles:
    - Checking if snapshot should be created (daily schedule)
    - Creating snapshots with latest checkpoint
    - Verifying snapshot integrity
    - Copying essential model files

    Example:
        config = SnapshotConfig(
            checkpoints_dir=Path("models/current_model"),
            snapshots_dir=Path("snapshots"),
            snapshot_time="02:00"
        )

        service = SnapshotService(config)

        if service.should_create_snapshot():
            result = service.create_snapshot()
            if result.success:
                print(f"Snapshot created: {result.snapshot_path}")
    """

    def __init__(self, config: SnapshotConfig):
        """
        Initialize snapshot service.

        Args:
            config: Snapshot configuration
        """
        self.config = config
        self._last_snapshot_date: Optional[date] = None

        # Ensure snapshots directory exists
        self.config.snapshots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_snapshot_date(self) -> Optional[date]:
        """Get date of last snapshot."""
        return self._last_snapshot_date

    @last_snapshot_date.setter
    def last_snapshot_date(self, value: date) -> None:
        """Set date of last snapshot."""
        self._last_snapshot_date = value

    def should_create_snapshot(self) -> bool:
        """
        Check if a snapshot should be created.

        Returns:
            True if snapshot should be created (daily, after configured time)
        """
        today = datetime.now().date()

        # Already created snapshot today?
        if self._last_snapshot_date == today:
            return False

        # Check if we're past snapshot time
        try:
            snapshot_time = datetime.strptime(self.config.snapshot_time, "%H:%M").time()
            current_time = datetime.now().time()

            if current_time >= snapshot_time:
                return True
        except ValueError as e:
            logger.warning(f"Invalid snapshot time format: {e}")

        return False

    def verify_snapshot(self, snapshot_dir: Path) -> bool:
        """
        Verify snapshot integrity.

        Checks:
        - config.json OR adapter_config.json exists and is valid JSON
        - Model weights exist:
          - Full model: model.safetensors or pytorch_model.bin
          - PEFT adapter: adapter_model.safetensors (with adapter_config.json)
        - Model/adapter file is not empty

        Args:
            snapshot_dir: Path to snapshot directory

        Returns:
            True if snapshot is valid
        """
        try:
            # Check for config file (model config or adapter config)
            config_file = snapshot_dir / "config.json"
            adapter_config_file = snapshot_dir / "adapter_config.json"

            has_model_config = config_file.exists()
            has_adapter_config = adapter_config_file.exists()

            if not has_model_config and not has_adapter_config:
                logger.debug(f"Missing config.json and adapter_config.json in {snapshot_dir}")
                return False

            # Check for model weights - support both full models and PEFT adapters
            model_file = None
            is_peft = False

            # First check for PEFT adapter (preferred for QLoRA checkpoints)
            adapter_file = snapshot_dir / "adapter_model.safetensors"
            if adapter_file.exists() and has_adapter_config:
                model_file = adapter_file
                is_peft = True

            # Then check for full model weights
            if model_file is None:
                model_file = snapshot_dir / "model.safetensors"
                if not model_file.exists():
                    model_file = snapshot_dir / "pytorch_model.bin"

            # If not found at root, check in checkpoint subdirectory
            if not model_file.exists():
                checkpoints = list(snapshot_dir.glob("checkpoint-*"))
                if checkpoints:
                    ckpt = checkpoints[0]
                    # Check for PEFT adapter first
                    adapter_file = ckpt / "adapter_model.safetensors"
                    ckpt_adapter_config = ckpt / "adapter_config.json"
                    if adapter_file.exists() and ckpt_adapter_config.exists():
                        model_file = adapter_file
                        is_peft = True
                    else:
                        # Fall back to full model
                        model_file = ckpt / "model.safetensors"
                        if not model_file.exists():
                            model_file = ckpt / "pytorch_model.bin"

            if not model_file.exists():
                logger.debug(f"Missing model weights in {snapshot_dir}")
                return False

            # Verify model file is not empty
            if model_file.stat().st_size == 0:
                logger.debug(f"Empty model file: {model_file}")
                return False

            # Verify config is valid JSON
            config_to_check = adapter_config_file if is_peft and has_adapter_config else config_file
            if config_to_check.exists():
                with open(config_to_check) as f:
                    json.load(f)

            logger.debug(f"Snapshot verified: {snapshot_dir} (PEFT={is_peft})")
            return True

        except Exception as e:
            logger.error(f"Snapshot verification failed: {e}")
            return False

    def create_snapshot(self, force: bool = False) -> SnapshotResult:
        """
        Create a snapshot of the current model.

        Args:
            force: Create snapshot even if one exists for today

        Returns:
            SnapshotResult with details
        """
        today = datetime.now().date()
        snapshot_dir = self.config.snapshots_dir / today.strftime("%Y-%m-%d")

        # Check for existing snapshot
        if snapshot_dir.exists() and not force:
            if self.verify_snapshot(snapshot_dir):
                logger.info(f"Snapshot already exists and verified: {snapshot_dir}")
                self._last_snapshot_date = today
                return SnapshotResult(
                    success=True,
                    snapshot_path=snapshot_dir,
                    size_bytes=self._get_dir_size(snapshot_dir)
                )
            else:
                logger.warning("Existing snapshot corrupt - recreating")
                shutil.rmtree(snapshot_dir)

        logger.info(f"Creating snapshot: {snapshot_dir}")

        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Find latest checkpoint
            checkpoints = sorted([
                d for d in self.config.checkpoints_dir.glob("checkpoint-*")
                if d.is_dir()
            ])

            checkpoint_name = None
            if checkpoints:
                latest = checkpoints[-1]
                checkpoint_name = latest.name
                logger.info(f"Copying checkpoint: {checkpoint_name}")
                shutil.copytree(latest, snapshot_dir / checkpoint_name)

            # Copy essential files from root
            copied_files = []
            for filename in self.config.essential_files:
                src = self.config.checkpoints_dir / filename
                if src.exists():
                    shutil.copy2(src, snapshot_dir / filename)
                    copied_files.append(filename)

            if copied_files:
                logger.info(f"Copied {len(copied_files)} essential files")

            # Verify snapshot
            if not self.verify_snapshot(snapshot_dir):
                logger.error("Snapshot verification failed after creation")
                shutil.rmtree(snapshot_dir)
                return SnapshotResult(
                    success=False,
                    error="Verification failed after creation"
                )

            self._last_snapshot_date = today
            size = self._get_dir_size(snapshot_dir)

            logger.info(f"Snapshot created: {snapshot_dir} ({size / 1024 / 1024:.1f} MB)")

            return SnapshotResult(
                success=True,
                snapshot_path=snapshot_dir,
                size_bytes=size,
                checkpoint_name=checkpoint_name
            )

        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir, ignore_errors=True)
            return SnapshotResult(success=False, error=str(e))

    def get_latest_snapshot(self) -> Optional[Path]:
        """Get path to most recent snapshot."""
        snapshots = sorted(self.config.snapshots_dir.glob("20*"))
        return snapshots[-1] if snapshots else None

    def list_snapshots(self) -> List[dict]:
        """List all snapshots with metadata."""
        snapshots = []
        for snap_dir in sorted(self.config.snapshots_dir.glob("20*")):
            if snap_dir.is_dir():
                snapshots.append({
                    "date": snap_dir.name,
                    "path": str(snap_dir),
                    "size_mb": self._get_dir_size(snap_dir) / 1024 / 1024,
                    "valid": self.verify_snapshot(snap_dir)
                })
        return snapshots

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


if __name__ == "__main__":
    # Quick test
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock checkpoint structure
        checkpoints_dir = tmpdir / "current_model"
        checkpoints_dir.mkdir()

        # Create a fake checkpoint
        ckpt = checkpoints_dir / "checkpoint-1000"
        ckpt.mkdir()
        (ckpt / "config.json").write_text('{"model_type": "qwen2"}')
        (ckpt / "model.safetensors").write_text('fake model weights')

        # Create config at root
        (checkpoints_dir / "config.json").write_text('{"model_type": "qwen2"}')
        (checkpoints_dir / "tokenizer.json").write_text('{}')

        # Test service
        config = SnapshotConfig(
            checkpoints_dir=checkpoints_dir,
            snapshots_dir=tmpdir / "snapshots",
            snapshot_time="00:00"  # Always past midnight
        )

        service = SnapshotService(config)

        print(f"Should create snapshot: {service.should_create_snapshot()}")

        result = service.create_snapshot()
        print(f"\nSnapshot result:")
        print(f"  Success: {result.success}")
        print(f"  Path: {result.snapshot_path}")
        print(f"  Size: {result.size_bytes} bytes")
        print(f"  Checkpoint: {result.checkpoint_name}")

        # List snapshots
        snapshots = service.list_snapshots()
        print(f"\nSnapshots: {snapshots}")

        # Verify
        if result.snapshot_path:
            print(f"\nVerify: {service.verify_snapshot(result.snapshot_path)}")

        print("\nSnapshotService ready for use!")
