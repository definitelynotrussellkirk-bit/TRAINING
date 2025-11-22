#!/usr/bin/env python3
"""
Tests for RetentionManager

Tests the retention policy system including:
- Checkpoint registration
- Daily snapshot creation
- 36h retention rule
- 150GB size limit enforcement
- Protection rules (latest, today, yesterday, best)
"""

import json
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "management"))

from retention_manager import (
    RetentionManager,
    CheckpointMetadata,
    SnapshotMetadata,
    RetentionIndex,
    CHECKPOINT_RETENTION_HOURS,
    TOTAL_SIZE_LIMIT_GB,
    GB
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def manager(temp_dir):
    """Create RetentionManager with temp directory"""
    return RetentionManager(output_dir=temp_dir)


def create_mock_checkpoint(output_dir: Path, step: int, size_mb: int = 100,
                          with_optimizer: bool = True, age_hours: float = 0) -> Path:
    """Create a mock checkpoint directory for testing"""
    checkpoint_dir = output_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create mock files
    (checkpoint_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (checkpoint_dir / "pytorch_model.bin").write_bytes(b"0" * (size_mb * 1024 * 1024))

    if with_optimizer:
        (checkpoint_dir / "optimizer.pt").write_bytes(b"0" * (size_mb * 1024 * 1024))

    # Create trainer_state.json with metrics
    trainer_state = {
        "global_step": step,
        "log_history": [
            {
                "step": step,
                "loss": 0.5 - (step * 0.0001),
                "eval_loss": 0.6 - (step * 0.0001)
            }
        ]
    }
    (checkpoint_dir / "trainer_state.json").write_text(json.dumps(trainer_state))

    # Set modification time if age specified
    if age_hours > 0:
        mtime = time.time() - (age_hours * 3600)
        for file in checkpoint_dir.rglob("*"):
            if file.is_file():
                file.touch()
                # Note: can't easily set mtime in tests, so we'll rely on created_at in metadata

    return checkpoint_dir


def test_initialization(manager, temp_dir):
    """Test RetentionManager initialization"""
    assert manager.output_dir == temp_dir
    assert manager.checkpoints_dir == temp_dir / "checkpoints"
    assert manager.snapshots_dir == temp_dir / "snapshots"
    assert manager.checkpoints_dir.exists()
    assert manager.snapshots_dir.exists()
    assert isinstance(manager.index, RetentionIndex)


def test_register_checkpoint(manager, temp_dir):
    """Test checkpoint registration"""
    # Create mock checkpoint
    checkpoint = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)

    # Register it
    manager.register_checkpoint(
        checkpoint_path=checkpoint,
        metrics={"loss": 0.45, "eval_loss": 0.52},
        is_latest=True
    )

    # Verify it's in index
    assert len(manager.index.checkpoints) == 1
    cp = manager.index.checkpoints[0]
    assert cp.step == 1000
    assert cp.has_optimizer is True
    assert cp.metrics["loss"] == 0.45
    assert cp.metrics["eval_loss"] == 0.52

    # Verify latest is updated
    assert manager.index.latest_checkpoint == cp.path

    # Verify symlink created
    assert manager.latest_link.exists()
    assert manager.latest_link.is_symlink()


def test_register_multiple_checkpoints(manager, temp_dir):
    """Test registering multiple checkpoints"""
    # Create and register 3 checkpoints
    for step in [1000, 1500, 2000]:
        checkpoint = create_mock_checkpoint(temp_dir, step=step, size_mb=50)
        manager.register_checkpoint(
            checkpoint_path=checkpoint,
            metrics={"loss": 0.5 - step*0.0001, "eval_loss": 0.6 - step*0.0001},
            is_latest=(step == 2000)
        )

    # Verify all are registered
    assert len(manager.index.checkpoints) == 3

    # Verify latest is correct
    assert manager.index.latest_checkpoint == "checkpoints/checkpoint-2000"

    # Verify best is tracked (lowest eval_loss)
    assert manager.index.best_checkpoint is not None
    assert "checkpoint-2000" in manager.index.best_checkpoint["path"]


def test_create_daily_snapshot(manager, temp_dir):
    """Test daily snapshot creation"""
    # Create and register a checkpoint
    checkpoint = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)
    manager.register_checkpoint(checkpoint_path=checkpoint, metrics={"loss": 0.45}, is_latest=True)

    # Create snapshot
    today = datetime.now().strftime("%Y-%m-%d")
    snapshot_dir = manager.create_daily_snapshot(date=today)

    # Verify snapshot created
    assert snapshot_dir is not None
    assert snapshot_dir.exists()
    assert (snapshot_dir / "config.json").exists()
    assert (snapshot_dir / "pytorch_model.bin").exists()
    assert (snapshot_dir / "snapshot_metadata.json").exists()

    # Verify NOT copied optimizer
    assert not (snapshot_dir / "optimizer.pt").exists()

    # Verify metadata
    with open(snapshot_dir / "snapshot_metadata.json") as f:
        meta = json.load(f)
    assert meta["date"] == today
    assert meta["source_step"] == 1000

    # Verify in index
    assert len(manager.index.snapshots) == 1
    snap = manager.index.snapshots[0]
    assert snap.date == today
    assert snap.source_step == 1000


def test_create_daily_snapshot_if_needed(manager, temp_dir):
    """Test create_daily_snapshot_if_needed only creates once"""
    # Create checkpoint
    checkpoint = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)
    manager.register_checkpoint(checkpoint_path=checkpoint, metrics={"loss": 0.45}, is_latest=True)

    # First call should create
    result1 = manager.create_daily_snapshot_if_needed()
    assert result1 is not None

    # Second call should skip
    result2 = manager.create_daily_snapshot_if_needed()
    assert result2 is None

    # Should only have one snapshot
    assert len(manager.index.snapshots) == 1


def test_protection_rules(manager, temp_dir):
    """Test that protected items are marked correctly"""
    # Create checkpoints
    for step in [1000, 1500, 2000]:
        checkpoint = create_mock_checkpoint(temp_dir, step=step, size_mb=50)
        manager.register_checkpoint(
            checkpoint_path=checkpoint,
            metrics={"loss": 0.5, "eval_loss": 0.6 - step*0.0001},
            is_latest=(step == 2000)
        )

    # Create snapshots for today and yesterday
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    for date in [yesterday, today]:
        snapshot_dir = manager.create_daily_snapshot(date=date)

    # Mark protected
    manager._mark_protected_items()

    # Latest checkpoint should be protected
    latest_cp = [c for c in manager.index.checkpoints if c.path == manager.index.latest_checkpoint][0]
    assert latest_cp.protected is True

    # Best checkpoint should be protected
    best_path = manager.index.best_checkpoint["path"]
    best_cp = [c for c in manager.index.checkpoints if c.path == best_path][0]
    assert best_cp.protected is True

    # Today and yesterday snapshots should be protected
    protected_snaps = [s for s in manager.index.snapshots if s.protected]
    assert len(protected_snaps) == 2
    assert set(s.date for s in protected_snaps) == {today, yesterday}


def test_36h_checkpoint_retention(manager, temp_dir):
    """Test that checkpoints older than 36h are deleted"""
    # Create checkpoints with different ages (simulate by creating_at timestamp)
    checkpoints = []

    # Old checkpoint (40h old) - should be deleted
    old_cp = create_mock_checkpoint(temp_dir, step=500, size_mb=50)
    checkpoints.append(old_cp)

    # Recent checkpoint (30h old) - should be kept
    recent_cp = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)
    checkpoints.append(recent_cp)

    # Latest checkpoint (1h old) - should be kept
    latest_cp = create_mock_checkpoint(temp_dir, step=1500, size_mb=50)
    checkpoints.append(latest_cp)

    # Register with manual created_at timestamps
    now = datetime.now()

    # Old
    manager.register_checkpoint(checkpoint_path=checkpoints[0], is_latest=False)
    manager.index.checkpoints[-1].created_at = (now - timedelta(hours=40)).isoformat()

    # Recent
    manager.register_checkpoint(checkpoint_path=checkpoints[1], is_latest=False)
    manager.index.checkpoints[-1].created_at = (now - timedelta(hours=30)).isoformat()

    # Latest
    manager.register_checkpoint(checkpoint_path=checkpoints[2], is_latest=True)

    # Enforce retention
    summary = manager.enforce_retention(dry_run=False)

    # Old checkpoint should be deleted
    assert summary["deleted_checkpoints"] == 1

    # Should have 2 remaining
    assert len(manager.index.checkpoints) == 2

    # Verify the old one is gone
    assert not any(c.step == 500 for c in manager.index.checkpoints)


def test_150gb_size_limit(manager, temp_dir):
    """Test that total size is limited to 150GB"""
    # Create many large checkpoints to exceed 150GB
    # Use small test limit for faster testing
    original_limit = TOTAL_SIZE_LIMIT_GB

    # Temporarily patch the limit to 1GB for testing
    import retention_manager
    retention_manager.TOTAL_SIZE_LIMIT_GB = 0.5  # 500MB limit

    # Create checkpoints totaling ~1GB
    for step in range(1000, 1600, 100):
        checkpoint = create_mock_checkpoint(temp_dir, step=step, size_mb=100)  # 200MB each (model + optimizer)
        manager.register_checkpoint(
            checkpoint_path=checkpoint,
            metrics={"loss": 0.5},
            is_latest=(step == 1500)
        )

    # Total = 6 checkpoints * 200MB = 1200MB > 500MB limit

    # Enforce retention
    summary = manager.enforce_retention(dry_run=False)

    # Should have deleted some checkpoints
    assert summary["deleted_checkpoints"] > 0

    # Total size should now be under limit
    assert manager.index.total_size_gb() <= 0.5

    # Restore original limit
    retention_manager.TOTAL_SIZE_LIMIT_GB = original_limit


def test_rebuild_index(manager, temp_dir):
    """Test rebuilding index from filesystem"""
    # Create some checkpoints directly on filesystem
    for step in [1000, 1500, 2000]:
        create_mock_checkpoint(temp_dir, step=step, size_mb=50)

    # Rebuild index
    new_index = manager._rebuild_index()

    # Should find all 3 checkpoints
    assert len(new_index.checkpoints) == 3

    # Should set latest to newest
    assert "checkpoint-2000" in new_index.latest_checkpoint


def test_get_status(manager, temp_dir):
    """Test status reporting"""
    # Create some checkpoints and snapshots
    for step in [1000, 1500]:
        checkpoint = create_mock_checkpoint(temp_dir, step=step, size_mb=50)
        manager.register_checkpoint(checkpoint_path=checkpoint, is_latest=(step==1500))

    manager.create_daily_snapshot_if_needed()

    # Get status
    status = manager.get_status()

    # Verify structure
    assert "total_size_gb" in status
    assert "checkpoints" in status
    assert "snapshots" in status
    assert status["checkpoints"]["count"] == 2
    assert status["snapshots"]["count"] == 1


def test_index_persistence(manager, temp_dir):
    """Test that index persists across manager instances"""
    # Register checkpoint
    checkpoint = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)
    manager.register_checkpoint(checkpoint_path=checkpoint, is_latest=True)

    # Create new manager instance
    manager2 = RetentionManager(output_dir=temp_dir)

    # Should load previous index
    assert len(manager2.index.checkpoints) == 1
    assert manager2.index.checkpoints[0].step == 1000


def test_dry_run_mode(manager, temp_dir):
    """Test that dry_run doesn't actually delete"""
    # Create old checkpoint
    checkpoint = create_mock_checkpoint(temp_dir, step=500, size_mb=50)
    manager.register_checkpoint(checkpoint_path=checkpoint, is_latest=False)

    # Set as very old
    manager.index.checkpoints[0].created_at = (
        datetime.now() - timedelta(hours=40)
    ).isoformat()

    # Create newer checkpoint as latest
    checkpoint2 = create_mock_checkpoint(temp_dir, step=1000, size_mb=50)
    manager.register_checkpoint(checkpoint_path=checkpoint2, is_latest=True)

    # Dry run
    summary = manager.enforce_retention(dry_run=True)

    # Should report what would be deleted
    assert summary["deleted_checkpoints"] == 1

    # But shouldn't actually delete
    assert len(manager.index.checkpoints) == 2
    assert checkpoint.exists()


def test_snapshot_without_optimizer(manager, temp_dir):
    """Test that snapshots don't include optimizer state"""
    # Create checkpoint with optimizer
    checkpoint = create_mock_checkpoint(temp_dir, step=1000, size_mb=100, with_optimizer=True)
    manager.register_checkpoint(checkpoint_path=checkpoint, is_latest=True)

    # Checkpoint should have optimizer
    assert (checkpoint / "optimizer.pt").exists()

    # Create snapshot
    snapshot_dir = manager.create_daily_snapshot()

    # Snapshot should NOT have optimizer
    assert not (snapshot_dir / "optimizer.pt").exists()

    # Snapshot should be smaller than checkpoint
    snap_meta = manager.index.snapshots[0]
    cp_meta = manager.index.checkpoints[0]
    assert snap_meta.size_bytes < cp_meta.size_bytes


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, "-v"])
