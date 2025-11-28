"""
Tests for the Storage Resolver system.

Tests cover:
- Storage types (zones, kinds, handles)
- Handle creation and parsing
- Path resolution
- Zone availability
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from core.storage_types import (
    StorageZone,
    StorageKind,
    StorageHandle,
    checkpoint_handle,
    snapshot_handle,
    dataset_handle,
    queue_handle,
)
from vault.storage_resolver import (
    StorageResolver,
    StorageResolverError,
    ZoneNotAvailable,
    KindNotConfigured,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_zones_config():
    """Create a sample storage_zones.json config."""
    return {
        "schema_version": 1,
        "zones": {
            "hot": {
                "description": "Fast local storage",
                "devices": ["trainer4090", "inference3090"],
                "roots": {
                    "trainer4090": "/path/to/training",
                    "inference3090": "/home/user/llm",
                },
            },
            "warm": {
                "description": "NAS storage",
                "devices": ["synology_data"],
                "roots": {
                    "synology_data": "/volume1/data/llm_training",
                },
            },
            "cold": {
                "description": "Archive storage",
                "devices": ["synology_archive"],
                "roots": {
                    "synology_archive": "/volume1/archive/llm_training",
                },
            },
        },
        "kind_patterns": {
            "checkpoint": {
                "default_zone": "hot",
                "subdir": "models/current_model/{key}",
            },
            "current_model": {
                "default_zone": "hot",
                "subdir": "models/current_model",
            },
            "snapshot": {
                "default_zone": "warm",
                "subdir": "snapshots/{key}",
            },
            "dataset": {
                "default_zone": "warm",
                "subdir": "data/{key}",
            },
            "queue": {
                "default_zone": "hot",
                "subdir": "queue/{key}",
            },
            "status": {
                "default_zone": "hot",
                "subdir": "status/{key}",
            },
        },
    }


@pytest.fixture
def config_file(sample_zones_config, tmp_path):
    """Create a temp config file."""
    config_path = tmp_path / "storage_zones.json"
    with open(config_path, "w") as f:
        json.dump(sample_zones_config, f)
    return config_path


@pytest.fixture
def resolver_trainer(config_file):
    """Create a resolver for trainer4090."""
    return StorageResolver(config_file, device_id="trainer4090")


@pytest.fixture
def resolver_synology(config_file):
    """Create a resolver for synology_data."""
    return StorageResolver(config_file, device_id="synology_data")


# =============================================================================
# TESTS - STORAGE TYPES
# =============================================================================

class TestStorageZone:
    def test_zone_values(self):
        assert StorageZone.HOT.value == "hot"
        assert StorageZone.WARM.value == "warm"
        assert StorageZone.COLD.value == "cold"

    def test_is_local(self):
        assert StorageZone.HOT.is_local
        assert not StorageZone.WARM.is_local
        assert not StorageZone.COLD.is_local

    def test_is_networked(self):
        assert not StorageZone.HOT.is_networked
        assert StorageZone.WARM.is_networked
        assert StorageZone.COLD.is_networked


class TestStorageKind:
    def test_default_zones(self):
        assert StorageKind.CHECKPOINT.default_zone == StorageZone.HOT
        assert StorageKind.SNAPSHOT.default_zone == StorageZone.WARM
        assert StorageKind.BACKUP.default_zone == StorageZone.COLD

    def test_is_model_related(self):
        assert StorageKind.CHECKPOINT.is_model_related
        assert StorageKind.SNAPSHOT.is_model_related
        assert not StorageKind.DATASET.is_model_related

    def test_is_data_related(self):
        assert StorageKind.DATASET.is_data_related
        assert StorageKind.BENCHMARK.is_data_related
        assert not StorageKind.CHECKPOINT.is_data_related


class TestStorageHandle:
    def test_create_handle(self):
        handle = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="checkpoint-182000",
            zone=StorageZone.HOT,
        )
        assert handle.kind == StorageKind.CHECKPOINT
        assert handle.key == "checkpoint-182000"
        assert handle.zone == StorageZone.HOT

    def test_handle_id(self):
        handle = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="checkpoint-182000",
            zone=StorageZone.HOT,
        )
        assert handle.handle_id == "checkpoint:checkpoint-182000@hot"

    def test_for_kind(self):
        handle = StorageHandle.for_kind(StorageKind.CHECKPOINT, "checkpoint-182000")
        assert handle.zone == StorageZone.HOT  # Default for checkpoint

        handle = StorageHandle.for_kind(StorageKind.SNAPSHOT, "snapshot-123")
        assert handle.zone == StorageZone.WARM  # Default for snapshot

    def test_parse(self):
        handle = StorageHandle.parse("checkpoint:checkpoint-182000@hot")
        assert handle.kind == StorageKind.CHECKPOINT
        assert handle.key == "checkpoint-182000"
        assert handle.zone == StorageZone.HOT

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            StorageHandle.parse("invalid")
        with pytest.raises(ValueError):
            StorageHandle.parse("bad:format")

    def test_with_zone(self):
        hot = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="checkpoint-182000",
            zone=StorageZone.HOT,
        )
        cold = hot.with_zone(StorageZone.COLD)
        assert cold.zone == StorageZone.COLD
        assert cold.key == hot.key
        assert cold.kind == hot.kind

    def test_frozen(self):
        """Handles are immutable."""
        handle = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="test",
            zone=StorageZone.HOT,
        )
        with pytest.raises(AttributeError):
            handle.key = "modified"

    def test_hashable(self):
        """Handles can be used in sets and as dict keys."""
        h1 = StorageHandle(StorageKind.CHECKPOINT, "test", StorageZone.HOT)
        h2 = StorageHandle(StorageKind.CHECKPOINT, "test", StorageZone.HOT)
        h3 = StorageHandle(StorageKind.CHECKPOINT, "other", StorageZone.HOT)

        assert hash(h1) == hash(h2)
        assert h1 == h2
        assert h1 != h3

        s = {h1, h2, h3}
        assert len(s) == 2

    def test_empty_key_validation(self):
        """Empty keys are rejected for most kinds."""
        with pytest.raises(ValueError):
            StorageHandle(StorageKind.CHECKPOINT, "", StorageZone.HOT)

        # But allowed for CURRENT_MODEL
        handle = StorageHandle(StorageKind.CURRENT_MODEL, "", StorageZone.HOT)
        assert handle.key == ""


class TestConvenienceHandles:
    def test_checkpoint_handle(self):
        h = checkpoint_handle("checkpoint-182000")
        assert h.kind == StorageKind.CHECKPOINT
        assert h.zone == StorageZone.HOT

    def test_snapshot_handle(self):
        h = snapshot_handle("snapshot-123")
        assert h.kind == StorageKind.SNAPSHOT
        assert h.zone == StorageZone.WARM

    def test_dataset_handle(self):
        h = dataset_handle("binary_l5")
        assert h.kind == StorageKind.DATASET
        assert h.zone == StorageZone.WARM

    def test_queue_handle(self):
        h = queue_handle("high")
        assert h.kind == StorageKind.QUEUE
        assert h.key == "high"


# =============================================================================
# TESTS - RESOLVER
# =============================================================================

class TestStorageResolver:
    def test_resolve_checkpoint_trainer(self, resolver_trainer):
        """Resolve checkpoint on trainer4090."""
        handle = StorageHandle(
            kind=StorageKind.CHECKPOINT,
            key="checkpoint-182000",
            zone=StorageZone.HOT,
        )
        path = resolver_trainer.resolve(handle)
        expected = Path("/path/to/training/models/current_model/checkpoint-182000")
        assert path == expected

    def test_resolve_queue(self, resolver_trainer):
        """Resolve queue directory."""
        handle = StorageHandle(
            kind=StorageKind.QUEUE,
            key="high",
            zone=StorageZone.HOT,
        )
        path = resolver_trainer.resolve(handle)
        expected = Path("/path/to/training/queue/high")
        assert path == expected

    def test_resolve_current_model(self, resolver_trainer):
        """Resolve current_model (no key)."""
        handle = StorageHandle(
            kind=StorageKind.CURRENT_MODEL,
            key="",
            zone=StorageZone.HOT,
        )
        path = resolver_trainer.resolve(handle)
        expected = Path("/path/to/training/models/current_model")
        assert path == expected

    def test_zone_not_available(self, resolver_trainer):
        """Error when zone not available on device."""
        handle = StorageHandle(
            kind=StorageKind.SNAPSHOT,
            key="snapshot-123",
            zone=StorageZone.WARM,  # Not available on trainer
        )
        with pytest.raises(ZoneNotAvailable):
            resolver_trainer.resolve(handle)

    def test_resolve_on_different_device(self, resolver_synology):
        """Resolve on synology which has WARM zone."""
        handle = StorageHandle(
            kind=StorageKind.SNAPSHOT,
            key="snapshot-123",
            zone=StorageZone.WARM,
        )
        path = resolver_synology.resolve(handle)
        expected = Path("/volume1/data/llm_training/snapshots/snapshot-123")
        assert path == expected

    def test_default_handle(self, resolver_trainer):
        """Create handle with default zone."""
        handle = resolver_trainer.default_handle(
            StorageKind.CHECKPOINT, "checkpoint-182000"
        )
        assert handle.zone == StorageZone.HOT

    def test_resolve_default(self, resolver_trainer):
        """Shorthand for resolve(default_handle(...))."""
        path = resolver_trainer.resolve_default(
            StorageKind.CHECKPOINT, "checkpoint-182000"
        )
        expected = Path("/path/to/training/models/current_model/checkpoint-182000")
        assert path == expected

    def test_resolve_or_none(self, resolver_trainer):
        """Returns None instead of raising."""
        handle = StorageHandle(
            kind=StorageKind.SNAPSHOT,
            key="snapshot-123",
            zone=StorageZone.WARM,
        )
        assert resolver_trainer.resolve_or_none(handle) is None

    def test_available_zones(self, resolver_trainer):
        """Get zones available on device."""
        zones = resolver_trainer.available_zones()
        assert StorageZone.HOT in zones
        assert StorageZone.WARM not in zones
        assert StorageZone.COLD not in zones

    def test_zone_root(self, resolver_trainer):
        """Get root path for zone."""
        root = resolver_trainer.zone_root(StorageZone.HOT)
        assert root == Path("/path/to/training")

        # Zone not on device
        warm_root = resolver_trainer.zone_root(StorageZone.WARM)
        assert warm_root is None

    def test_get_info(self, resolver_trainer):
        """Get resolver info."""
        info = resolver_trainer.get_info()
        assert info["device_id"] == "trainer4090"
        assert "hot" in info["available_zones"]
        assert "checkpoint" in info["configured_kinds"]


class TestResolverEdgeCases:
    def test_missing_config(self, tmp_path):
        """Fallback when config missing."""
        missing = tmp_path / "nonexistent.json"
        resolver = StorageResolver(missing, device_id="test")
        # Should have fallback config
        assert len(resolver._zones) > 0

    def test_reload(self, config_file):
        """Reload configuration."""
        resolver = StorageResolver(config_file, device_id="trainer4090")
        zones_before = list(resolver._zones.keys())
        resolver.reload()
        zones_after = list(resolver._zones.keys())
        assert zones_before == zones_after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
