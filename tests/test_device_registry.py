"""
Tests for the Device Registry system.

Tests cover:
- Device loading from config
- Role-based filtering
- GPU/compute capability queries
- Storage zone queries
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from core.devices import (
    DeviceRole,
    DeviceInfo,
    DeviceRegistry,
    GPUInfo,
    CPUInfo,
    NetworkInfo,
    get_device_registry,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_devices_config():
    """Create a sample devices.json config."""
    return {
        "schema_version": 1,
        "devices": {
            "trainer4090": {
                "hostname": "192.168.x.x",
                "description": "Primary trainer",
                "roles": ["trainer", "eval_worker", "storage_hot"],
                "gpus": [{"name": "RTX 4090", "count": 1, "vram_gb": 24}],
                "cpu": {"cores": 16, "threads": 32},
                "memory_gb": 64,
                "storage_zones": ["hot"],
                "network": {"speed_gbps": 10, "tags": ["lan_core"]},
                "enabled": True,
            },
            "inference3090": {
                "hostname": "192.168.x.x",
                "roles": ["inference", "eval_worker"],
                "gpus": [{"name": "RTX 3090", "count": 1, "vram_gb": 24}],
                "cpu": {"cores": 8, "threads": 16},
                "memory_gb": 32,
                "storage_zones": ["hot"],
                "network": {"speed_gbps": 10, "tags": ["lan_core"]},
                "enabled": True,
            },
            "macmini_worker": {
                "hostname": "macmini.local",
                "roles": ["eval_worker", "data_forge", "vault_worker"],
                "gpus": [],
                "cpu": {"cores": 8, "threads": 8},
                "memory_gb": 16,
                "storage_zones": [],
                "network": {"speed_gbps": 1, "tags": ["lan_edge"]},
                "enabled": True,
            },
            "disabled_device": {
                "hostname": "disabled.local",
                "roles": ["trainer"],
                "gpus": [{"name": "RTX 3080", "count": 1, "vram_gb": 10}],
                "cpu": {"cores": 8, "threads": 16},
                "memory_gb": 32,
                "storage_zones": [],
                "network": {"speed_gbps": 1, "tags": []},
                "enabled": False,
            },
            "synology_nas": {
                "hostname": "192.168.x.x",
                "roles": ["storage_warm"],
                "gpus": [],
                "cpu": {"cores": 4, "threads": 8},
                "memory_gb": 8,
                "storage_zones": ["warm"],
                "network": {"speed_gbps": 10, "tags": ["nas"]},
                "enabled": True,
            },
        },
    }


@pytest.fixture
def config_file(sample_devices_config, tmp_path):
    """Create a temp config file."""
    config_path = tmp_path / "devices.json"
    with open(config_path, "w") as f:
        json.dump(sample_devices_config, f)
    return config_path


@pytest.fixture
def registry(config_file):
    """Create a registry from temp config."""
    return DeviceRegistry(config_file)


# =============================================================================
# TESTS - DATACLASSES
# =============================================================================

class TestGPUInfo:
    def test_total_vram(self):
        gpu = GPUInfo(name="RTX 4090", count=2, vram_gb=24)
        assert gpu.total_vram() == 48

    def test_from_dict(self):
        data = {"name": "RTX 3090", "count": 1, "vram_gb": 24}
        gpu = GPUInfo.from_dict(data)
        assert gpu.name == "RTX 3090"
        assert gpu.vram_gb == 24

    def test_to_dict(self):
        gpu = GPUInfo(name="RTX 4090", count=1, vram_gb=24)
        d = gpu.to_dict()
        assert d["name"] == "RTX 4090"


class TestDeviceInfo:
    def test_has_role(self):
        device = DeviceInfo(
            device_id="test",
            hostname="test.local",
            roles=[DeviceRole.TRAINER, DeviceRole.EVAL_WORKER],
        )
        assert device.has_role(DeviceRole.TRAINER)
        assert device.has_role(DeviceRole.EVAL_WORKER)
        assert not device.has_role(DeviceRole.INFERENCE)

    def test_has_any_role(self):
        device = DeviceInfo(
            device_id="test",
            hostname="test.local",
            roles=[DeviceRole.TRAINER],
        )
        assert device.has_any_role([DeviceRole.TRAINER, DeviceRole.INFERENCE])
        assert not device.has_any_role([DeviceRole.INFERENCE, DeviceRole.DATA_FORGE])

    def test_has_gpu(self):
        gpu_device = DeviceInfo(
            device_id="gpu",
            hostname="gpu.local",
            gpus=[GPUInfo(name="RTX 4090", count=1, vram_gb=24)],
        )
        cpu_device = DeviceInfo(
            device_id="cpu",
            hostname="cpu.local",
            gpus=[],
        )
        assert gpu_device.has_gpu()
        assert not cpu_device.has_gpu()

    def test_total_vram(self):
        device = DeviceInfo(
            device_id="multi",
            hostname="multi.local",
            gpus=[
                GPUInfo(name="RTX 4090", count=2, vram_gb=24),
                GPUInfo(name="RTX 3090", count=1, vram_gb=24),
            ],
        )
        assert device.total_vram() == 72  # 48 + 24

    def test_has_storage_zone(self):
        device = DeviceInfo(
            device_id="test",
            hostname="test.local",
            storage_zones=["hot", "warm"],
        )
        assert device.has_storage_zone("hot")
        assert device.has_storage_zone("warm")
        assert not device.has_storage_zone("cold")

    def test_is_compute_device(self):
        trainer = DeviceInfo(
            device_id="trainer",
            hostname="trainer.local",
            roles=[DeviceRole.TRAINER],
        )
        storage = DeviceInfo(
            device_id="storage",
            hostname="storage.local",
            roles=[DeviceRole.STORAGE_WARM],
        )
        assert trainer.is_compute_device()
        assert not storage.is_compute_device()


# =============================================================================
# TESTS - REGISTRY
# =============================================================================

class TestDeviceRegistry:
    def test_load_devices(self, registry):
        """Registry loads all devices from config."""
        devices = registry.all_devices()
        assert len(devices) == 5

    def test_get_device(self, registry):
        """Can get specific device by ID."""
        trainer = registry.get("trainer4090")
        assert trainer is not None
        assert trainer.hostname == "192.168.x.x"

    def test_get_nonexistent(self, registry):
        """Returns None for nonexistent device."""
        assert registry.get("nonexistent") is None

    def test_enabled_devices(self, registry):
        """Only returns enabled devices."""
        enabled = registry.enabled_devices()
        assert len(enabled) == 4  # One device is disabled
        ids = [d.device_id for d in enabled]
        assert "disabled_device" not in ids

    def test_devices_with_role_trainer(self, registry):
        """Find devices with trainer role."""
        trainers = registry.devices_with_role(DeviceRole.TRAINER)
        assert len(trainers) == 1
        assert trainers[0].device_id == "trainer4090"

    def test_devices_with_role_eval_worker(self, registry):
        """Find all eval workers."""
        workers = registry.devices_with_role(DeviceRole.EVAL_WORKER)
        assert len(workers) == 3
        ids = {d.device_id for d in workers}
        assert "trainer4090" in ids
        assert "inference3090" in ids
        assert "macmini_worker" in ids

    def test_devices_with_role_includes_disabled(self, registry):
        """Can optionally include disabled devices."""
        # enabled_only=True (default) - excludes disabled
        trainers = registry.devices_with_role(DeviceRole.TRAINER, enabled_only=True)
        assert len(trainers) == 1

        # enabled_only=False - includes disabled
        trainers_all = registry.devices_with_role(DeviceRole.TRAINER, enabled_only=False)
        assert len(trainers_all) == 2

    def test_devices_with_storage_zone(self, registry):
        """Find devices with specific storage zone."""
        hot_devices = registry.devices_with_storage_zone("hot")
        assert len(hot_devices) == 2

        warm_devices = registry.devices_with_storage_zone("warm")
        assert len(warm_devices) == 1
        assert warm_devices[0].device_id == "synology_nas"

    def test_devices_with_gpu(self, registry):
        """Find devices with GPU."""
        gpu_devices = registry.devices_with_gpu()
        assert len(gpu_devices) == 2
        ids = {d.device_id for d in gpu_devices}
        assert "trainer4090" in ids
        assert "inference3090" in ids

    def test_compute_devices(self, registry):
        """Find compute devices."""
        compute = registry.compute_devices()
        assert len(compute) == 3  # trainer, inference, macmini (has data_forge)

    def test_storage_devices(self, registry):
        """Find storage devices."""
        storage = registry.storage_devices()
        # trainer4090 (storage_hot) and synology_nas (storage_warm)
        # Note: inference3090 doesn't have storage role in test fixture
        assert len(storage) == 2

    def test_get_trainer(self, registry):
        """Get primary trainer."""
        trainer = registry.get_trainer()
        assert trainer is not None
        assert trainer.device_id == "trainer4090"

    def test_get_inference(self, registry):
        """Get primary inference device."""
        inference = registry.get_inference()
        assert inference is not None
        assert inference.device_id == "inference3090"

    def test_get_summary(self, registry):
        """Get registry summary."""
        summary = registry.get_summary()
        assert summary["total_devices"] == 5
        assert summary["enabled_devices"] == 4
        assert summary["total_gpus"] == 2
        assert summary["total_vram_gb"] == 48
        assert "trainer" in summary["role_counts"]
        assert "hot" in summary["zone_counts"]


# =============================================================================
# TESTS - EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_missing_config_file(self, tmp_path):
        """Registry handles missing config gracefully."""
        missing_path = tmp_path / "nonexistent.json"
        registry = DeviceRegistry(missing_path)
        assert len(registry.all_devices()) == 0

    def test_empty_config(self, tmp_path):
        """Registry handles empty config."""
        config_path = tmp_path / "empty.json"
        config_path.write_text('{"devices": {}}')
        registry = DeviceRegistry(config_path)
        assert len(registry.all_devices()) == 0

    def test_unknown_role(self, tmp_path):
        """Unknown roles are ignored with warning."""
        config = {
            "devices": {
                "test": {
                    "hostname": "test.local",
                    "roles": ["trainer", "unknown_role"],
                    "gpus": [],
                    "cpu": {"cores": 4, "threads": 4},
                    "memory_gb": 16,
                    "storage_zones": [],
                    "network": {"speed_gbps": 1, "tags": []},
                },
            },
        }
        config_path = tmp_path / "test.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        registry = DeviceRegistry(config_path)
        device = registry.get("test")
        assert device is not None
        assert len(device.roles) == 1
        assert DeviceRole.TRAINER in device.roles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
