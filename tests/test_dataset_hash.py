#!/usr/bin/env python3
"""
Test script for dataset hash tracking system
"""

import tempfile
import json
from pathlib import Path
from dataset_hash import (
    compute_dataset_hash,
    save_dataset_metadata,
    load_dataset_metadata,
    validate_checkpoint_compatibility,
    clear_checkpoints
)


def test_hash_computation():
    """Test that same file produces same hash"""
    print("Testing hash computation...")

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"test": "data"}\n')
        temp_path = Path(f.name)

    try:
        hash1 = compute_dataset_hash(temp_path)
        hash2 = compute_dataset_hash(temp_path)

        assert hash1 == hash2, "Same file should produce same hash"
        print(f"  ✅ Hash: {hash1}")

    finally:
        temp_path.unlink()


def test_metadata_save_load():
    """Test saving and loading metadata"""
    print("\nTesting metadata save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create fake dataset
        dataset_file = output_dir / "data.jsonl"
        dataset_file.write_text('{"test": "data"}\n')

        lora_config = {"r": 128, "alpha": 128, "dropout": 0.05}

        # Save metadata
        save_dataset_metadata(output_dir, dataset_file, lora_config)

        # Load it back
        metadata = load_dataset_metadata(output_dir)

        assert metadata is not None, "Metadata should load"
        assert metadata["dataset_name"] == "data.jsonl", "Dataset name should match"
        assert metadata["lora_config"] == lora_config, "LoRA config should match"

        print(f"  ✅ Metadata saved and loaded successfully")
        print(f"     Dataset: {metadata['dataset_name']}")
        print(f"     Hash: {metadata['dataset_hash']}")


def test_compatibility_check():
    """Test checkpoint compatibility validation"""
    print("\nTesting compatibility check...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create dataset 1
        dataset1 = output_dir / "data1.jsonl"
        dataset1.write_text('{"test": "data1"}\n')

        # Create dataset 2 (different)
        dataset2 = output_dir / "data2.jsonl"
        dataset2.write_text('{"test": "data2"}\n' * 100)  # Different size

        lora_config = {"r": 128, "alpha": 128, "dropout": 0.05}

        # Save metadata for dataset1
        save_dataset_metadata(output_dir, dataset1, lora_config)

        # Check compatibility with dataset1 (should match)
        compatible, reason = validate_checkpoint_compatibility(
            output_dir, dataset1, lora_config
        )
        assert compatible, "Should be compatible with same dataset"
        print(f"  ✅ Same dataset: compatible")

        # Check compatibility with dataset2 (should NOT match)
        compatible, reason = validate_checkpoint_compatibility(
            output_dir, dataset2, lora_config
        )
        assert not compatible, "Should NOT be compatible with different dataset"
        print(f"  ✅ Different dataset: incompatible ({reason})")

        # Check compatibility with different LoRA config
        lora_config2 = {"r": 64, "alpha": 64, "dropout": 0.05}
        compatible, reason = validate_checkpoint_compatibility(
            output_dir, dataset1, lora_config2
        )
        assert not compatible, "Should NOT be compatible with different LoRA config"
        print(f"  ✅ Different LoRA config: incompatible ({reason})")


def test_checkpoint_clearing():
    """Test clearing checkpoints"""
    print("\nTesting checkpoint clearing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create fake checkpoints
        for i in [100, 200, 300]:
            checkpoint_dir = output_dir / f"checkpoint-{i}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "test.txt").write_text("fake checkpoint")

        # Verify they exist
        checkpoints = list(output_dir.glob("checkpoint-*"))
        assert len(checkpoints) == 3, "Should have 3 checkpoints"

        # Clear them
        num_cleared = clear_checkpoints(output_dir)
        assert num_cleared == 3, "Should clear 3 checkpoints"

        # Verify they're gone
        checkpoints = list(output_dir.glob("checkpoint-*"))
        assert len(checkpoints) == 0, "Should have 0 checkpoints after clearing"

        print(f"  ✅ Cleared 3 checkpoints successfully")


def test_dataset_switching_scenario():
    """Test the actual bug scenario: switching datasets"""
    print("\nTesting dataset switching scenario (the bug we hit)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Simulate training on dataset A
        dataset_a = output_dir / "syllo_hard_20k.jsonl"
        dataset_a.write_text('{"data": "A"}\n' * 20000)

        lora_config = {"r": 128, "alpha": 128, "dropout": 0.05}

        # Save metadata (as if we just trained)
        save_dataset_metadata(output_dir, dataset_a, lora_config)

        # Create checkpoints (simulating previous training at step 4000)
        checkpoint_4000 = output_dir / "checkpoint-4000"
        checkpoint_4000.mkdir()
        (checkpoint_4000 / "adapter_model.safetensors").write_text("fake weights")

        print(f"  📁 Simulated state: checkpoint-4000 from {dataset_a.name}")

        # Now switch to dataset B (different file)
        dataset_b = output_dir / "syllo_hard_20000.jsonl"  # Note: different filename
        dataset_b.write_text('{"data": "B"}\n' * 20000)

        print(f"  🔄 Switching to new dataset: {dataset_b.name}")

        # Check compatibility
        compatible, reason = validate_checkpoint_compatibility(
            output_dir, dataset_b, lora_config
        )

        assert not compatible, "Should detect dataset change"
        print(f"  ✅ Detected incompatibility: {reason}")

        # Clear checkpoints (what the fix does)
        num_cleared = clear_checkpoints(output_dir)
        print(f"  🧹 Cleared {num_cleared} checkpoint(s)")

        # Verify cleared
        checkpoints = list(output_dir.glob("checkpoint-*"))
        assert len(checkpoints) == 0, "Checkpoints should be cleared"
        print(f"  ✅ Ready for fresh training on new dataset")


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET HASH TRACKING TEST SUITE")
    print("=" * 60)

    test_hash_computation()
    test_metadata_save_load()
    test_compatibility_check()
    test_checkpoint_clearing()
    test_dataset_switching_scenario()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
