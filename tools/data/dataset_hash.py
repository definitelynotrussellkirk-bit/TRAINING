#!/usr/bin/env python3
"""
Dataset Hash Tracking System

Prevents checkpoint reuse across different datasets by tracking dataset identity.
Solves the bug where switching datasets with existing checkpoints causes step counter mismatches.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


def compute_dataset_hash(dataset_path: Path) -> str:
    """
    Compute unique hash for a dataset.

    Uses filename + size for speed (instead of hashing full content).
    This is sufficient for detecting dataset changes.

    Args:
        dataset_path: Path to training dataset file

    Returns:
        Hex string hash
    """
    dataset_path = Path(dataset_path)

    # Use filename + size as identifier (fast, sufficient for our use case)
    identifier = f"{dataset_path.name}:{dataset_path.stat().st_size}"

    return hashlib.md5(identifier.encode()).hexdigest()


def save_dataset_metadata(output_dir: Path, dataset_path: Path, lora_config: Dict) -> None:
    """
    Save dataset metadata to output directory for checkpoint validation.

    Args:
        output_dir: Model output directory (where checkpoints are saved)
        dataset_path: Path to training dataset
        lora_config: LoRA configuration dict (r, alpha, dropout, etc.)
    """
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)

    metadata = {
        "dataset_name": dataset_path.name,
        "dataset_hash": compute_dataset_hash(dataset_path),
        "dataset_path": str(dataset_path.absolute()),
        "lora_config": lora_config
    }

    metadata_file = output_dir / ".dataset_metadata.json"

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_dataset_metadata(output_dir: Path) -> Optional[Dict]:
    """
    Load dataset metadata from output directory.

    Returns:
        Metadata dict or None if not found
    """
    output_dir = Path(output_dir)
    metadata_file = output_dir / ".dataset_metadata.json"

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def validate_checkpoint_compatibility(
    output_dir: Path,
    dataset_path: Path,
    lora_config: Dict
) -> tuple[bool, Optional[str]]:
    """
    Check if existing checkpoints are compatible with current dataset and config.

    Args:
        output_dir: Model output directory
        dataset_path: Current training dataset
        lora_config: Current LoRA configuration

    Returns:
        (is_compatible, reason) tuple
        - is_compatible: True if checkpoints can be reused
        - reason: String explanation if incompatible, None if compatible
    """
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)

    # Load previous metadata
    prev_metadata = load_dataset_metadata(output_dir)

    if prev_metadata is None:
        # No metadata = old checkpoints (before this feature)
        # Conservatively assume incompatible
        return False, "No dataset metadata found (old checkpoints)"

    # Compute current dataset hash
    current_hash = compute_dataset_hash(dataset_path)

    # Check dataset hash
    if prev_metadata["dataset_hash"] != current_hash:
        prev_name = prev_metadata.get("dataset_name", "unknown")
        return False, f"Dataset changed: {prev_name} → {dataset_path.name}"

    # Check LoRA config
    prev_lora = prev_metadata.get("lora_config", {})

    # Critical LoRA params that must match
    critical_params = ["r", "alpha", "dropout"]

    for param in critical_params:
        if prev_lora.get(param) != lora_config.get(param):
            return False, f"LoRA config mismatch: {param} changed ({prev_lora.get(param)} → {lora_config.get(param)})"

    # All checks passed
    return True, None


def clear_checkpoints(output_dir: Path, logger=None) -> int:
    """
    Clear all checkpoint directories in output_dir.

    Args:
        output_dir: Model output directory
        logger: Optional logger for status messages

    Returns:
        Number of checkpoints cleared
    """
    import shutil

    output_dir = Path(output_dir)
    checkpoints = list(output_dir.glob("checkpoint-*"))

    if logger:
        logger.info(f"🧹 Clearing {len(checkpoints)} incompatible checkpoints...")

    for checkpoint in checkpoints:
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
            if logger:
                logger.info(f"   Removed: {checkpoint.name}")

    if logger and checkpoints:
        logger.info(f"✅ Cleared {len(checkpoints)} checkpoints - starting fresh")

    return len(checkpoints)
