#!/usr/bin/env python3
"""
Training Gating - Ensure only validated data is used for training.

This module provides functions to gate training on validated data,
preventing use of unvalidated or rejected shards.

Usage:
    from forge.gating import get_training_shards, require_validation

    # Get only validated shards for a dataset
    shards = get_training_shards("binary_training_v1")

    # Check if a specific file can be used for training
    ok, reason = can_train_on(Path("data.jsonl"))
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ValidationRequired(Exception):
    """Raised when trying to train on unvalidated data."""
    pass


def get_training_shards(
    dataset_id: str,
    require_validation: bool = True,
) -> List[Path]:
    """
    Get validated shards for a dataset.

    Args:
        dataset_id: Dataset to get shards for
        require_validation: If True, raise error if no validated shards

    Returns:
        List of paths to validated shard files

    Raises:
        ValidationRequired: If require_validation=True and no validated shards
    """
    from forge.state import get_forge_state, ShardStatus

    state_mgr = get_forge_state()
    dataset_state = state_mgr.get_dataset_state(dataset_id)

    if dataset_state is None:
        if require_validation:
            raise ValidationRequired(
                f"Dataset '{dataset_id}' has no Forge state. "
                f"Register shards and run validation first."
            )
        logger.warning(f"No Forge state for {dataset_id}, returning empty list")
        return []

    # Get ready shards
    ready_shards = dataset_state.get_by_status(ShardStatus.READY)

    if not ready_shards:
        if require_validation:
            # Provide helpful error with status breakdown
            counts = dataset_state.summary()
            status_str = ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
            raise ValidationRequired(
                f"Dataset '{dataset_id}' has no validated shards. "
                f"Status: {status_str}. "
                f"Run: python -m forge.cli validate --dataset {dataset_id}"
            )
        logger.warning(f"No validated shards for {dataset_id}")
        return []

    # Return paths
    paths = []
    for shard in ready_shards:
        # Prefer validated_path, fall back to raw_path
        path = shard.validated_path or shard.raw_path
        if path:
            paths.append(Path(path))

    logger.info(f"Found {len(paths)} validated shards for {dataset_id}")
    return paths


def get_all_training_shards(
    dataset_ids: Optional[List[str]] = None,
    require_validation: bool = True,
) -> Dict[str, List[Path]]:
    """
    Get validated shards for multiple datasets.

    Args:
        dataset_ids: List of datasets (if None, gets all)
        require_validation: If True, raise error if any dataset has no shards

    Returns:
        Dict of dataset_id -> list of shard paths
    """
    from forge.state import get_forge_state

    state_mgr = get_forge_state()

    if dataset_ids is None:
        dataset_ids = state_mgr.list_datasets()

    result = {}
    for dataset_id in dataset_ids:
        try:
            shards = get_training_shards(dataset_id, require_validation=require_validation)
            result[dataset_id] = shards
        except ValidationRequired:
            if require_validation:
                raise
            result[dataset_id] = []

    return result


def can_train_on(
    file_path: Path,
    dataset_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Check if a file can be used for training.

    Args:
        file_path: Path to the file
        dataset_id: Optional dataset ID (auto-detected if not provided)

    Returns:
        Tuple of (allowed, reason)
    """
    from forge.state import get_forge_state, ShardStatus

    file_path = Path(file_path)
    shard_name = file_path.name

    # Auto-detect dataset from filename if not provided
    if dataset_id is None:
        dataset_id = _detect_dataset_from_filename(shard_name)

    if dataset_id is None:
        # No dataset association, allow by default (for backward compatibility)
        return True, "no_dataset_association"

    state_mgr = get_forge_state()
    dataset_state = state_mgr.get_dataset_state(dataset_id)

    if dataset_state is None:
        return True, "dataset_not_registered"

    shard = dataset_state.get_shard(shard_name)

    if shard is None:
        return True, "shard_not_registered"

    if shard.status == ShardStatus.READY.value:
        return True, "validated"
    elif shard.status == ShardStatus.REJECTED.value:
        return False, f"rejected (invalid: {shard.invalid_fraction:.1%})"
    elif shard.status in (ShardStatus.PENDING.value, ShardStatus.VALIDATING.value):
        return False, f"validation_in_progress ({shard.status})"
    else:
        return False, f"unvalidated ({shard.status})"


def _detect_dataset_from_filename(filename: str) -> Optional[str]:
    """
    Detect dataset ID from filename patterns.

    Examples:
        "train_SYLLO_L1_20251128.jsonl" -> "syllo_training_v1"
        "sparring_binary_L5_100.jsonl" -> "sparring_v1"
        "bin_training_batch_001.jsonl" -> "binary_training_v1"
    """
    filename_lower = filename.lower()

    if "sparring" in filename_lower:
        return "sparring_v1"
    if "syllo" in filename_lower or "_sy_" in filename_lower:
        return "syllo_training_v1"
    if "binary" in filename_lower or "_bin_" in filename_lower or "bin_" in filename_lower:
        return "binary_training_v1"

    return None


def validate_before_training(
    files: List[Path],
    dataset_id: Optional[str] = None,
    allow_unvalidated: bool = False,
) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """
    Check which files can be used for training.

    Args:
        files: List of files to check
        dataset_id: Optional dataset ID
        allow_unvalidated: If True, include unvalidated files with warning

    Returns:
        Tuple of (allowed_files, rejected_files_with_reasons)
    """
    allowed = []
    rejected = []

    for file_path in files:
        ok, reason = can_train_on(file_path, dataset_id)

        if ok:
            allowed.append(file_path)
        elif allow_unvalidated and reason in ("no_dataset_association", "dataset_not_registered", "shard_not_registered"):
            logger.warning(f"Using unvalidated file: {file_path.name} ({reason})")
            allowed.append(file_path)
        else:
            rejected.append((file_path, reason))
            logger.warning(f"Rejecting {file_path.name}: {reason}")

    return allowed, rejected


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Training Gating")
    print("=" * 60)

    from forge.state import get_forge_state

    state_mgr = get_forge_state()
    datasets = state_mgr.list_datasets()

    if not datasets:
        print("No datasets registered.")
        print("\nTo register a dataset:")
        print("  1. Create contract in configs/datasets/{id}.yaml")
        print("  2. Register shards via Forge CLI or API")
        print("  3. Run validation")
        sys.exit(0)

    for dataset_id in datasets:
        try:
            shards = get_training_shards(dataset_id, require_validation=False)
            print(f"\n{dataset_id}: {len(shards)} validated shards")
            for shard in shards[:3]:
                print(f"  - {shard.name}")
            if len(shards) > 3:
                print(f"  ... and {len(shards) - 3} more")
        except ValidationRequired as e:
            print(f"\n{dataset_id}: {e}")
