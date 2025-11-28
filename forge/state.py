#!/usr/bin/env python3
"""
Shard State Registry - Tracks validation state of training data shards.

Each dataset has a registry file (data/registry/{dataset_id}.json) that
tracks all shards and their validation status.

Shard lifecycle:
    unknown → pending → validating → ready | rejected

Usage:
    from forge.state import ForgeState, ShardStatus

    state = ForgeState()

    # Register a new shard
    state.register_shard("binary_training_v1", "shard_001.jsonl", "/path/to/raw")

    # Update after validation
    state.update_shard("binary_training_v1", "shard_001.jsonl",
                       status=ShardStatus.READY,
                       invalid_fraction=0.003)

    # Get ready shards for training
    ready = state.get_ready_shards("binary_training_v1")
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import fcntl
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ShardStatus(str, Enum):
    """Shard validation status."""
    UNKNOWN = "unknown"       # Not yet validated
    PENDING = "pending"       # Queued for validation
    VALIDATING = "validating" # Currently being validated
    READY = "ready"           # Passed validation, can be used for training
    REJECTED = "rejected"     # Failed validation
    ARCHIVED = "archived"     # Moved to cold storage


@dataclass
class ShardInfo:
    """Information about a single shard."""
    name: str
    dataset_id: str

    # Paths
    raw_path: Optional[str] = None
    validated_path: Optional[str] = None
    report_path: Optional[str] = None

    # Status
    status: str = ShardStatus.UNKNOWN.value

    # Validation results
    invalid_fraction: Optional[float] = None
    rows_total: Optional[int] = None
    rows_invalid: Optional[int] = None
    leakage_count: Optional[int] = None

    # Timestamps
    created_at: Optional[str] = None
    validated_at: Optional[str] = None
    last_updated_at: Optional[str] = None

    # Job tracking
    validation_job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetState:
    """State of all shards for a dataset."""
    dataset_id: str
    contract_version: str
    shards: List[ShardInfo] = field(default_factory=list)
    last_updated_at: Optional[str] = None

    def get_shard(self, shard_name: str) -> Optional[ShardInfo]:
        """Get shard by name."""
        for shard in self.shards:
            if shard.name == shard_name:
                return shard
        return None

    def get_by_status(self, status: ShardStatus) -> List[ShardInfo]:
        """Get shards with given status."""
        return [s for s in self.shards if s.status == status.value]

    def summary(self) -> Dict[str, int]:
        """Get count by status."""
        counts = {s.value: 0 for s in ShardStatus}
        for shard in self.shards:
            counts[shard.status] = counts.get(shard.status, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "contract_version": self.contract_version,
            "shards": [s.to_dict() for s in self.shards],
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetState":
        shards = [ShardInfo.from_dict(s) for s in data.get("shards", [])]
        return cls(
            dataset_id=data["dataset_id"],
            contract_version=data.get("contract_version", "1"),
            shards=shards,
            last_updated_at=data.get("last_updated_at"),
        )


class ForgeState:
    """
    Manages shard state across all datasets.

    State is persisted to data/registry/{dataset_id}.json
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        if registry_dir is None:
            try:
                from core.paths import get_base_dir
                registry_dir = get_base_dir() / "data" / "registry"
            except ImportError:
                registry_dir = Path("data/registry")

        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, DatasetState] = {}

    @contextmanager
    def _lock(self, dataset_id: str):
        """File-based lock for concurrent access."""
        lock_file = self.registry_dir / f".{dataset_id}.lock"
        lock_file.touch(exist_ok=True)

        with open(lock_file, 'r') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _get_state_path(self, dataset_id: str) -> Path:
        return self.registry_dir / f"{dataset_id}.json"

    def get_dataset_state(self, dataset_id: str) -> Optional[DatasetState]:
        """Load dataset state from disk."""
        # Check cache first
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        path = self._get_state_path(dataset_id)
        if not path.exists():
            return None

        try:
            with self._lock(dataset_id):
                with open(path) as f:
                    data = json.load(f)
                    state = DatasetState.from_dict(data)
                    self._cache[dataset_id] = state
                    return state
        except Exception as e:
            logger.error(f"Failed to load state for {dataset_id}: {e}")
            return None

    def save_dataset_state(self, state: DatasetState):
        """Save dataset state to disk."""
        state.last_updated_at = datetime.utcnow().isoformat() + "Z"

        path = self._get_state_path(state.dataset_id)
        with self._lock(state.dataset_id):
            with open(path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

        self._cache[state.dataset_id] = state

    def register_shard(
        self,
        dataset_id: str,
        shard_name: str,
        raw_path: str,
        contract_version: str = "1",
    ) -> ShardInfo:
        """
        Register a new shard for a dataset.

        Creates dataset state if it doesn't exist.
        """
        state = self.get_dataset_state(dataset_id)

        if state is None:
            state = DatasetState(
                dataset_id=dataset_id,
                contract_version=contract_version,
            )

        # Check if shard already exists
        existing = state.get_shard(shard_name)
        if existing:
            logger.debug(f"Shard already registered: {dataset_id}/{shard_name}")
            return existing

        # Create new shard
        shard = ShardInfo(
            name=shard_name,
            dataset_id=dataset_id,
            raw_path=raw_path,
            status=ShardStatus.UNKNOWN.value,
            created_at=datetime.utcnow().isoformat() + "Z",
        )

        state.shards.append(shard)
        self.save_dataset_state(state)

        logger.info(f"Registered shard: {dataset_id}/{shard_name}")
        return shard

    def update_shard(
        self,
        dataset_id: str,
        shard_name: str,
        **updates
    ) -> Optional[ShardInfo]:
        """
        Update a shard's fields.

        Common updates:
            status=ShardStatus.READY
            invalid_fraction=0.003
            validated_at="2025-11-28T..."
        """
        state = self.get_dataset_state(dataset_id)
        if state is None:
            logger.error(f"No state for dataset: {dataset_id}")
            return None

        shard = state.get_shard(shard_name)
        if shard is None:
            logger.error(f"Shard not found: {dataset_id}/{shard_name}")
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(shard, key):
                # Convert enum to value
                if isinstance(value, ShardStatus):
                    value = value.value
                setattr(shard, key, value)

        shard.last_updated_at = datetime.utcnow().isoformat() + "Z"

        self.save_dataset_state(state)
        logger.debug(f"Updated shard: {dataset_id}/{shard_name} with {list(updates.keys())}")

        return shard

    def get_ready_shards(self, dataset_id: str) -> List[ShardInfo]:
        """Get all shards with status=ready for a dataset."""
        state = self.get_dataset_state(dataset_id)
        if state is None:
            return []
        return state.get_by_status(ShardStatus.READY)

    def get_pending_shards(self, dataset_id: str) -> List[ShardInfo]:
        """Get all shards needing validation."""
        state = self.get_dataset_state(dataset_id)
        if state is None:
            return []
        return [s for s in state.shards if s.status in (
            ShardStatus.UNKNOWN.value,
            ShardStatus.PENDING.value
        )]

    def list_datasets(self) -> List[str]:
        """List all datasets with state."""
        return [p.stem for p in self.registry_dir.glob("*.json")]

    def get_all_states(self) -> Dict[str, DatasetState]:
        """Get state for all datasets."""
        states = {}
        for dataset_id in self.list_datasets():
            state = self.get_dataset_state(dataset_id)
            if state:
                states[dataset_id] = state
        return states

    def get_summary(self) -> Dict[str, Any]:
        """Get summary across all datasets."""
        summary = {
            "datasets": {},
            "total_shards": 0,
            "total_ready": 0,
            "total_rejected": 0,
        }

        for dataset_id in self.list_datasets():
            state = self.get_dataset_state(dataset_id)
            if state:
                counts = state.summary()
                summary["datasets"][dataset_id] = {
                    "total": len(state.shards),
                    "by_status": counts,
                }
                summary["total_shards"] += len(state.shards)
                summary["total_ready"] += counts.get(ShardStatus.READY.value, 0)
                summary["total_rejected"] += counts.get(ShardStatus.REJECTED.value, 0)

        return summary

    def invalidate_cache(self, dataset_id: Optional[str] = None):
        """Clear the in-memory cache."""
        if dataset_id:
            self._cache.pop(dataset_id, None)
        else:
            self._cache.clear()


# Convenience functions

_state: Optional[ForgeState] = None


def get_forge_state() -> ForgeState:
    """Get the global ForgeState instance."""
    global _state
    if _state is None:
        _state = ForgeState()
    return _state


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    state = get_forge_state()

    print("Forge State Registry")
    print("=" * 60)

    datasets = state.list_datasets()

    if not datasets:
        print("No datasets registered yet.")
        print(f"Registry location: {state.registry_dir}")
        sys.exit(0)

    summary = state.get_summary()
    print(f"Total shards: {summary['total_shards']}")
    print(f"Ready: {summary['total_ready']}")
    print(f"Rejected: {summary['total_rejected']}")

    print("\nDatasets:")
    for dataset_id, info in summary["datasets"].items():
        status_str = ", ".join(
            f"{k}:{v}" for k, v in info["by_status"].items() if v > 0
        )
        print(f"  {dataset_id}: {info['total']} shards ({status_str})")
