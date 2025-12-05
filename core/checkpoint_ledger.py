"""
Checkpoint Ledger - Single source of truth for checkpoint metadata and stats.

The Ledger records checkpoint creation with exact stats at save time,
provides canonical naming, and serves as the authority for checkpoint info.

Usage:
    from core.checkpoint_ledger import CheckpointLedger, CheckpointRecord

    ledger = CheckpointLedger()

    # Record a checkpoint (called by on_save callback)
    record = ledger.record(
        step=190000,
        path="/path/to/checkpoint-190000",
        train_loss=0.432,
        val_loss=0.458,
        skill_name="binary",
        skill_level=5,
    )

    # Get canonical name
    print(record.canonical_name)  # "checkpoint-190000-20251127-1430"

    # Query ledger
    record = ledger.get(190000)
    records = ledger.list_all()
    best = ledger.get_best(metric="train_loss")

Design:
    1. Central index at status/checkpoint_ledger.json
    2. Sidecar .ledger.json in each checkpoint dir (self-documenting)
    3. Canonical naming: checkpoint-{step}-{YYYYMMDD}-{HHMM}
    4. Stats recorded at exact moment of save

RPG Flavor:
    The Ledger is the Great Book where all milestones are recorded.
    When a champion achieves a new level, the Scribe records it here -
    the exact stats, the time, the context. Nothing is forgotten.
"""

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("checkpoint_ledger")

# Sidecar filename (written inside each checkpoint dir)
LEDGER_SIDECAR = ".ledger.json"

# Central index filename
LEDGER_INDEX = "checkpoint_ledger.json"


@dataclass
class CheckpointRecord:
    """
    Complete record of a checkpoint at creation time.

    This captures everything we know about the checkpoint when it was saved.
    """
    # Identity
    step: int
    timestamp: str  # ISO format
    path: str  # Original path at creation
    canonical_name: str  # Canonical name (step-date-time)

    # Training stats at save time
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None

    # Evaluation stats (if available)
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None

    # Training context
    training_file: Optional[str] = None
    skill_name: Optional[str] = None
    skill_level: Optional[int] = None
    total_examples_trained: Optional[int] = None

    # === HERO TRACKING (v1.2.0) ===
    # Which hero owns this checkpoint
    hero_id: Optional[str] = None  # e.g., "ojas-qwen3-8b", "dio-qwen3-0.6b"
    campaign_id: Optional[str] = None  # e.g., "campaign-001"

    # Physical properties
    size_bytes: Optional[int] = None
    has_optimizer: bool = True

    # Usage tracking (for per-device retention)
    last_used: Optional[str] = None  # ISO timestamp of last use (inference, eval, etc.)
    locations: List[str] = field(default_factory=list)  # device_ids where this checkpoint exists

    # === SKILL METRICS (v1.1.0) ===
    # Per-skill metrics at checkpoint time for curriculum decisions
    # Format: {"sy": {"accuracy": 0.85, "level": 8, "eval_count": 3}, "bin": {...}}
    skill_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Current skill levels at save time (snapshot for curriculum)
    # Format: {"sy": 8, "bin": 5}
    skill_levels_at_save: Dict[str, int] = field(default_factory=dict)

    # Linked eval IDs - references to EvaluationLedger records
    # Format: ["checkpoint_step:skill:level", ...]
    linked_eval_ids: List[str] = field(default_factory=list)

    # Cumulative effort per skill at checkpoint time (from StrainTracker)
    # Format: {"sy": 150.5, "bin": 89.2}
    skill_effort_at_save: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_by: str = "training_daemon"
    ledger_version: str = "1.2.0"  # Added hero_id, campaign_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointRecord":
        # Handle missing fields gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @property
    def size_gb(self) -> float:
        if self.size_bytes:
            return self.size_bytes / (1024 ** 3)
        return 0.0

    @property
    def age_hours(self) -> float:
        try:
            created = datetime.fromisoformat(self.timestamp)
            return (datetime.now() - created).total_seconds() / 3600
        except (ValueError, TypeError):
            return 0.0

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        parts = [f"Step {self.step:,}"]
        if self.train_loss:
            parts.append(f"Loss {self.train_loss:.4f}")
        if self.skill_name:
            parts.append(f"{self.skill_name} L{self.skill_level or '?'}")
        return " | ".join(parts)


def generate_canonical_name(step: int, timestamp: Optional[datetime] = None) -> str:
    """
    Generate canonical checkpoint name.

    Format: checkpoint-{step}-{YYYYMMDD}-{HHMM}

    Examples:
        checkpoint-190000-20251127-1430
        checkpoint-200000-20251127-1845
    """
    ts = timestamp or datetime.now()
    date_str = ts.strftime("%Y%m%d")
    time_str = ts.strftime("%H%M")
    return f"checkpoint-{step}-{date_str}-{time_str}"


def parse_checkpoint_name(name: str) -> Dict[str, Any]:
    """
    Parse a checkpoint name (canonical or legacy).

    Returns:
        {
            "step": int,
            "date": Optional[str],  # YYYYMMDD
            "time": Optional[str],  # HHMM
            "is_canonical": bool,
        }
    """
    # Remove path components
    name = Path(name).name

    # Try canonical format: checkpoint-{step}-{YYYYMMDD}-{HHMM}
    parts = name.split("-")
    if len(parts) >= 4 and parts[0] == "checkpoint":
        try:
            step = int(parts[1])
            date = parts[2] if len(parts[2]) == 8 else None
            time = parts[3] if len(parts) > 3 and len(parts[3]) == 4 else None
            return {
                "step": step,
                "date": date,
                "time": time,
                "is_canonical": date is not None,
            }
        except ValueError:
            pass

    # Try legacy format: checkpoint-{step}
    if name.startswith("checkpoint-"):
        try:
            step = int(name.replace("checkpoint-", "").split("-")[0])
            return {
                "step": step,
                "date": None,
                "time": None,
                "is_canonical": False,
            }
        except ValueError:
            pass

    return {"step": 0, "date": None, "time": None, "is_canonical": False}


def extract_step(name: str) -> int:
    """Extract step number from checkpoint name."""
    return parse_checkpoint_name(name)["step"]


class CheckpointLedger:
    """
    Central registry for checkpoint metadata and stats.

    Maintains:
        1. Central index at status/checkpoint_ledger.json
        2. Sidecar files in each checkpoint directory

    Thread-safe for concurrent access.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the ledger.

        Args:
            base_dir: Base training directory (default: auto-detect)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Auto-detect
            self.base_dir = Path(__file__).parent.parent

        self.status_dir = self.base_dir / "status"
        self.index_path = self.status_dir / LEDGER_INDEX
        self.checkpoints_dir = self.base_dir / "current_model"

        self._lock = threading.Lock()
        self._index: Dict[int, CheckpointRecord] = {}
        self._index_mtime: float = 0.0  # Track file modification time

        self._load_index()

    def _get_local_device_id(self) -> Optional[str]:
        """Get the device_id for the local machine (cached)."""
        if not hasattr(self, '_local_device_id'):
            self._local_device_id = None
            try:
                from core.hosts import get_local_host
                local = get_local_host()
                if local and local.device_id:
                    self._local_device_id = local.device_id
            except Exception:
                pass

            # Fallback: try to detect from hostname
            if not self._local_device_id:
                import socket
                hostname = socket.gethostname().lower()
                if "4090" in hostname or "trainer" in hostname:
                    self._local_device_id = "trainer4090"
                elif "3090" in hostname or "inference" in hostname:
                    self._local_device_id = "inference3090"
                else:
                    self._local_device_id = "trainer4090"  # Default

        return self._local_device_id

    def _load_index(self, force: bool = False):
        """Load the central index."""
        if not self.index_path.exists():
            return

        # Check if file changed (skip reload if unchanged)
        try:
            current_mtime = self.index_path.stat().st_mtime
            if not force and current_mtime == self._index_mtime:
                return  # No changes
        except OSError:
            return

        try:
            with open(self.index_path) as f:
                data = json.load(f)

            new_index: Dict[int, CheckpointRecord] = {}
            for step_str, record_data in data.get("checkpoints", {}).items():
                try:
                    record = CheckpointRecord.from_dict(record_data)
                    new_index[record.step] = record
                except Exception as e:
                    logger.warning(f"Failed to load record for step {step_str}: {e}")

            self._index = new_index
            self._index_mtime = current_mtime
            logger.debug(f"Loaded {len(self._index)} checkpoint records from ledger")
        except Exception as e:
            logger.warning(f"Failed to load ledger index: {e}")

    def _ensure_fresh(self):
        """Reload index if file changed on disk. Call before reads."""
        self._load_index(force=False)

    def _save_index(self):
        """
        Save the central index atomically.

        Uses file locking and atomic write to prevent race conditions
        when multiple processes update the ledger.
        """
        import fcntl
        import tempfile

        self.status_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self.index_path.with_suffix(".lock")

        # Acquire file lock for atomic read-modify-write
        with open(lock_file, "w") as lock_fd:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read the file to merge any changes from other processes
                if self.index_path.exists():
                    try:
                        with open(self.index_path) as f:
                            disk_data = json.load(f)
                        # Merge disk data with our changes
                        for step_str, record_data in disk_data.get("checkpoints", {}).items():
                            step = int(step_str)
                            if step not in self._index:
                                # New record from disk that we don't have
                                try:
                                    self._index[step] = CheckpointRecord.from_dict(record_data)
                                except Exception:
                                    pass
                            else:
                                # Merge locations from disk with our locations
                                disk_locations = set(record_data.get("locations", []))
                                our_locations = set(self._index[step].locations)
                                merged = list(disk_locations | our_locations)
                                self._index[step].locations = merged
                    except Exception as e:
                        logger.warning(f"Could not merge from disk: {e}")

                # Prepare data
                data = {
                    "version": "1.0.0",
                    "updated_at": datetime.now().isoformat(),
                    "checkpoint_count": len(self._index),
                    "checkpoints": {
                        str(step): record.to_dict()
                        for step, record in sorted(self._index.items())
                    },
                }

                # Atomic write: write to temp file then rename
                fd, temp_path = tempfile.mkstemp(
                    dir=self.status_dir,
                    prefix="ledger_",
                    suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f, indent=2)
                    os.replace(temp_path, self.index_path)
                    self._index_mtime = self.index_path.stat().st_mtime
                except Exception:
                    os.unlink(temp_path)
                    raise
            finally:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

    def _write_sidecar(self, checkpoint_dir: Path, record: CheckpointRecord):
        """Write sidecar file to checkpoint directory."""
        sidecar_path = checkpoint_dir / LEDGER_SIDECAR
        try:
            with open(sidecar_path, "w") as f:
                json.dump(record.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write sidecar to {checkpoint_dir}: {e}")

    def _read_sidecar(self, checkpoint_dir: Path) -> Optional[CheckpointRecord]:
        """Read sidecar file from checkpoint directory."""
        sidecar_path = checkpoint_dir / LEDGER_SIDECAR
        if not sidecar_path.exists():
            return None

        try:
            with open(sidecar_path) as f:
                data = json.load(f)
            return CheckpointRecord.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to read sidecar from {checkpoint_dir}: {e}")
            return None

    def record(
        self,
        step: int,
        path: str,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch: Optional[float] = None,
        accuracy: Optional[float] = None,
        perplexity: Optional[float] = None,
        training_file: Optional[str] = None,
        skill_name: Optional[str] = None,
        skill_level: Optional[int] = None,
        total_examples_trained: Optional[int] = None,
        rename: bool = True,
    ) -> CheckpointRecord:
        """
        Record a checkpoint in the ledger.

        This should be called from the on_save callback immediately after
        the Trainer saves a checkpoint.

        Args:
            step: Global step number
            path: Path to checkpoint directory
            train_loss: Training loss at save time
            val_loss: Validation loss at save time
            learning_rate: Learning rate at save time
            epoch: Current epoch (fractional)
            accuracy: Evaluation accuracy (if available)
            perplexity: Model perplexity (if available)
            training_file: Name of file being trained
            skill_name: Current skill being trained
            skill_level: Current skill level
            total_examples_trained: Total examples seen
            rename: Whether to rename to canonical name

        Returns:
            CheckpointRecord with all metadata
        """
        timestamp = datetime.now()
        canonical_name = generate_canonical_name(step, timestamp)

        checkpoint_path = Path(path)
        # Always store absolute paths in the ledger
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.base_dir / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()

        # Calculate size
        size_bytes = None
        if checkpoint_path.exists():
            size_bytes = sum(
                f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()
            )

        # Check for optimizer
        has_optimizer = (checkpoint_path / "optimizer.pt").exists()

        # Get local device ID for location tracking
        local_device_id = self._get_local_device_id()

        record = CheckpointRecord(
            step=step,
            timestamp=timestamp.isoformat(),
            path=str(checkpoint_path),
            canonical_name=canonical_name,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epoch=epoch,
            accuracy=accuracy,
            perplexity=perplexity,
            training_file=training_file,
            skill_name=skill_name,
            skill_level=skill_level,
            total_examples_trained=total_examples_trained,
            size_bytes=size_bytes,
            has_optimizer=has_optimizer,
            last_used=timestamp.isoformat(),
            locations=[local_device_id] if local_device_id else [],
        )

        with self._lock:
            # Write sidecar
            self._write_sidecar(checkpoint_path, record)

            # Rename to canonical name if requested
            if rename and checkpoint_path.exists():
                canonical_path = checkpoint_path.parent / canonical_name
                if checkpoint_path != canonical_path and not canonical_path.exists():
                    try:
                        shutil.move(str(checkpoint_path), str(canonical_path))
                        record.path = str(canonical_path)
                        logger.info(f"Renamed checkpoint: {checkpoint_path.name} -> {canonical_name}")
                    except Exception as e:
                        logger.warning(f"Failed to rename checkpoint: {e}")

            # Update index
            self._index[step] = record
            self._save_index()

        # Update campaign peak metrics (outside lock)
        self._update_campaign_peak_loss(train_loss, val_loss)

        # Sync to VaultKeeper catalog (outside lock)
        self._sync_to_vault(record)

        # Capture curriculum state at save time (v1.1.0)
        self._capture_curriculum_state_at_save(record)

        return record

    def _capture_curriculum_state_at_save(self, record: CheckpointRecord) -> None:
        """
        Capture curriculum state (skill levels, effort) at checkpoint save.

        Non-fatal: errors are logged but don't affect checkpoint saving.
        """
        try:
            skill_levels: Dict[str, int] = {}
            skill_effort: Dict[str, float] = {}

            # Get skill levels from curriculum state
            curriculum_state_file = self.base_dir / "data_manager" / "curriculum_state.json"
            if curriculum_state_file.exists():
                with open(curriculum_state_file) as f:
                    state = json.load(f)
                for skill_id, skill_state in state.get("skills", {}).items():
                    skill_levels[skill_id] = skill_state.get("current_level", 1)

            # Get effort from active campaign
            try:
                from guild.campaigns.loader import load_active_campaign
                campaign = load_active_campaign()
                if campaign:
                    skill_effort = dict(campaign.skill_effort)
            except Exception:
                pass

            # Update record (no lock needed - we just created it)
            if skill_levels or skill_effort:
                record.skill_levels_at_save = skill_levels
                record.skill_effort_at_save = skill_effort
                # Re-save to include the new data
                with self._lock:
                    self._save_index()

        except Exception as e:
            logger.debug(f"Could not capture curriculum state at save: {e}")

    def _update_campaign_peak_loss(
        self,
        train_loss: Optional[float],
        val_loss: Optional[float]
    ):
        """
        Update campaign peak loss metrics if these are new personal bests.

        Safe to call - never throws, just logs warnings.
        """
        try:
            from guild.campaigns.loader import load_active_campaign

            campaign = load_active_campaign()
            if campaign is None:
                return

            # Track both train and validation loss
            if train_loss is not None:
                is_new_peak = campaign.update_peak_metric(
                    "lowest_loss", train_loss, lower_is_better=True
                )
                if is_new_peak:
                    logger.info(
                        f"[Campaign {campaign.id}] New lowest loss: {train_loss:.4f}"
                    )

            if val_loss is not None:
                is_new_peak = campaign.update_peak_metric(
                    "lowest_val_loss", val_loss, lower_is_better=True
                )
                if is_new_peak:
                    logger.info(
                        f"[Campaign {campaign.id}] New lowest val loss: {val_loss:.4f}"
                    )

        except Exception as e:
            # Don't crash checkpoint saving because peak tracking failed
            logger.warning(f"Failed to update peak loss: {e}")

    def _sync_to_vault(self, record: CheckpointRecord) -> None:
        """
        Sync checkpoint record to VaultKeeper catalog.

        This keeps the VaultKeeper catalog in sync with the Ledger.
        Converts device_id locations to stronghold locations via the
        device mapping, then registers the asset with VaultKeeper.

        Non-fatal: errors are logged but don't affect checkpoint saving.
        """
        try:
            from vault.keeper import get_vault_keeper
            from vault.device_mapping import get_mapping
            from vault.assets import Asset, AssetType, AssetLocation, LocationStatus

            keeper = get_vault_keeper()
            mapping = get_mapping()

            # Convert device_id locations to stronghold locations
            locations = []
            local_device_id = self._get_local_device_id()

            for device_id in record.locations:
                try:
                    stronghold = mapping.device_to_stronghold(device_id)
                    base_path = mapping.get_base_path(device_id)

                    # Get relative path from base
                    try:
                        rel_path = str(Path(record.path).relative_to(base_path))
                    except ValueError:
                        # Path not relative to base, use as-is
                        rel_path = record.path

                    locations.append(AssetLocation(
                        stronghold=stronghold,
                        path=rel_path,
                        status=LocationStatus.VERIFIED,
                        is_primary=(device_id == local_device_id),
                    ))
                except KeyError:
                    # Unknown device_id, skip
                    continue

            if not locations:
                return

            # Generate vault asset_id from step
            asset_id = f"checkpoint_{record.step}"

            # Create and register asset
            asset = Asset(
                asset_id=asset_id,
                asset_type=AssetType.CHECKPOINT,
                name=record.canonical_name,
                size_bytes=record.size_bytes or 0,
                locations=locations,
                metadata={
                    "step": record.step,
                    "train_loss": record.train_loss,
                    "canonical_name": record.canonical_name,
                    "ledger_synced": True,
                },
            )
            keeper.register(asset)
            logger.debug(f"Synced checkpoint {record.step} to VaultKeeper catalog")

        except Exception as e:
            # Non-fatal: don't crash checkpoint saving
            logger.debug(f"VaultKeeper sync failed (non-fatal): {e}")

    def _add_vault_location(self, step: int, device_id: str) -> None:
        """
        Add a location to VaultKeeper when Ledger location is added.

        Called after record_usage() to keep VaultKeeper in sync.
        """
        try:
            from vault.keeper import get_vault_keeper
            from vault.device_mapping import get_mapping

            keeper = get_vault_keeper()
            mapping = get_mapping()

            asset_id = f"checkpoint_{step}"
            stronghold = mapping.device_to_stronghold(device_id)
            record = self.get(step)

            if record:
                base_path = mapping.get_base_path(device_id)
                try:
                    rel_path = str(Path(record.path).relative_to(base_path))
                except ValueError:
                    rel_path = record.path

                keeper.add_location(asset_id, stronghold, rel_path)
                logger.debug(f"Added VaultKeeper location: {asset_id} @ {stronghold}")

        except Exception as e:
            logger.debug(f"VaultKeeper location add failed (non-fatal): {e}")

    def _remove_vault_location(self, step: int, device_id: str) -> None:
        """
        Remove a location from VaultKeeper when Ledger location is removed.

        Called after remove_location() to keep VaultKeeper in sync.
        """
        try:
            from vault.keeper import get_vault_keeper
            from vault.device_mapping import get_mapping

            keeper = get_vault_keeper()
            mapping = get_mapping()

            asset_id = f"checkpoint_{step}"
            stronghold = mapping.device_to_stronghold(device_id)
            record = self.get(step)

            if record:
                base_path = mapping.get_base_path(device_id)
                try:
                    rel_path = str(Path(record.path).relative_to(base_path))
                except ValueError:
                    rel_path = record.path

                keeper.remove_location(asset_id, stronghold, rel_path)
                logger.debug(f"Removed VaultKeeper location: {asset_id} @ {stronghold}")

        except Exception as e:
            logger.debug(f"VaultKeeper location remove failed (non-fatal): {e}")

    def get(self, step: int) -> Optional[CheckpointRecord]:
        """Get record for a specific step."""
        self._ensure_fresh()
        return self._index.get(step)

    def get_latest(self) -> Optional[CheckpointRecord]:
        """Get the most recent checkpoint record."""
        self._ensure_fresh()
        if not self._index:
            return None
        max_step = max(self._index.keys())
        return self._index[max_step]

    def get_best(
        self,
        metric: str = "train_loss",
        lower_is_better: bool = True,
        hero_id: Optional[str] = None,
    ) -> Optional[CheckpointRecord]:
        """
        Get the best checkpoint by a metric.

        Args:
            metric: Metric name (train_loss, val_loss, accuracy, perplexity)
            lower_is_better: Whether lower values are better
            hero_id: Only consider checkpoints for this hero (optional)

        Returns:
            Best checkpoint record or None
        """
        self._ensure_fresh()
        candidates = []
        for record in self._index.values():
            # Filter by hero_id if specified
            if hero_id is not None and record.hero_id != hero_id:
                # Also check path-based inference for legacy records
                if record.hero_id is None:
                    from core.eval_dynamics import infer_hero_from_path
                    inferred = infer_hero_from_path(record.path)
                    if inferred != hero_id:
                        continue
                else:
                    continue

            value = getattr(record, metric, None)
            if value is not None:
                candidates.append((value, record))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=not lower_is_better)
        return candidates[0][1]

    def list_by_hero(self, hero_id: str, limit: Optional[int] = None) -> List[CheckpointRecord]:
        """
        List checkpoints for a specific hero, newest first.

        Args:
            hero_id: Hero ID to filter by
            limit: Maximum number of records to return

        Returns:
            List of checkpoint records for that hero
        """
        self._ensure_fresh()
        records = []
        for record in self._index.values():
            # Direct match
            if record.hero_id == hero_id:
                records.append(record)
                continue

            # Legacy records: infer from path
            if record.hero_id is None:
                from core.eval_dynamics import infer_hero_from_path
                inferred = infer_hero_from_path(record.path)
                if inferred == hero_id:
                    records.append(record)

        records.sort(key=lambda r: r.step, reverse=True)
        if limit:
            records = records[:limit]
        return records

    def list_all(self, limit: Optional[int] = None) -> List[CheckpointRecord]:
        """List all checkpoint records, newest first."""
        self._ensure_fresh()
        records = sorted(self._index.values(), key=lambda r: r.step, reverse=True)
        if limit:
            records = records[:limit]
        return records

    def list_by_skill(self, skill_name: str) -> List[CheckpointRecord]:
        """List checkpoints for a specific skill."""
        self._ensure_fresh()
        return [
            r for r in self._index.values()
            if r.skill_name == skill_name
        ]

    def record_usage(self, step: int, device_id: str) -> bool:
        """
        Record that a checkpoint was used on a device.

        Updates last_used timestamp and adds device to locations list.
        Called when checkpoint is loaded for inference, eval, etc.

        Args:
            step: Checkpoint step number
            device_id: Device ID (e.g., "trainer4090", "inference3090")

        Returns:
            True if updated, False if checkpoint not found
        """
        self._ensure_fresh()

        with self._lock:
            record = self._index.get(step)
            if not record:
                logger.warning(f"Cannot record usage: checkpoint {step} not in ledger")
                return False

            # Update last_used
            record.last_used = datetime.now().isoformat()

            # Add device to locations if not present
            if device_id not in record.locations:
                record.locations.append(device_id)

            self._save_index()
            logger.debug(f"Recorded usage: checkpoint {step} on {device_id}")

        # Sync location to VaultKeeper (outside lock)
        self._add_vault_location(step, device_id)
        return True

    def remove_location(self, step: int, device_id: str) -> bool:
        """
        Remove a device from a checkpoint's locations.

        Called when a checkpoint is deleted from a device.

        Args:
            step: Checkpoint step number
            device_id: Device ID to remove

        Returns:
            True if updated, False if checkpoint not found
        """
        self._ensure_fresh()

        with self._lock:
            record = self._index.get(step)
            if not record:
                return False

            if device_id in record.locations:
                record.locations.remove(device_id)
                self._save_index()
                logger.debug(f"Removed location: checkpoint {step} from {device_id}")

        # Sync removal to VaultKeeper (outside lock)
        self._remove_vault_location(step, device_id)
        return True

    def cleanup_stale_entries(self, dry_run: bool = False) -> int:
        """
        Remove ledger entries for checkpoints that no longer exist anywhere.

        A checkpoint is considered stale if:
        - Its path doesn't exist AND
        - It has no locations listed (or all locations are unknown)

        This cleans up historical entries for deleted checkpoints.

        Args:
            dry_run: If True, only count without deleting

        Returns:
            Number of entries removed (or would be removed if dry_run)
        """
        from pathlib import Path
        import fcntl
        import tempfile

        self._ensure_fresh()
        stale_steps = []

        with self._lock:
            for step, record in list(self._index.items()):
                # Check if checkpoint exists locally
                path_exists = record.path and Path(record.path).exists()

                # Check if it has any known locations
                has_locations = bool(record.locations) and any(
                    loc not in ('unknown', '') for loc in record.locations
                )

                # If doesn't exist locally and has no known locations, it's stale
                if not path_exists and not has_locations:
                    stale_steps.append(step)

            if not dry_run and stale_steps:
                for step in stale_steps:
                    del self._index[step]

                # Direct save WITHOUT merge (cleanup is authoritative)
                lock_file = self.index_path.with_suffix(".lock")
                with open(lock_file, "w") as lock_fd:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                    try:
                        data = {
                            "version": "1.0.0",
                            "updated_at": datetime.now().isoformat(),
                            "checkpoint_count": len(self._index),
                            "checkpoints": {
                                str(step): record.to_dict()
                                for step, record in sorted(self._index.items())
                            },
                        }
                        fd, temp_path = tempfile.mkstemp(
                            dir=self.status_dir,
                            prefix="ledger_",
                            suffix=".tmp"
                        )
                        try:
                            with os.fdopen(fd, "w") as f:
                                json.dump(data, f, indent=2)
                            os.replace(temp_path, self.index_path)
                        except Exception:
                            os.unlink(temp_path)
                            raise
                    finally:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

                logger.info(f"Cleaned up {len(stale_steps)} stale ledger entries")

        return len(stale_steps)

    def verify_local_checkpoints(self, device_id: str, dry_run: bool = False) -> int:
        """
        Verify ledger entries for a device match actual filesystem state.

        Removes the device from locations for checkpoints that don't exist locally.
        Uses direct save to avoid merge logic re-adding locations.

        Args:
            device_id: Device to verify
            dry_run: If True, only count without modifying

        Returns:
            Number of locations removed
        """
        from pathlib import Path
        import fcntl
        import tempfile

        self._ensure_fresh()
        removed = 0

        with self._lock:
            records = [r for r in self._index.values() if device_id in r.locations]
            for record in records:
                if record.path and not Path(record.path).exists():
                    if not dry_run:
                        record.locations.remove(device_id)
                    removed += 1

            if not dry_run and removed > 0:
                # Direct save WITHOUT merge
                lock_file = self.index_path.with_suffix(".lock")
                with open(lock_file, "w") as lock_fd:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                    try:
                        data = {
                            "version": "1.0.0",
                            "updated_at": datetime.now().isoformat(),
                            "checkpoint_count": len(self._index),
                            "checkpoints": {
                                str(step): record.to_dict()
                                for step, record in sorted(self._index.items())
                            },
                        }
                        fd, temp_path = tempfile.mkstemp(
                            dir=self.status_dir,
                            prefix="ledger_",
                            suffix=".tmp"
                        )
                        try:
                            with os.fdopen(fd, "w") as f:
                                json.dump(data, f, indent=2)
                            os.replace(temp_path, self.index_path)
                        except Exception:
                            os.unlink(temp_path)
                            raise
                    finally:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)

                logger.info(f"Removed {device_id} from {removed} non-existent checkpoints")

        return removed

    def list_by_device(self, device_id: str, sort_by_usage: bool = True) -> List[CheckpointRecord]:
        """
        List checkpoints that exist on a specific device.

        Args:
            device_id: Device ID to filter by
            sort_by_usage: If True, sort by last_used (most recent first)

        Returns:
            List of checkpoint records on this device
        """
        self._ensure_fresh()
        records = [r for r in self._index.values() if device_id in r.locations]

        if sort_by_usage:
            # Sort by last_used, with None values last
            records.sort(
                key=lambda r: r.last_used or "1970-01-01T00:00:00",
                reverse=True
            )
        else:
            records.sort(key=lambda r: r.step, reverse=True)

        return records

    def get_retention_candidates(
        self,
        device_id: str,
        keep_count: int,
    ) -> List[CheckpointRecord]:
        """
        Get checkpoints that should be deleted from a device.

        Returns checkpoints beyond the keep_count, sorted by least recently used.
        Never returns checkpoints that are the only copy (is_vault locations).

        Args:
            device_id: Device to check
            keep_count: Number of checkpoints to keep

        Returns:
            List of checkpoint records that can be deleted
        """
        records = self.list_by_device(device_id, sort_by_usage=True)

        if len(records) <= keep_count:
            return []

        # The most recently used are kept, older ones are candidates
        candidates = records[keep_count:]

        # Filter out checkpoints that only exist on this device (safety)
        # NAS device_id is "synology_data" per hosts.json
        safe_candidates = [
            r for r in candidates
            if len(r.locations) > 1 or "synology_data" in r.locations
        ]

        return safe_candidates

    def scan_orphans(self) -> List[Path]:
        """
        Find checkpoint directories not in the ledger.

        These are checkpoints that exist on disk but weren't recorded
        (e.g., from before the ledger was implemented).
        """
        orphans = []
        if not self.checkpoints_dir.exists():
            return orphans

        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                parsed = parse_checkpoint_name(item.name)
                if parsed["step"] and parsed["step"] not in self._index:
                    orphans.append(item)

        return orphans

    def adopt_orphan(self, checkpoint_dir: Path) -> Optional[CheckpointRecord]:
        """
        Adopt an orphan checkpoint into the ledger.

        Reads what info we can from the checkpoint and creates a record.
        """
        if not checkpoint_dir.exists():
            return None

        parsed = parse_checkpoint_name(checkpoint_dir.name)
        step = parsed["step"]
        if not step:
            return None

        # Try to read existing sidecar
        existing = self._read_sidecar(checkpoint_dir)
        if existing:
            with self._lock:
                self._index[step] = existing
                self._save_index()
            return existing

        # Try to get stats from trainer_state.json
        train_loss = None
        epoch = None
        state_file = checkpoint_dir / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                # Get last logged loss
                for entry in reversed(state.get("log_history", [])):
                    if "loss" in entry:
                        train_loss = entry["loss"]
                        break
                epoch = state.get("epoch")
            except Exception:
                pass

        # Create record with available info
        timestamp = datetime.fromtimestamp(checkpoint_dir.stat().st_mtime)

        return self.record(
            step=step,
            path=str(checkpoint_dir),
            train_loss=train_loss,
            epoch=epoch,
            rename=False,  # Don't rename existing checkpoints
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        self._ensure_fresh()
        records = list(self._index.values())

        if not records:
            return {
                "count": 0,
                "latest_step": None,
                "best_loss": None,
                "total_size_gb": 0,
            }

        losses = [r.train_loss for r in records if r.train_loss is not None]
        sizes = [r.size_bytes for r in records if r.size_bytes is not None]

        return {
            "count": len(records),
            "latest_step": max(r.step for r in records),
            "oldest_step": min(r.step for r in records),
            "best_loss": min(losses) if losses else None,
            "worst_loss": max(losses) if losses else None,
            "total_size_gb": round(sum(sizes) / (1024 ** 3), 2) if sizes else 0,
            "skills_trained": list(set(r.skill_name for r in records if r.skill_name)),
        }

    def run_health_check(self, fix_issues: bool = True) -> Dict[str, Any]:
        """
        Run health check on the ledger and optionally fix issues.

        Checks for:
        1. Orphan checkpoints (on disk but not in ledger)
        2. Stale entries (in ledger but path doesn't exist)

        Args:
            fix_issues: If True, adopt orphans and remove stale entries

        Returns:
            Dict with health check results
        """
        result = {
            "healthy": True,
            "orphans_found": 0,
            "orphans_adopted": 0,
            "stale_found": 0,
            "stale_removed": 0,
            "ledger_count": len(self._index),
            "issues": [],
        }

        # Check for orphans
        orphans = self.scan_orphans()
        result["orphans_found"] = len(orphans)

        if orphans:
            result["healthy"] = False
            result["issues"].append(f"{len(orphans)} orphan checkpoints found")

            if fix_issues:
                for orphan in orphans:
                    try:
                        record = self.adopt_orphan(orphan)
                        if record:
                            result["orphans_adopted"] += 1
                            logger.info(f"Adopted orphan checkpoint: {orphan.name}")
                    except Exception as e:
                        logger.warning(f"Failed to adopt orphan {orphan}: {e}")

        # Check for stale entries (path doesn't exist)
        stale = []
        for step, record in list(self._index.items()):
            if record.path and not Path(record.path).exists():
                # Check if it has other known locations
                if not record.locations:
                    stale.append(step)

        result["stale_found"] = len(stale)

        if stale:
            result["healthy"] = False
            result["issues"].append(f"{len(stale)} stale ledger entries found")

            if fix_issues:
                for step in stale:
                    # Use cleanup method to properly remove
                    with self._lock:
                        if step in self._index:
                            del self._index[step]
                            result["stale_removed"] += 1

                if result["stale_removed"] > 0:
                    self._save_index_direct()
                    logger.info(f"Removed {result['stale_removed']} stale entries")

        result["ledger_count"] = len(self._index)

        if result["healthy"]:
            logger.info(f"Ledger health check passed ({result['ledger_count']} checkpoints)")
        else:
            logger.warning(f"Ledger health check: {', '.join(result['issues'])}")

        return result

    # =========================================================================
    # SKILL METRICS INTEGRATION (v1.1.0)
    # =========================================================================

    def sync_skill_metrics(self, step: int) -> bool:
        """
        Sync skill metrics from EvaluationLedger to this checkpoint.

        Aggregates all eval results for this checkpoint and updates:
        - skill_metrics: per-skill accuracy, level, eval_count
        - linked_eval_ids: list of eval keys

        Args:
            step: Checkpoint step number

        Returns:
            True if updated, False if checkpoint not found
        """
        from core.evaluation_ledger import get_eval_ledger

        self._ensure_fresh()
        record = self._index.get(step)
        if not record:
            return False

        eval_ledger = get_eval_ledger()
        evals = eval_ledger.get_by_checkpoint(step)

        if not evals:
            return True  # No evals to sync, but checkpoint exists

        # Aggregate by skill
        skill_metrics: Dict[str, Dict[str, Any]] = {}
        linked_ids: List[str] = []

        for eval_rec in evals:
            skill = eval_rec.skill
            linked_ids.append(eval_rec.key)

            if skill not in skill_metrics:
                skill_metrics[skill] = {
                    "accuracy": 0.0,
                    "best_accuracy": 0.0,
                    "level": 0,
                    "max_level": 0,
                    "eval_count": 0,
                    "correct": 0,
                    "total": 0,
                }

            sm = skill_metrics[skill]
            sm["eval_count"] += 1
            sm["correct"] += eval_rec.correct
            sm["total"] += eval_rec.total
            sm["max_level"] = max(sm["max_level"], eval_rec.level)
            sm["best_accuracy"] = max(sm["best_accuracy"], eval_rec.accuracy)

            # Track level with highest accuracy for this skill
            if eval_rec.accuracy >= sm["accuracy"]:
                sm["accuracy"] = eval_rec.accuracy
                sm["level"] = eval_rec.level

        with self._lock:
            record.skill_metrics = skill_metrics
            record.linked_eval_ids = linked_ids
            self._save_index()

        logger.info(f"Synced skill metrics for step {step}: {list(skill_metrics.keys())}")
        return True

    def capture_curriculum_state(self, step: int) -> bool:
        """
        Capture current curriculum state (skill levels, effort) at checkpoint.

        Reads from:
        - CurriculumManager for current skill levels
        - Campaign for skill effort

        Args:
            step: Checkpoint step number

        Returns:
            True if updated, False if checkpoint not found
        """
        self._ensure_fresh()
        record = self._index.get(step)
        if not record:
            return False

        skill_levels: Dict[str, int] = {}
        skill_effort: Dict[str, float] = {}

        # Get skill levels from curriculum state
        try:
            curriculum_state_file = self.base_dir / "data_manager" / "curriculum_state.json"
            if curriculum_state_file.exists():
                with open(curriculum_state_file) as f:
                    state = json.load(f)
                for skill_id, skill_state in state.get("skills", {}).items():
                    skill_levels[skill_id] = skill_state.get("current_level", 1)
        except Exception as e:
            logger.debug(f"Could not read curriculum state: {e}")

        # Get effort from active campaign
        try:
            from guild.campaigns.loader import load_active_campaign
            campaign = load_active_campaign()
            if campaign:
                skill_effort = dict(campaign.skill_effort)
        except Exception as e:
            logger.debug(f"Could not read campaign effort: {e}")

        with self._lock:
            record.skill_levels_at_save = skill_levels
            record.skill_effort_at_save = skill_effort
            self._save_index()

        logger.debug(f"Captured curriculum state for step {step}: levels={skill_levels}")
        return True

    def link_eval(self, step: int, skill: str, level: int, accuracy: float) -> bool:
        """
        Link an evaluation result to a checkpoint.

        Called after recording an eval to update the checkpoint's skill_metrics.
        More lightweight than full sync_skill_metrics().

        Args:
            step: Checkpoint step
            skill: Skill ID
            level: Skill level
            accuracy: Eval accuracy

        Returns:
            True if linked, False if checkpoint not found
        """
        self._ensure_fresh()
        record = self._index.get(step)
        if not record:
            return False

        eval_key = f"{step}:{skill}:{level}"

        with self._lock:
            # Initialize skill_metrics if needed
            if skill not in record.skill_metrics:
                record.skill_metrics[skill] = {
                    "accuracy": 0.0,
                    "best_accuracy": 0.0,
                    "level": 0,
                    "max_level": 0,
                    "eval_count": 0,
                    "correct": 0,
                    "total": 0,
                }

            sm = record.skill_metrics[skill]
            sm["eval_count"] += 1
            sm["max_level"] = max(sm["max_level"], level)
            sm["best_accuracy"] = max(sm["best_accuracy"], accuracy)

            if accuracy >= sm["accuracy"]:
                sm["accuracy"] = accuracy
                sm["level"] = level

            # Add to linked_eval_ids if not present
            if eval_key not in record.linked_eval_ids:
                record.linked_eval_ids.append(eval_key)

            self._save_index()

        return True

    def get_skill_metrics_summary(self, step: int) -> Dict[str, Any]:
        """
        Get a summary of skill metrics for a checkpoint.

        Returns:
            Dict with skill metrics or empty dict if checkpoint not found
        """
        record = self.get(step)
        if not record:
            return {}

        return {
            "step": step,
            "skill_metrics": record.skill_metrics,
            "skill_levels": record.skill_levels_at_save,
            "skill_effort": record.skill_effort_at_save,
            "eval_count": len(record.linked_eval_ids),
        }

    def get_best_for_skill(
        self,
        skill: str,
        metric: str = "accuracy",
        min_level: Optional[int] = None,
    ) -> Optional[CheckpointRecord]:
        """
        Get the best checkpoint for a specific skill.

        Args:
            skill: Skill ID (e.g., "sy", "bin")
            metric: Metric to compare ("accuracy" or "best_accuracy")
            min_level: Only consider checkpoints at or above this level

        Returns:
            Best checkpoint record or None
        """
        self._ensure_fresh()
        candidates = []

        for record in self._index.values():
            if skill not in record.skill_metrics:
                continue

            sm = record.skill_metrics[skill]
            if min_level and sm.get("max_level", 0) < min_level:
                continue

            value = sm.get(metric, 0.0)
            candidates.append((value, record))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


# =============================================================================
# SINGLETON
# =============================================================================

_ledger: Optional[CheckpointLedger] = None


def get_ledger(base_dir: Optional[str] = None) -> CheckpointLedger:
    """Get or create the checkpoint ledger singleton."""
    global _ledger
    if _ledger is None:
        _ledger = CheckpointLedger(base_dir)
    return _ledger


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def record_checkpoint(
    step: int,
    path: str,
    **kwargs,
) -> CheckpointRecord:
    """Record a checkpoint (convenience wrapper)."""
    return get_ledger().record(step=step, path=path, **kwargs)


def get_checkpoint_info(step: int) -> Optional[CheckpointRecord]:
    """Get checkpoint info by step."""
    return get_ledger().get(step)


def get_latest_checkpoint() -> Optional[CheckpointRecord]:
    """Get the latest checkpoint record."""
    return get_ledger().get_latest()


# =============================================================================
# REMOTE LEDGER CLIENT
# =============================================================================


class RemoteLedgerClient:
    """
    Client for querying the ledger API from remote hosts.

    This provides the same interface as CheckpointLedger but queries
    the VaultKeeper API over HTTP instead of reading local files.

    Usage:
        from core.checkpoint_ledger import RemoteLedgerClient
        from core.hosts import get_service_url

        # From 3090, query 4090's ledger
        client = RemoteLedgerClient(get_service_url("ledger"))

        # Same interface as local ledger
        latest = client.get_latest()
        best = client.get_best(metric="train_loss")
        all_records = client.list_all(limit=20)
    """

    def __init__(self, base_url: str, timeout: int = 10):
        """
        Initialize the remote ledger client.

        Args:
            base_url: Base URL of the ledger API (from get_service_url("ledger"))
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: Dict[int, CheckpointRecord] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 30  # Cache for 30 seconds

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make an HTTP request to the ledger API."""
        import urllib.request
        import urllib.error
        from urllib.parse import urlencode

        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.URLError as e:
            logger.warning(f"Remote ledger request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Remote ledger error: {e}")
            return None

    def _record_from_dict(self, data: Dict) -> Optional[CheckpointRecord]:
        """Convert API response to CheckpointRecord."""
        try:
            return CheckpointRecord.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to parse checkpoint record: {e}")
            return None

    def get(self, step: int) -> Optional[CheckpointRecord]:
        """Get record for a specific step."""
        # Check cache first
        if step in self._cache:
            return self._cache[step]

        data = self._request(f"/{step}")
        if data and "error" not in data:
            record = self._record_from_dict(data)
            if record:
                self._cache[step] = record
            return record
        return None

    def get_latest(self) -> Optional[CheckpointRecord]:
        """Get the most recent checkpoint record."""
        # First get the list to find latest step
        data = self._request("", {"limit": "1"})
        if data and data.get("checkpoints"):
            return self._record_from_dict(data["checkpoints"][0])
        return None

    def get_best(self, metric: str = "train_loss", lower_is_better: bool = True) -> Optional[CheckpointRecord]:
        """Get the best checkpoint by a metric."""
        params = {
            "metric": metric,
            "lower": "true" if lower_is_better else "false",
        }
        data = self._request("/best", params)
        if data and "error" not in data:
            return self._record_from_dict(data)
        return None

    def list_all(self, limit: Optional[int] = None) -> List[CheckpointRecord]:
        """List all checkpoint records, newest first."""
        params = {}
        if limit:
            params["limit"] = str(limit)

        data = self._request("", params)
        if data and data.get("checkpoints"):
            return [
                r for r in (self._record_from_dict(cp) for cp in data["checkpoints"])
                if r is not None
            ]
        return []

    def list_by_skill(self, skill_name: str) -> List[CheckpointRecord]:
        """List checkpoints for a specific skill."""
        data = self._request("", {"skill": skill_name})
        if data and data.get("checkpoints"):
            return [
                r for r in (self._record_from_dict(cp) for cp in data["checkpoints"])
                if r is not None
            ]
        return []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        data = self._request("/summary")
        return data if data else {
            "count": 0,
            "latest_step": None,
            "best_loss": None,
            "total_size_gb": 0,
            "error": "Failed to fetch summary",
        }

    def is_available(self) -> bool:
        """Check if the remote ledger API is reachable."""
        data = self._request("/summary")
        return data is not None and "error" not in data

    def refresh_cache(self):
        """Clear the local cache."""
        self._cache.clear()
        self._cache_time = None


def get_remote_ledger(host: Optional[str] = None) -> RemoteLedgerClient:
    """
    Get a remote ledger client.

    Args:
        host: Host ID (e.g., "4090") or full URL. If None, uses trainer host.

    Returns:
        RemoteLedgerClient configured for the specified host
    """
    # Try to use host registry
    try:
        from core.hosts import get_service_url, is_trainer_local

        # If we're on the trainer, just return local ledger
        if is_trainer_local() and host is None:
            logger.info("On trainer host, using local ledger")
            # Return a wrapper that uses local ledger but has same interface
            # Actually, caller should use get_ledger() directly in this case
            pass

        # Get URL from registry
        if host and not host.startswith("http"):
            url = get_service_url("ledger", host)
            if url:
                return RemoteLedgerClient(url)

    except ImportError:
        pass

    # Use provided URL or default
    if host and host.startswith("http"):
        return RemoteLedgerClient(host)

    # Default to trainer - get URL from hosts.json
    from core.hosts import get_service_url
    ledger_url = get_service_url("ledger")
    if not ledger_url:
        raise RuntimeError("No ledger service configured in hosts.json")
    return RemoteLedgerClient(ledger_url)


def get_ledger_client(base_dir: Optional[str] = None) -> "CheckpointLedger | RemoteLedgerClient":
    """
    Get the appropriate ledger client (local or remote).

    On the trainer host, returns the local CheckpointLedger.
    On other hosts, returns a RemoteLedgerClient.

    This is the recommended way to access the ledger from any host.
    """
    try:
        from core.hosts import is_trainer_local

        if is_trainer_local():
            return get_ledger(base_dir)
        else:
            return get_remote_ledger()
    except ImportError:
        # No host registry, assume local
        return get_ledger(base_dir)
