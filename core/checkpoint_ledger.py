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

    # Physical properties
    size_bytes: Optional[int] = None
    has_optimizer: bool = True

    # Metadata
    created_by: str = "training_daemon"
    ledger_version: str = "1.0.0"

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

        self._load_index()

    def _load_index(self):
        """Load the central index."""
        if not self.index_path.exists():
            return

        try:
            with open(self.index_path) as f:
                data = json.load(f)

            for step_str, record_data in data.get("checkpoints", {}).items():
                try:
                    record = CheckpointRecord.from_dict(record_data)
                    self._index[record.step] = record
                except Exception as e:
                    logger.warning(f"Failed to load record for step {step_str}: {e}")

            logger.info(f"Loaded {len(self._index)} checkpoint records from ledger")
        except Exception as e:
            logger.warning(f"Failed to load ledger index: {e}")

    def _save_index(self):
        """Save the central index."""
        self.status_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "checkpoint_count": len(self._index),
            "checkpoints": {
                str(step): record.to_dict()
                for step, record in sorted(self._index.items())
            },
        }

        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

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

        # Calculate size
        size_bytes = None
        if checkpoint_path.exists():
            size_bytes = sum(
                f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()
            )

        # Check for optimizer
        has_optimizer = (checkpoint_path / "optimizer.pt").exists()

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

        return record

    def get(self, step: int) -> Optional[CheckpointRecord]:
        """Get record for a specific step."""
        return self._index.get(step)

    def get_latest(self) -> Optional[CheckpointRecord]:
        """Get the most recent checkpoint record."""
        if not self._index:
            return None
        max_step = max(self._index.keys())
        return self._index[max_step]

    def get_best(self, metric: str = "train_loss", lower_is_better: bool = True) -> Optional[CheckpointRecord]:
        """
        Get the best checkpoint by a metric.

        Args:
            metric: Metric name (train_loss, val_loss, accuracy, perplexity)
            lower_is_better: Whether lower values are better

        Returns:
            Best checkpoint record or None
        """
        candidates = []
        for record in self._index.values():
            value = getattr(record, metric, None)
            if value is not None:
                candidates.append((value, record))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=not lower_is_better)
        return candidates[0][1]

    def list_all(self, limit: Optional[int] = None) -> List[CheckpointRecord]:
        """List all checkpoint records, newest first."""
        records = sorted(self._index.values(), key=lambda r: r.step, reverse=True)
        if limit:
            records = records[:limit]
        return records

    def list_by_skill(self, skill_name: str) -> List[CheckpointRecord]:
        """List checkpoints for a specific skill."""
        return [
            r for r in self._index.values()
            if r.skill_name == skill_name
        ]

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

        # From 3090, query 4090's ledger
        client = RemoteLedgerClient("http://192.168.x.x:8767/api/ledger")

        # Same interface as local ledger
        latest = client.get_latest()
        best = client.get_best(metric="train_loss")
        all_records = client.list_all(limit=20)
    """

    def __init__(self, base_url: str, timeout: int = 10):
        """
        Initialize the remote ledger client.

        Args:
            base_url: Base URL of the ledger API (e.g., "http://192.168.x.x:8767/api/ledger")
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

    # Default to trainer
    return RemoteLedgerClient("http://192.168.x.x:8767/api/ledger")


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
