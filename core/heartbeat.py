"""
Heartbeat System - Worker liveness tracking.

Each worker (training daemon, eval workers, etc.) periodically writes
a heartbeat file. The WorldState aggregator reads these to determine
which workers are alive.

Usage:
    # In a worker process
    from core.heartbeat import HeartbeatWriter

    hb = HeartbeatWriter(
        worker_id="training_daemon",
        role="training",
        device="GPU0",
    )

    # In your main loop
    hb.beat(
        status="running",
        current_job_id="train_xyz",
        extra={"step": 183625, "it_per_sec": 0.21}
    )

    # Reading heartbeats (from aggregator)
    from core.heartbeat import get_all_heartbeats, get_heartbeat

    workers = get_all_heartbeats()
    for w in workers:
        print(f"{w.worker_id}: {w.status} (age: {w.age_seconds}s)")
"""

import json
import logging
import os
import socket
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# HEARTBEAT DATA
# =============================================================================

@dataclass
class Heartbeat:
    """A single heartbeat from a worker."""
    worker_id: str
    role: str                    # "training", "eval", "data_gen", etc.
    host: str
    pid: int
    device: Optional[str]        # "GPU0", "CPU", etc.
    status: str                  # "starting", "running", "idle", "stopping", "error"
    updated_at: str              # ISO timestamp
    current_job_id: Optional[str] = None
    current_job_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # Worker-specific data

    @property
    def age_seconds(self) -> float:
        """Seconds since last heartbeat."""
        try:
            updated = datetime.fromisoformat(self.updated_at)
            return (datetime.now() - updated).total_seconds()
        except Exception:
            return float("inf")

    @property
    def is_stale(self) -> bool:
        """Whether heartbeat is stale (>60s old)."""
        return self.age_seconds > 60

    @property
    def is_alive(self) -> bool:
        """Whether worker appears alive (heartbeat <30s old)."""
        return self.age_seconds < 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Heartbeat":
        return cls(
            worker_id=data.get("worker_id", "unknown"),
            role=data.get("role", "unknown"),
            host=data.get("host", "unknown"),
            pid=data.get("pid", 0),
            device=data.get("device"),
            status=data.get("status", "unknown"),
            updated_at=data.get("updated_at", ""),
            current_job_id=data.get("current_job_id"),
            current_job_type=data.get("current_job_type"),
            extra=data.get("extra", {}),
        )


# =============================================================================
# HEARTBEAT WRITER (for workers)
# =============================================================================

class HeartbeatWriter:
    """
    Writes heartbeat files for a worker process.

    Usage:
        hb = HeartbeatWriter("training_daemon", "training", "GPU0")
        hb.beat(status="running", current_job_id="job123")
    """

    def __init__(
        self,
        worker_id: str,
        role: str,
        device: Optional[str] = None,
        heartbeat_dir: Optional[Path] = None,
    ):
        self.worker_id = worker_id
        self.role = role
        self.device = device
        self.host = socket.gethostname()
        self.pid = os.getpid()

        if heartbeat_dir is None:
            try:
                from core.paths import get_base_dir
                heartbeat_dir = get_base_dir() / "status" / "heartbeats"
            except ImportError:
                heartbeat_dir = Path(__file__).parent.parent / "status" / "heartbeats"

        self.heartbeat_dir = heartbeat_dir
        self.heartbeat_file = heartbeat_dir / f"{worker_id}.json"
        self._last_beat = 0.0

    def beat(
        self,
        status: str = "running",
        current_job_id: Optional[str] = None,
        current_job_type: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        min_interval: float = 5.0,
    ) -> bool:
        """
        Write a heartbeat.

        Args:
            status: Worker status ("starting", "running", "idle", "stopping", "error")
            current_job_id: ID of current job being processed
            current_job_type: Type of current job
            extra: Additional worker-specific data (e.g., step, it_per_sec)
            min_interval: Minimum seconds between writes (to avoid I/O spam)

        Returns:
            True if heartbeat was written, False if skipped (too soon)
        """
        now = time.time()
        if now - self._last_beat < min_interval:
            return False

        hb = Heartbeat(
            worker_id=self.worker_id,
            role=self.role,
            host=self.host,
            pid=self.pid,
            device=self.device,
            status=status,
            updated_at=datetime.now().isoformat(),
            current_job_id=current_job_id,
            current_job_type=current_job_type,
            extra=extra or {},
        )

        try:
            self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
            with open(self.heartbeat_file, "w") as f:
                json.dump(hb.to_dict(), f, indent=2)
            self._last_beat = now
            return True
        except Exception as e:
            logger.error(f"Failed to write heartbeat: {e}")
            return False

    def stop(self):
        """Write a final heartbeat indicating shutdown."""
        self.beat(status="stopped", min_interval=0)

    def clear(self):
        """Remove the heartbeat file (on clean shutdown)."""
        try:
            if self.heartbeat_file.exists():
                self.heartbeat_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear heartbeat file: {e}")


# =============================================================================
# HEARTBEAT READER (for aggregator)
# =============================================================================

def _get_heartbeat_dir() -> Path:
    """Get heartbeat directory."""
    try:
        from core.paths import get_base_dir
        return get_base_dir() / "status" / "heartbeats"
    except ImportError:
        return Path(__file__).parent.parent / "status" / "heartbeats"


def get_heartbeat(worker_id: str) -> Optional[Heartbeat]:
    """Get heartbeat for a specific worker."""
    hb_file = _get_heartbeat_dir() / f"{worker_id}.json"

    if not hb_file.exists():
        return None

    try:
        with open(hb_file) as f:
            data = json.load(f)
        return Heartbeat.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to read heartbeat for {worker_id}: {e}")
        return None


def get_all_heartbeats() -> List[Heartbeat]:
    """Get all worker heartbeats."""
    hb_dir = _get_heartbeat_dir()

    if not hb_dir.exists():
        return []

    heartbeats = []
    for hb_file in hb_dir.glob("*.json"):
        try:
            with open(hb_file) as f:
                data = json.load(f)
            heartbeats.append(Heartbeat.from_dict(data))
        except Exception as e:
            logger.error(f"Failed to read heartbeat {hb_file}: {e}")

    return heartbeats


def get_live_workers() -> List[Heartbeat]:
    """Get only workers with fresh heartbeats (<30s old)."""
    return [hb for hb in get_all_heartbeats() if hb.is_alive]


def get_stale_workers() -> List[Heartbeat]:
    """Get workers with stale heartbeats (>60s old)."""
    return [hb for hb in get_all_heartbeats() if hb.is_stale]


def get_workers_by_role(role: str) -> List[Heartbeat]:
    """Get all workers with a specific role."""
    return [hb for hb in get_all_heartbeats() if hb.role == role]


def get_training_worker() -> Optional[Heartbeat]:
    """Get the training daemon heartbeat if alive."""
    workers = get_workers_by_role("training")
    alive = [w for w in workers if w.is_alive]
    return alive[0] if alive else None


def get_workers_on_device(device: str) -> List[Heartbeat]:
    """Get all workers using a specific device."""
    return [hb for hb in get_all_heartbeats() if hb.device == device]


def cleanup_stale_heartbeats(max_age_seconds: float = 300):
    """
    Remove heartbeat files older than max_age_seconds.

    Called periodically to clean up after crashed workers.
    """
    hb_dir = _get_heartbeat_dir()
    if not hb_dir.exists():
        return

    now = time.time()
    for hb_file in hb_dir.glob("*.json"):
        try:
            mtime = hb_file.stat().st_mtime
            if now - mtime > max_age_seconds:
                hb_file.unlink()
                logger.info(f"Cleaned up stale heartbeat: {hb_file.name}")
        except Exception as e:
            logger.error(f"Failed to cleanup heartbeat {hb_file}: {e}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heartbeat System")
    parser.add_argument("--list", action="store_true", help="List all heartbeats")
    parser.add_argument("--live", action="store_true", help="List only live workers")
    parser.add_argument("--stale", action="store_true", help="List only stale workers")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale heartbeats")
    parser.add_argument("--test", type=str, help="Write a test heartbeat with given worker_id")

    args = parser.parse_args()

    if args.cleanup:
        cleanup_stale_heartbeats()
        print("Cleaned up stale heartbeats")

    elif args.test:
        hb = HeartbeatWriter(args.test, role="test", device="CPU")
        hb.beat(status="running", current_job_id="test_job_123", min_interval=0)
        print(f"Wrote test heartbeat for {args.test}")

    elif args.live:
        workers = get_live_workers()
        if not workers:
            print("No live workers")
        else:
            print(f"Live Workers ({len(workers)}):")
            for w in workers:
                job = f" -> {w.current_job_id}" if w.current_job_id else ""
                print(f"  {w.worker_id} ({w.role}) on {w.device}: {w.status}{job}")

    elif args.stale:
        workers = get_stale_workers()
        if not workers:
            print("No stale workers")
        else:
            print(f"Stale Workers ({len(workers)}):")
            for w in workers:
                print(f"  {w.worker_id}: last seen {w.age_seconds:.0f}s ago")

    else:
        workers = get_all_heartbeats()
        if not workers:
            print("No heartbeats found")
        else:
            print(f"All Heartbeats ({len(workers)}):")
            print("-" * 60)
            for w in workers:
                status_icon = "🟢" if w.is_alive else ("🟡" if not w.is_stale else "🔴")
                job = f" -> {w.current_job_id}" if w.current_job_id else ""
                extra_info = ""
                if w.extra:
                    if "step" in w.extra:
                        extra_info += f" step={w.extra['step']}"
                    if "it_per_sec" in w.extra:
                        extra_info += f" {w.extra['it_per_sec']:.2f} it/s"
                print(f"{status_icon} {w.worker_id} ({w.role})")
                print(f"   Device: {w.device or 'N/A'} | Status: {w.status} | Age: {w.age_seconds:.0f}s")
                if job:
                    print(f"   Job: {w.current_job_id}")
                if extra_info:
                    print(f"   {extra_info.strip()}")
                print()
