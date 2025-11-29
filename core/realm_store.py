"""
Realm State Store - Single Source of Truth for all Realm state.

This is THE canonical store for all state in the Realm. Instead of having
multiple disconnected sources (training_status.json, heartbeats/, events.jsonl,
battle_log.db), everything flows through here.

Architecture:
    PRODUCERS (write) -> RealmStateStore -> CONSUMERS (read)

    Producers: Training daemon, eval workers, job system, heartbeats
    Consumers: Tavern UI, CLI tools, monitoring, external APIs

Usage:
    from core.realm_store import get_store, update_training, get_realm_state

    # Producer updates state
    update_training(
        status="training",
        step=183220,
        loss=0.0277,
        file="train_binary.jsonl",
        speed=2.5,
        eta_seconds=120,
    )

    # Consumer reads state
    state = get_realm_state()
    print(state["training"]["step"])  # 183220

    # Or get specific section
    training = get_training_state()

Design Principles:
    1. Single file store (status/realm_state.json)
    2. Thread-safe updates with locking
    3. Typed sections (training, jobs, workers, hero, events)
    4. Timestamp on every update
    5. No duplicate data - one place for each piece of info
    6. Fast reads (JSON in memory, periodic flush)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# STATE SECTIONS (typed for clarity)
# =============================================================================

@dataclass
class TrainingState:
    """Current training status."""
    status: str = "idle"  # idle, training, paused, stopped
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    file: Optional[str] = None
    speed: Optional[float] = None  # steps/sec
    eta_seconds: Optional[int] = None
    strain: Optional[float] = None  # loss - floor
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class QueueState:
    """Job queue status."""
    depth: int = 0
    high_priority: int = 0
    normal_priority: int = 0
    low_priority: int = 0
    status: str = "ok"  # ok, low, empty, stale
    updated_at: Optional[str] = None


@dataclass
class WorkerState:
    """A single worker's state."""
    worker_id: str = ""
    role: str = ""  # training_daemon, eval_worker, etc.
    status: str = "unknown"  # running, idle, stale, stopped
    device: Optional[str] = None
    current_job: Optional[str] = None
    last_heartbeat: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeroState:
    """Hero (model) state."""
    name: str = "DIO"
    title: str = ""
    level: int = 0
    xp: int = 0
    campaign_id: Optional[str] = None
    current_skill: Optional[str] = None
    current_skill_level: int = 0
    updated_at: Optional[str] = None


@dataclass
class EventEntry:
    """A single event/log entry."""
    id: str = ""
    timestamp: str = ""
    kind: str = ""  # training_started, checkpoint_saved, etc.
    channel: str = "system"  # system, training, eval, jobs, guild
    severity: str = "info"  # info, success, warning, error
    message: str = ""
    icon: str = "📢"
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# REALM STATE STORE
# =============================================================================

class RealmStateStore:
    """
    Single source of truth for all Realm state.

    Thread-safe, file-backed, with in-memory caching.
    """

    # Channel icons
    CHANNEL_ICONS = {
        "system": "⚙️",
        "training": "📈",
        "eval": "📊",
        "jobs": "⚔️",
        "vault": "🗃️",
        "guild": "🏰",
        "debug": "🔧",
    }

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            try:
                from core.paths import get_base_dir
                store_path = get_base_dir() / "status" / "realm_store.json"
            except ImportError:
                store_path = Path(__file__).parent.parent / "status" / "realm_store.json"

        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._events: deque = deque(maxlen=100)  # Last 100 events in memory
        self._dirty = False
        self._last_flush = time.time()
        self._flush_interval = 1.0  # Flush every 1 second if dirty

        # Subscribers for real-time updates
        self._subscribers: List[Callable[[str, Dict], None]] = []

        # Load existing state
        self._load()

        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _load(self):
        """Load state from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                    self._state = data.get("state", {})
                    # Load events into deque
                    for e in data.get("events", []):
                        self._events.append(e)
            except Exception as e:
                logger.error(f"Failed to load realm store: {e}")
                self._state = {}

    def _save(self):
        """Save state to disk."""
        try:
            with open(self.store_path, "w") as f:
                json.dump({
                    "state": self._state,
                    "events": list(self._events),
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2, default=str)
            self._dirty = False
        except Exception as e:
            logger.error(f"Failed to save realm store: {e}")

    def _flush_loop(self):
        """Background thread that flushes dirty state to disk."""
        while True:
            time.sleep(0.5)
            if self._dirty and (time.time() - self._last_flush) >= self._flush_interval:
                with self._lock:
                    self._save()
                    self._last_flush = time.time()

    def _notify_subscribers(self, section: str, data: Dict):
        """Notify all subscribers of a state change."""
        for callback in self._subscribers:
            try:
                callback(section, data)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    def subscribe(self, callback: Callable[[str, Dict], None]):
        """Subscribe to state changes."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, Dict], None]):
        """Unsubscribe from state changes."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    # =========================================================================
    # GENERIC READ/WRITE
    # =========================================================================

    def get(self, section: str, default: Any = None) -> Any:
        """Get a state section."""
        with self._lock:
            return self._state.get(section, default)

    def set(self, section: str, value: Any):
        """Set a state section."""
        with self._lock:
            self._state[section] = value
            self._state[f"{section}_updated_at"] = datetime.now().isoformat()
            self._dirty = True
        self._notify_subscribers(section, value)

    def update(self, section: str, **kwargs):
        """Update fields in a state section."""
        with self._lock:
            if section not in self._state:
                self._state[section] = {}
            self._state[section].update(kwargs)
            self._state[section]["updated_at"] = datetime.now().isoformat()
            self._dirty = True
        self._notify_subscribers(section, self._state[section])

    def get_all(self) -> Dict[str, Any]:
        """Get the entire state."""
        with self._lock:
            return {
                "state": dict(self._state),
                "events": list(self._events)[-50:],  # Last 50 events
                "timestamp": datetime.now().isoformat(),
            }

    # =========================================================================
    # TYPED ACCESSORS - TRAINING
    # =========================================================================

    def update_training(
        self,
        status: Optional[str] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        file: Optional[str] = None,
        speed: Optional[float] = None,
        eta_seconds: Optional[int] = None,
        strain: Optional[float] = None,
        **extra
    ):
        """Update training state."""
        updates = {}
        if status is not None: updates["status"] = status
        if step is not None: updates["step"] = step
        if total_steps is not None: updates["total_steps"] = total_steps
        if loss is not None: updates["loss"] = loss
        if learning_rate is not None: updates["learning_rate"] = learning_rate
        if file is not None: updates["file"] = file
        if speed is not None: updates["speed"] = speed
        if eta_seconds is not None: updates["eta_seconds"] = eta_seconds
        if strain is not None: updates["strain"] = strain
        updates.update(extra)

        self.update("training", **updates)

    def get_training(self) -> Dict[str, Any]:
        """Get training state."""
        return self.get("training", {})

    # =========================================================================
    # TYPED ACCESSORS - QUEUE
    # =========================================================================

    def update_queue(
        self,
        depth: Optional[int] = None,
        high_priority: Optional[int] = None,
        normal_priority: Optional[int] = None,
        low_priority: Optional[int] = None,
        status: Optional[str] = None,
    ):
        """Update queue state."""
        updates = {}
        if depth is not None: updates["depth"] = depth
        if high_priority is not None: updates["high_priority"] = high_priority
        if normal_priority is not None: updates["normal_priority"] = normal_priority
        if low_priority is not None: updates["low_priority"] = low_priority
        if status is not None: updates["status"] = status

        self.update("queue", **updates)

    def get_queue(self) -> Dict[str, Any]:
        """Get queue state."""
        return self.get("queue", {})

    # =========================================================================
    # TYPED ACCESSORS - WORKERS
    # =========================================================================

    def update_worker(
        self,
        worker_id: str,
        role: Optional[str] = None,
        status: Optional[str] = None,
        device: Optional[str] = None,
        current_job: Optional[str] = None,
        **extra
    ):
        """Update a worker's state."""
        with self._lock:
            if "workers" not in self._state:
                self._state["workers"] = {}

            if worker_id not in self._state["workers"]:
                self._state["workers"][worker_id] = {"worker_id": worker_id}

            worker = self._state["workers"][worker_id]
            if role is not None: worker["role"] = role
            if status is not None: worker["status"] = status
            if device is not None: worker["device"] = device
            if current_job is not None: worker["current_job"] = current_job
            worker["last_heartbeat"] = datetime.now().isoformat()
            worker.update(extra)

            self._dirty = True

        self._notify_subscribers("workers", self._state["workers"])

    def get_workers(self) -> Dict[str, Dict]:
        """Get all workers' state."""
        return self.get("workers", {})

    def get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get a specific worker's state."""
        workers = self.get_workers()
        return workers.get(worker_id)

    # =========================================================================
    # TYPED ACCESSORS - HERO
    # =========================================================================

    def update_hero(
        self,
        name: Optional[str] = None,
        title: Optional[str] = None,
        level: Optional[int] = None,
        xp: Optional[int] = None,
        campaign_id: Optional[str] = None,
        current_skill: Optional[str] = None,
        current_skill_level: Optional[int] = None,
        **extra
    ):
        """Update hero state."""
        updates = {}
        if name is not None: updates["name"] = name
        if title is not None: updates["title"] = title
        if level is not None: updates["level"] = level
        if xp is not None: updates["xp"] = xp
        if campaign_id is not None: updates["campaign_id"] = campaign_id
        if current_skill is not None: updates["current_skill"] = current_skill
        if current_skill_level is not None: updates["current_skill_level"] = current_skill_level
        updates.update(extra)

        self.update("hero", **updates)

    def get_hero(self) -> Dict[str, Any]:
        """Get hero state."""
        return self.get("hero", {})

    # =========================================================================
    # EVENTS
    # =========================================================================

    def emit_event(
        self,
        kind: str,
        message: str,
        channel: str = "system",
        severity: str = "info",
        details: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Emit an event to the battle log.

        Args:
            kind: Event kind (training_started, checkpoint_saved, etc.)
            message: Human-readable message
            channel: Channel (system, training, eval, jobs, vault, guild)
            severity: Severity (info, success, warning, error)
            details: Additional details dict

        Returns:
            The event dict
        """
        event = {
            "id": f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{kind}",
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "channel": channel,
            "severity": severity,
            "message": message,
            "icon": self.CHANNEL_ICONS.get(channel, "📢"),
            "details": details or {},
        }

        with self._lock:
            self._events.append(event)
            self._dirty = True

        self._notify_subscribers("events", event)
        return event

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events, newest first."""
        with self._lock:
            events = list(self._events)
        return list(reversed(events[-limit:]))

    # =========================================================================
    # MODE
    # =========================================================================

    def set_mode(self, mode: str, reason: str = ""):
        """Set realm mode (training, idle)."""
        with self._lock:
            old_mode = self._state.get("mode", "idle")
            self._state["mode"] = mode
            self._state["mode_changed_at"] = datetime.now().isoformat()
            self._state["mode_reason"] = reason
            self._dirty = True

        if old_mode != mode:
            self.emit_event(
                "mode_changed",
                f"Mode changed: {old_mode} → {mode}" + (f" ({reason})" if reason else ""),
                channel="system",
                severity="info",
                details={"from": old_mode, "to": mode, "reason": reason},
            )

    def get_mode(self) -> str:
        """Get current realm mode."""
        return self.get("mode", "idle")

    # =========================================================================
    # FORCE FLUSH
    # =========================================================================

    def flush(self):
        """Force flush state to disk."""
        with self._lock:
            self._save()


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_store: Optional[RealmStateStore] = None
_store_lock = threading.Lock()


def get_store() -> RealmStateStore:
    """Get the singleton RealmStateStore instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = RealmStateStore()
    return _store


# Convenience functions that use the singleton
def update_training(**kwargs):
    """Update training state."""
    get_store().update_training(**kwargs)


def get_training_state() -> Dict[str, Any]:
    """Get training state."""
    return get_store().get_training()


def update_queue(**kwargs):
    """Update queue state."""
    get_store().update_queue(**kwargs)


def get_queue_state() -> Dict[str, Any]:
    """Get queue state."""
    return get_store().get_queue()


def update_worker(worker_id: str, **kwargs):
    """Update worker state."""
    get_store().update_worker(worker_id, **kwargs)


def get_workers_state() -> Dict[str, Dict]:
    """Get all workers' state."""
    return get_store().get_workers()


def update_hero(**kwargs):
    """Update hero state."""
    get_store().update_hero(**kwargs)


def get_hero_state() -> Dict[str, Any]:
    """Get hero state."""
    return get_store().get_hero()


def emit_event(kind: str, message: str, **kwargs) -> Dict:
    """Emit an event."""
    return get_store().emit_event(kind, message, **kwargs)


def get_events(limit: int = 50) -> List[Dict]:
    """Get recent events."""
    return get_store().get_events(limit)


def set_mode(mode: str, reason: str = ""):
    """Set realm mode."""
    get_store().set_mode(mode, reason)


def get_mode() -> str:
    """Get realm mode."""
    return get_store().get_mode()


def get_realm_state() -> Dict[str, Any]:
    """Get complete realm state."""
    return get_store().get_all()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Realm State Store")
    parser.add_argument("--show", action="store_true", help="Show current state")
    parser.add_argument("--events", action="store_true", help="Show recent events")
    parser.add_argument("--test", action="store_true", help="Run test updates")

    args = parser.parse_args()

    store = get_store()

    if args.test:
        print("Running test updates...")
        store.update_training(status="training", step=100, loss=0.5)
        store.emit_event("test", "Test event from CLI", channel="debug")
        store.flush()
        print("Done. Check status/realm_store.json")

    elif args.events:
        events = store.get_events(20)
        print(f"Recent Events ({len(events)}):")
        for e in events:
            print(f"  [{e['timestamp'][:19]}] {e['icon']} {e['channel']}: {e['message']}")

    else:
        state = store.get_all()
        print(json.dumps(state, indent=2, default=str))
