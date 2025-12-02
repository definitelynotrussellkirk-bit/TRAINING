"""
Realm State Store - Single Source of Truth for all Realm state.

NOW NETWORK-BASED: This module now uses the RealmState HTTP service instead
of direct file writes. The API remains the same, but now all operations
go through the network service.

Architecture:
    PRODUCERS (write) -> RealmStateStore (HTTP client) -> RealmService (SQLite)
    CONSUMERS (read) -> RealmStateStore (HTTP client) -> RealmService (SQLite)

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

Design Principles:
    1. Network-based (HTTP to RealmService on port 8866)
    2. Thread-safe (HTTP client handles concurrency)
    3. Typed sections (training, jobs, workers, hero, events)
    4. Same API as before (drop-in replacement)
    5. Graceful fallback to file-based if service unavailable
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import RealmClient
try:
    from realm.client import RealmClient
    REALM_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("realm.client not available - using file-based fallback")
    REALM_CLIENT_AVAILABLE = False


# =============================================================================
# REALM STATE STORE (Network-based wrapper)
# =============================================================================

class RealmStateStore:
    """
    Single source of truth for all Realm state.

    Now uses HTTP client to talk to RealmService instead of direct file writes.
    Provides same API as before for backward compatibility.
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
        "data": "📦",
        "checkpoint": "💾",
    }

    def __init__(self, service_url: Optional[str] = None):
        if service_url is None:
            # Get RealmState service URL from hosts.json
            try:
                from core.hosts import get_service_url
                service_url = get_service_url("realm_state")
                if not service_url:
                    # Fallback to localhost if not in hosts.json
                    service_url = "http://localhost:8866"
            except:
                # Fallback if hosts.py not available
                service_url = "http://localhost:8866"

        self.service_url = service_url
        self._client = None

        if REALM_CLIENT_AVAILABLE:
            self._client = RealmClient(service_url)
            # Test connection
            if not self._client.health():
                logger.warning(f"RealmState service not available at {service_url}, using fallback")
                self._client = None

        if self._client is None:
            logger.info("Using file-based fallback for realm state")
            self._init_fallback()

    def _init_fallback(self):
        """Initialize file-based fallback."""
        try:
            from core.paths import get_base_dir
            self.fallback_path = get_base_dir() / "status" / "realm_store.json"
        except ImportError:
            self.fallback_path = Path(__file__).parent.parent / "status" / "realm_store.json"

        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_state = {}
        self._fallback_events = []

        # Load existing fallback data
        if self.fallback_path.exists():
            try:
                with open(self.fallback_path, "r") as f:
                    data = json.load(f)
                    self._fallback_state = data.get("state", {})
                    self._fallback_events = data.get("events", [])
            except Exception as e:
                logger.error(f"Failed to load fallback state: {e}")

    def _save_fallback(self):
        """Save fallback state to disk."""
        if not hasattr(self, "fallback_path"):
            return
        try:
            with open(self.fallback_path, "w") as f:
                json.dump({
                    "state": self._fallback_state,
                    "events": self._fallback_events[-100:],  # Keep last 100
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save fallback state: {e}")

    # =========================================================================
    # GENERIC READ/WRITE
    # =========================================================================

    def get(self, section: str, default: Any = None) -> Any:
        """Get a state section."""
        if self._client:
            data = self._client.get_section(section)
            return data if data is not None else default
        else:
            return self._fallback_state.get(section, default)

    def set(self, section: str, value: Any):
        """Set a state section."""
        if self._client:
            self._client.update(section, **value)
        else:
            self._fallback_state[section] = value
            self._fallback_state[f"{section}_updated_at"] = datetime.now().isoformat()
            self._save_fallback()

    def update(self, section: str, **kwargs):
        """Update fields in a state section."""
        if self._client:
            self._client.update(section, **kwargs)
        else:
            if section not in self._fallback_state:
                self._fallback_state[section] = {}
            self._fallback_state[section].update(kwargs)
            self._fallback_state[section]["updated_at"] = datetime.now().isoformat()
            self._save_fallback()

    def get_all(self) -> Dict[str, Any]:
        """Get the entire state."""
        if self._client:
            data = self._client.get_state()
            return data if data else {"state": {}, "events": [], "timestamp": datetime.now().isoformat()}
        else:
            return {
                "state": dict(self._fallback_state),
                "events": self._fallback_events[-50:],
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
        if self._client:
            self._client.update_training(
                status=status, step=step, total_steps=total_steps,
                loss=loss, learning_rate=learning_rate, file=file,
                speed=speed, eta_seconds=eta_seconds, strain=strain,
                **extra
            )
        else:
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
        if self._client:
            self._client.update_queue(
                depth=depth, high_priority=high_priority,
                normal_priority=normal_priority, low_priority=low_priority,
                status=status
            )
        else:
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
        if self._client:
            self._client.update_worker(
                worker_id=worker_id, role=role, status=status,
                device=device, current_job=current_job, **extra
            )
        else:
            if "workers" not in self._fallback_state:
                self._fallback_state["workers"] = {}

            if worker_id not in self._fallback_state["workers"]:
                self._fallback_state["workers"][worker_id] = {"worker_id": worker_id}

            worker = self._fallback_state["workers"][worker_id]
            if role is not None: worker["role"] = role
            if status is not None: worker["status"] = status
            if device is not None: worker["device"] = device
            if current_job is not None: worker["current_job"] = current_job
            worker["last_heartbeat"] = datetime.now().isoformat()
            worker.update(extra)
            self._save_fallback()

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
        if self._client:
            self._client.update_hero(
                name=name, title=title, level=level, xp=xp,
                campaign_id=campaign_id, current_skill=current_skill,
                current_skill_level=current_skill_level, **extra
            )
        else:
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
    # TYPED ACCESSORS - SKILLS
    # =========================================================================

    def update_skills(
        self,
        skills: Optional[Dict[str, Dict]] = None,
        active_skill: Optional[str] = None,
        **extra
    ):
        """Update skills/curriculum state."""
        if self._client:
            self._client.update_skills(skills=skills, active_skill=active_skill, **extra)
        else:
            updates = {}
            if skills is not None: updates["skills"] = skills
            if active_skill is not None: updates["active_skill"] = active_skill
            updates.update(extra)
            self.update("skills", **updates)

    def get_skills(self) -> Dict[str, Any]:
        """Get skills/curriculum state."""
        return self.get("skills", {})

    def update_skill(
        self,
        skill_id: str,
        mastered_level: Optional[int] = None,
        training_level: Optional[int] = None,
        accuracy: Optional[float] = None,
        **extra
    ):
        """Update a single skill's state."""
        current = self.get_skills()
        skills = current.get("skills", {})

        if skill_id not in skills:
            skills[skill_id] = {"id": skill_id}

        skill = skills[skill_id]
        if mastered_level is not None: skill["mastered_level"] = mastered_level
        if training_level is not None: skill["training_level"] = training_level
        if accuracy is not None: skill["last_accuracy"] = accuracy
        skill.update(extra)

        self.update_skills(skills=skills)

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
        """Emit an event to the battle log."""
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

        if self._client:
            self._client.emit_event(kind, message, channel, severity, details)
        else:
            self._fallback_events.append(event)
            if len(self._fallback_events) > 100:
                self._fallback_events = self._fallback_events[-100:]
            self._save_fallback()

        return event

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events, newest first."""
        if self._client:
            events = self._client.get_events(limit)
            return list(reversed(events))  # Newest first
        else:
            return list(reversed(self._fallback_events[-limit:]))

    # =========================================================================
    # MODE
    # =========================================================================

    def set_mode(self, mode: str, reason: str = ""):
        """Set realm mode (training, idle)."""
        if self._client:
            self._client.set_mode(mode, reason)
        else:
            old_mode = self._fallback_state.get("mode", "idle")
            self._fallback_state["mode"] = mode
            self._fallback_state["mode_changed_at"] = datetime.now().isoformat()
            self._fallback_state["mode_reason"] = reason
            self._save_fallback()

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
        if self._client:
            return self._client.get_mode()
        else:
            return self._fallback_state.get("mode", "idle")

    # =========================================================================
    # COMPATIBILITY METHODS
    # =========================================================================

    def flush(self):
        """Flush state (no-op for network client, saves for fallback)."""
        if not self._client:
            self._save_fallback()


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_store: Optional[RealmStateStore] = None


def get_store() -> RealmStateStore:
    """Get the singleton RealmStateStore instance."""
    global _store
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


def update_skills(**kwargs):
    """Update skills/curriculum state."""
    get_store().update_skills(**kwargs)


def get_skills_state() -> Dict[str, Any]:
    """Get skills/curriculum state."""
    return get_store().get_skills()


def update_skill(skill_id: str, **kwargs):
    """Update a single skill's state."""
    get_store().update_skill(skill_id, **kwargs)


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
    parser.add_argument("--service-url", help="RealmState service URL")

    args = parser.parse_args()

    if args.service_url:
        store = RealmStateStore(args.service_url)
    else:
        store = get_store()

    if args.test:
        print("Running test updates...")
        store.update_training(status="training", step=100, loss=0.5)
        store.emit_event("test", "Test event from CLI", channel="debug")
        store.flush()
        print("Done.")

    elif args.events:
        events = store.get_events(20)
        print(f"Recent Events ({len(events)}):")
        for e in events:
            print(f"  [{e['timestamp'][:19]}] {e['icon']} {e['channel']}: {e['message']}")

    else:
        state = store.get_all()
        print(json.dumps(state, indent=2, default=str))
