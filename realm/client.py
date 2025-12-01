"""
Realm State Client - HTTP client for producers and consumers.

This replaces direct file writes to realm_store.json with HTTP calls
to the RealmState service.

Usage (Producer):
    from realm.client import RealmClient
    client = RealmClient("http://localhost:8866")
    client.update("training", status="training", step=100, loss=0.5)
    client.emit_event("training_started", "Training started", channel="training")

Usage (Consumer):
    from realm.client import RealmClient
    client = RealmClient("http://localhost:8866")
    state = client.get_state()
    training = client.get_section("training")
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class RealmClient:
    """HTTP client for RealmState service."""

    def __init__(self, base_url: str = "http://localhost:8866"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request."""
        url = f"{self.base_url}{path}"

        try:
            if method == "GET":
                req = urllib.request.Request(url, method="GET")
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(req, timeout=5) as response:
                    return json.loads(response.read().decode())

            elif method == "POST":
                req = urllib.request.Request(url, method="POST")
                req.add_header("Content-Type", "application/json")
                body = json.dumps(data or {}, default=str).encode()
                with urllib.request.urlopen(req, data=body, timeout=5) as response:
                    return json.loads(response.read().decode())

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP {e.code} for {method} {path}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            logger.error(f"Connection error for {method} {path}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"Request error for {method} {path}: {e}")
            return None

    def health(self) -> bool:
        """Check if service is healthy."""
        result = self._request("GET", "/health")
        return result is not None and result.get("status") == "ok"

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get complete realm state."""
        return self._request("GET", "/api/state")

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Get specific state section."""
        return self._request("GET", f"/api/state/{section}")

    def update(self, section: str, **kwargs):
        """Update a state section."""
        return self._request("POST", f"/api/update/{section}", kwargs)

    def emit_event(
        self,
        kind: str,
        message: str,
        channel: str = "system",
        severity: str = "info",
        details: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Emit an event."""
        event = {
            "kind": kind,
            "message": message,
            "channel": channel,
            "severity": severity,
            "details": details or {},
        }
        return self._request("POST", "/api/event", event)

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        result = self._request("GET", f"/api/events?limit={limit}")
        return result.get("events", []) if result else []

    # Convenience methods matching core/realm_store.py API

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

    def get_training(self) -> Optional[Dict[str, Any]]:
        """Get training state."""
        return self.get_section("training")

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

    def get_queue(self) -> Optional[Dict[str, Any]]:
        """Get queue state."""
        return self.get_section("queue")

    def update_worker(
        self,
        worker_id: str,
        role: Optional[str] = None,
        status: Optional[str] = None,
        device: Optional[str] = None,
        current_job: Optional[str] = None,
        **extra
    ):
        """Update worker state."""
        # Get existing worker state
        workers = self.get_section("workers") or {}
        worker = workers.get(worker_id, {"worker_id": worker_id})

        # Update fields
        if role is not None: worker["role"] = role
        if status is not None: worker["status"] = status
        if device is not None: worker["device"] = device
        if current_job is not None: worker["current_job"] = current_job
        worker["last_heartbeat"] = datetime.now().isoformat()
        worker.update(extra)

        # Update entire workers section
        workers[worker_id] = worker
        self.update("workers", **workers)

    def get_workers(self) -> Dict[str, Dict]:
        """Get all workers' state."""
        return self.get_section("workers") or {}

    def get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get specific worker state."""
        workers = self.get_workers()
        return workers.get(worker_id)

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

    def get_hero(self) -> Optional[Dict[str, Any]]:
        """Get hero state."""
        return self.get_section("hero")

    def set_mode(self, mode: str, reason: str = ""):
        """Set realm mode."""
        # Get current mode
        state = self.get_section("mode_info") or {}
        old_mode = state.get("mode", "idle")

        # Update mode
        self.update("mode_info", mode=mode, mode_reason=reason, mode_changed_at=datetime.now().isoformat())

        # Emit event if changed
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
        state = self.get_section("mode_info") or {}
        return state.get("mode", "idle")


# Singleton pattern for convenience
_client: Optional[RealmClient] = None


def get_client(base_url: str = "http://localhost:8866") -> RealmClient:
    """Get or create singleton RealmClient."""
    global _client
    if _client is None:
        _client = RealmClient(base_url)
    return _client


if __name__ == "__main__":
    # Test client
    import sys

    client = RealmClient()

    if "--test" in sys.argv:
        print("Testing RealmClient...")
        print(f"Health check: {client.health()}")
        print(f"Getting state...")
        state = client.get_state()
        if state:
            print(json.dumps(state, indent=2, default=str))
        else:
            print("Failed to get state")
    else:
        print("Usage: python3 -m realm.client --test")
