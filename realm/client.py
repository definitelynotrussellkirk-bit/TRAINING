"""
Realm State Client - HTTP client for producers and consumers.

Features:
- Connection pooling via requests.Session
- Debounced updates to reduce server load
- Atomic worker updates via dedicated endpoint
- History recording support
- SSE subscription support

Usage (Producer):
    from realm.client import RealmClient
    client = RealmClient("http://localhost:8866")
    client.update_training(status="training", step=100, loss=0.5)
    client.emit_event("training_started", "Training started", channel="training")

Usage (Consumer):
    from realm.client import RealmClient
    client = RealmClient("http://localhost:8866")
    state = client.get_state()
    training = client.get_section("training")

Usage (SSE):
    client = RealmClient("http://localhost:8866")
    for event in client.stream():
        print(event)
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Try to use requests for connection pooling
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.error
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - using urllib (no connection pooling)")


class DebouncedUpdater:
    """Batches rapid updates into a single request."""

    def __init__(self, flush_interval: float = 0.5):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._flush_callback: Optional[Callable] = None
        self._timer: Optional[threading.Timer] = None

    def set_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set the callback to invoke when flushing."""
        self._flush_callback = callback

    def add(self, section: str, data: Dict[str, Any]):
        """Add data to be flushed."""
        with self._lock:
            if section not in self._pending:
                self._pending[section] = {}
            self._pending[section].update(data)

            # Schedule flush if not already scheduled
            if self._timer is None:
                self._timer = threading.Timer(self._flush_interval, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self):
        """Flush all pending updates."""
        with self._lock:
            pending = self._pending
            self._pending = {}
            self._timer = None

        if self._flush_callback:
            for section, data in pending.items():
                try:
                    self._flush_callback(section, data)
                except Exception as e:
                    logger.error(f"Debounced flush error for {section}: {e}")

    def flush_now(self):
        """Force immediate flush."""
        if self._timer:
            self._timer.cancel()
        self._flush()


class RealmClient:
    """HTTP client for RealmState service with connection pooling."""

    def __init__(
        self,
        base_url: str = "http://localhost:8866",
        timeout: float = 5.0,
        debounce_interval: float = 0.0,  # 0 = disabled
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Connection pooling
        if REQUESTS_AVAILABLE:
            self._session = requests.Session()
            # Set retry strategy
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=10)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        else:
            self._session = None

        # Debouncing
        self._debouncer: Optional[DebouncedUpdater] = None
        if debounce_interval > 0:
            self._debouncer = DebouncedUpdater(debounce_interval)
            self._debouncer.set_callback(self._do_update)

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request with connection pooling."""
        url = f"{self.base_url}{path}"

        try:
            if REQUESTS_AVAILABLE:
                if method == "GET":
                    response = self._session.get(url, timeout=self.timeout)
                elif method == "POST":
                    response = self._session.post(url, json=data or {}, timeout=self.timeout)
                elif method == "DELETE":
                    response = self._session.delete(url, timeout=self.timeout)
                else:
                    raise ValueError(f"Unknown method: {method}")

                if response.status_code >= 400:
                    logger.error(f"HTTP {response.status_code} for {method} {path}: {response.text}")
                    return None
                return response.json()

            else:
                # Fallback to urllib
                if method == "GET":
                    req = urllib.request.Request(url, method="GET")
                    req.add_header("Content-Type", "application/json")
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        return json.loads(response.read().decode())

                elif method == "POST":
                    req = urllib.request.Request(url, method="POST")
                    req.add_header("Content-Type", "application/json")
                    body = json.dumps(data or {}, default=str).encode()
                    with urllib.request.urlopen(req, data=body, timeout=self.timeout) as response:
                        return json.loads(response.read().decode())

                elif method == "DELETE":
                    req = urllib.request.Request(url, method="DELETE")
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        return json.loads(response.read().decode())

        except Exception as e:
            logger.error(f"Request error for {method} {path}: {e}")
            return None

    def close(self):
        """Close the client and flush pending updates."""
        if self._debouncer:
            self._debouncer.flush_now()
        if self._session and REQUESTS_AVAILABLE:
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # HEALTH & METRICS
    # =========================================================================

    def health(self) -> bool:
        """Check if service is healthy."""
        result = self._request("GET", "/health")
        return result is not None and result.get("status") == "ok"

    def metrics(self) -> Optional[Dict[str, Any]]:
        """Get service metrics."""
        return self._request("GET", "/metrics")

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get complete realm state."""
        return self._request("GET", "/api/state")

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Get specific state section."""
        return self._request("GET", f"/api/state/{section}")

    def update(self, section: str, record_history: bool = False, **kwargs):
        """Update a state section."""
        if record_history:
            kwargs["_record_history"] = True

        if self._debouncer and section in ("training",):
            # Debounce training updates
            self._debouncer.add(section, kwargs)
        else:
            self._do_update(section, kwargs)

    def _do_update(self, section: str, data: Dict[str, Any]):
        """Actually send the update."""
        return self._request("POST", f"/api/update/{section}", data)

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

    def get_events(
        self,
        limit: int = 50,
        channel: Optional[str] = None,
        kind: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get events with optional filtering."""
        params = [f"limit={limit}"]
        if channel:
            params.append(f"channel={channel}")
        if kind:
            params.append(f"kind={kind}")
        if since:
            params.append(f"since={since}")

        query = "&".join(params)
        result = self._request("GET", f"/api/events?{query}")
        return result.get("events", []) if result else []

    # =========================================================================
    # HISTORY
    # =========================================================================

    def get_history(
        self,
        section: str,
        limit: int = 100,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical snapshots for a section."""
        params = [f"limit={limit}"]
        if since:
            params.append(f"since={since}")

        query = "&".join(params)
        result = self._request("GET", f"/api/history/{section}?{query}")
        return result.get("history", []) if result else []

    # =========================================================================
    # SSE STREAMING
    # =========================================================================

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """Subscribe to SSE stream. Yields events as they arrive."""
        url = f"{self.base_url}/api/stream"

        if REQUESTS_AVAILABLE:
            try:
                with self._session.get(url, stream=True, timeout=None) as response:
                    for line in response.iter_lines(decode_unicode=True):
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                pass
                        elif line.startswith("event:"):
                            # Event type - could be used for routing
                            pass
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                raise
        else:
            # urllib doesn't support streaming well
            raise NotImplementedError("SSE streaming requires the 'requests' package")

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
        record_history: bool = False,
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
        self.update("training", record_history=record_history, **updates)

    def get_training(self) -> Optional[Dict[str, Any]]:
        """Get training state."""
        return self.get_section("training")

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

    def get_queue(self) -> Optional[Dict[str, Any]]:
        """Get queue state."""
        return self.get_section("queue")

    # =========================================================================
    # TYPED ACCESSORS - WORKERS (atomic updates)
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
        """Atomically update a single worker via dedicated endpoint."""
        updates = {}
        if role is not None: updates["role"] = role
        if status is not None: updates["status"] = status
        if device is not None: updates["device"] = device
        if current_job is not None: updates["current_job"] = current_job
        updates.update(extra)
        # Use dedicated atomic endpoint
        return self._request("POST", f"/api/worker/{worker_id}", updates)

    def get_workers(self) -> Dict[str, Dict]:
        """Get all workers' state."""
        result = self._request("GET", "/api/workers")
        return result.get("workers", {}) if result else {}

    def get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get specific worker state."""
        return self._request("GET", f"/api/workers/{worker_id}")

    def remove_worker(self, worker_id: str):
        """Remove a worker."""
        return self._request("DELETE", f"/api/worker/{worker_id}")

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

    def get_hero(self) -> Optional[Dict[str, Any]]:
        """Get hero state."""
        return self.get_section("hero")

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
        updates = {}
        if skills is not None: updates["skills"] = skills
        if active_skill is not None: updates["active_skill"] = active_skill
        updates.update(extra)
        self.update("skills", **updates)

    def get_skills(self) -> Optional[Dict[str, Any]]:
        """Get skills/curriculum state."""
        return self.get_section("skills")

    # =========================================================================
    # MODE
    # =========================================================================

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


# =============================================================================
# SINGLETON
# =============================================================================

_client: Optional[RealmClient] = None


def get_client(base_url: str = "http://localhost:8866") -> RealmClient:
    """Get or create singleton RealmClient."""
    global _client
    if _client is None:
        _client = RealmClient(base_url)
    return _client


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
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

    elif "--metrics" in sys.argv:
        print("Service metrics:")
        metrics = client.metrics()
        if metrics:
            print(json.dumps(metrics, indent=2))
        else:
            print("Failed to get metrics")

    elif "--stream" in sys.argv:
        print("Streaming events (Ctrl+C to stop)...")
        try:
            for event in client.stream():
                print(json.dumps(event, indent=2))
        except KeyboardInterrupt:
            print("\nStopped.")

    else:
        print("Usage: python3 -m realm.client [--test|--metrics|--stream]")
