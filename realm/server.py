"""
Realm State Service - HTTP server backed by SQLite with SSE support.

Single source of truth for all Realm state, accessible over the network.

Features:
- SQLite backend with state sections, events, and history
- Server-Sent Events (SSE) for real-time push updates
- Atomic worker updates
- Event channel filtering
- Staleness detection and cleanup
- Metrics endpoint
- Schema versioning

Usage:
    python3 -m realm.server --port 8866
    python3 -m realm.server --port 8866 --host 0.0.0.0  # Listen on all interfaces
"""

import argparse
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Schema version - increment when breaking changes are made
SCHEMA_VERSION = 2

# Staleness thresholds
WORKER_STALE_SECONDS = 60  # Workers stale after 60s without heartbeat
TRAINING_STALE_SECONDS = 30  # Training state stale after 30s


class SSEManager:
    """Manages Server-Sent Events connections."""

    def __init__(self):
        self._clients: Dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._client_id = 0

    def add_client(self) -> tuple[int, queue.Queue]:
        """Add a new SSE client, returns (client_id, message_queue)."""
        with self._lock:
            self._client_id += 1
            q = queue.Queue(maxsize=100)
            self._clients[self._client_id] = q
            logger.info(f"SSE client {self._client_id} connected (total: {len(self._clients)})")
            return self._client_id, q

    def remove_client(self, client_id: int):
        """Remove an SSE client."""
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"SSE client {client_id} disconnected (total: {len(self._clients)})")

    def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to all connected clients."""
        message = f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
        with self._lock:
            dead_clients = []
            for client_id, q in self._clients.items():
                try:
                    q.put_nowait(message)
                except queue.Full:
                    dead_clients.append(client_id)
                    logger.warning(f"SSE client {client_id} queue full, dropping")

            # Clean up dead clients
            for client_id in dead_clients:
                del self._clients[client_id]

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


class RealmStateDB:
    """SQLite backend for realm state with history and metrics."""

    def __init__(self, db_path: Path, sse_manager: SSEManager):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sse = sse_manager

        # Metrics
        self._metrics = {
            "requests_total": 0,
            "updates_total": 0,
            "events_total": 0,
            "errors_total": 0,
            "sse_broadcasts": 0,
            "started_at": datetime.now().isoformat(),
        }

        self._init_db()

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path), timeout=10)

    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # State table - key/value store for state sections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    section TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Events table - battle log / event stream
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    icon TEXT,
                    details TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel)")

            # History table - state snapshots for time-series
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    section TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_section_time ON history(section, timestamp DESC)")

            # Workers table - dedicated table for efficient worker updates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    role TEXT,
                    status TEXT,
                    device TEXT,
                    current_job TEXT,
                    last_heartbeat TEXT NOT NULL,
                    extra TEXT
                )
            """)

            conn.commit()
            conn.close()

    def _cleanup_loop(self):
        """Background thread for staleness detection and cleanup."""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._mark_stale_workers()
                self._check_training_staleness()
                self._prune_old_events()
                self._prune_old_history()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is alive."""
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return True
        except (OSError, TypeError, ProcessLookupError):
            return False

    def _check_training_staleness(self):
        """Check if training status says 'training' but process is dead, and reset if so."""
        try:
            # Get base dir
            try:
                from core.paths import get_base_dir
                base_dir = get_base_dir()
            except ImportError:
                base_dir = Path(__file__).parent.parent

            # Check PID file
            pid_file = base_dir / ".pids" / "hero_loop.pid"
            control_state_file = base_dir / "control" / "state.json"
            training_status_file = base_dir / "status" / "training_status.json"

            # Read current state
            is_training_state = False
            if control_state_file.exists():
                try:
                    with open(control_state_file, 'r') as f:
                        control_state = json.load(f)
                    is_training_state = control_state.get("mode") == "training"
                except Exception:
                    pass

            is_training_status = False
            if training_status_file.exists():
                try:
                    with open(training_status_file, 'r') as f:
                        training_status = json.load(f)
                    is_training_status = training_status.get("status") == "training"
                except Exception:
                    pass

            # If neither says training, nothing to check
            if not is_training_state and not is_training_status:
                return

            # Check if process is alive
            process_alive = False
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    process_alive = self._is_process_alive(pid)
                except Exception:
                    pass

            # If status says training but process is dead, reset state
            if (is_training_state or is_training_status) and not process_alive:
                logger.warning("Training state stale: status='training' but hero_loop process is dead. Resetting to idle.")

                # Reset control/state.json
                if control_state_file.exists():
                    try:
                        with open(control_state_file, 'r') as f:
                            state = json.load(f)
                        state["mode"] = "idle"
                        state["started_at"] = None
                        with open(control_state_file, 'w') as f:
                            json.dump(state, f, indent=2)
                        logger.info("  Reset control/state.json to idle")
                    except Exception as e:
                        logger.error(f"  Failed to reset control/state.json: {e}")

                # Reset status/training_status.json
                if training_status_file.exists():
                    try:
                        with open(training_status_file, 'r') as f:
                            status = json.load(f)
                        status["status"] = "idle"
                        status["timestamp"] = datetime.now().isoformat()
                        with open(training_status_file, 'w') as f:
                            json.dump(status, f, indent=2)
                        logger.info("  Reset status/training_status.json to idle")
                    except Exception as e:
                        logger.error(f"  Failed to reset status/training_status.json: {e}")

                # Clean up stale PID file
                if pid_file.exists():
                    try:
                        pid_file.unlink()
                        logger.info("  Removed stale PID file")
                    except Exception:
                        pass

                # Broadcast the state change
                self._sse.broadcast("training", {"status": "idle", "stale_reset": True})

        except Exception as e:
            logger.error(f"Training staleness check error: {e}")

    def _mark_stale_workers(self):
        """Mark workers as stale if no recent heartbeat."""
        threshold = (datetime.now() - timedelta(seconds=WORKER_STALE_SECONDS)).isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workers
                SET status = 'stale'
                WHERE last_heartbeat < ? AND status != 'stale'
            """, (threshold,))
            if cursor.rowcount > 0:
                logger.info(f"Marked {cursor.rowcount} workers as stale")
                conn.commit()
                # Broadcast worker state change
                self._sse.broadcast("workers", self.get_all_workers())
            conn.close()

    def _prune_old_events(self, keep_count: int = 1000):
        """Remove old events, keeping only the most recent."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM events WHERE id NOT IN (
                    SELECT id FROM events ORDER BY timestamp DESC LIMIT ?
                )
            """, (keep_count,))
            if cursor.rowcount > 0:
                logger.debug(f"Pruned {cursor.rowcount} old events")
            conn.commit()
            conn.close()

    def _prune_old_history(self, max_age_hours: int = 24):
        """Remove history older than max_age_hours."""
        threshold = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history WHERE timestamp < ?", (threshold,))
            if cursor.rowcount > 0:
                logger.debug(f"Pruned {cursor.rowcount} old history entries")
            conn.commit()
            conn.close()

    def update_section(self, section: str, data: Dict[str, Any], record_history: bool = False):
        """Update a state section."""
        self._metrics["updates_total"] += 1
        now = datetime.now().isoformat()

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Get existing data
            cursor.execute("SELECT data FROM state WHERE section = ?", (section,))
            row = cursor.fetchone()

            if row:
                existing = json.loads(row[0])
                existing.update(data)
                existing["updated_at"] = now
                merged = existing
            else:
                data["updated_at"] = now
                merged = data

            data_json = json.dumps(merged, default=str)

            # Upsert
            cursor.execute("""
                INSERT OR REPLACE INTO state (section, data, updated_at)
                VALUES (?, ?, ?)
            """, (section, data_json, now))

            # Record history snapshot if requested
            if record_history:
                cursor.execute("""
                    INSERT INTO history (timestamp, section, data)
                    VALUES (?, ?, ?)
                """, (now, section, data_json))

            conn.commit()
            conn.close()

        # Broadcast update via SSE
        self._sse.broadcast(section, merged)
        self._metrics["sse_broadcasts"] += 1

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Get a state section."""
        self._metrics["requests_total"] += 1
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM state WHERE section = ?", (section,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state sections."""
        self._metrics["requests_total"] += 1
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT section, data FROM state")
            rows = cursor.fetchall()
            conn.close()

            state = {}
            for section, data_json in rows:
                state[section] = json.loads(data_json)

            # Include workers from dedicated table
            state["workers"] = self.get_all_workers()

            return state

    # =========================================================================
    # WORKER-SPECIFIC OPERATIONS (atomic, efficient)
    # =========================================================================

    def update_worker(self, worker_id: str, **kwargs):
        """Atomically update a single worker."""
        self._metrics["updates_total"] += 1
        now = datetime.now().isoformat()

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Get existing worker
            cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
            row = cursor.fetchone()

            role = kwargs.get("role")
            status = kwargs.get("status")
            device = kwargs.get("device")
            current_job = kwargs.get("current_job")

            # Extract extra fields
            known_fields = {"role", "status", "device", "current_job", "worker_id"}
            extra = {k: v for k, v in kwargs.items() if k not in known_fields}

            if row:
                # Update existing
                cursor.execute("""
                    UPDATE workers SET
                        role = COALESCE(?, role),
                        status = COALESCE(?, status),
                        device = COALESCE(?, device),
                        current_job = COALESCE(?, current_job),
                        last_heartbeat = ?,
                        extra = ?
                    WHERE worker_id = ?
                """, (role, status, device, current_job, now,
                      json.dumps(extra) if extra else None, worker_id))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO workers (worker_id, role, status, device, current_job, last_heartbeat, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (worker_id, role, status or "starting", device, current_job, now,
                      json.dumps(extra) if extra else None))

            conn.commit()
            conn.close()

        # Broadcast worker update
        self._sse.broadcast("worker_update", {"worker_id": worker_id, **kwargs, "last_heartbeat": now})

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific worker's state."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_worker(row, cursor.description)
            return None

    def get_all_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers' state."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workers ORDER BY worker_id")
            rows = cursor.fetchall()
            desc = cursor.description
            conn.close()

            workers = {}
            for row in rows:
                worker = self._row_to_worker(row, desc)
                workers[worker["worker_id"]] = worker

            return workers

    def _row_to_worker(self, row, description) -> Dict[str, Any]:
        """Convert a database row to worker dict."""
        cols = [d[0] for d in description]
        worker = dict(zip(cols, row))

        # Parse extra JSON
        if worker.get("extra"):
            try:
                extra = json.loads(worker["extra"])
                worker.update(extra)
            except:
                pass
            del worker["extra"]

        return worker

    def remove_worker(self, worker_id: str):
        """Remove a worker."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
            conn.commit()
            conn.close()

        self._sse.broadcast("worker_removed", {"worker_id": worker_id})

    # =========================================================================
    # EVENTS
    # =========================================================================

    def add_event(self, event: Dict[str, Any]) -> bool:
        """Add an event to the log. Returns False if duplicate."""
        self._metrics["events_total"] += 1

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            # Check for duplicate
            cursor.execute("SELECT 1 FROM events WHERE id = ?", (event["id"],))
            if cursor.fetchone():
                conn.close()
                return False  # Duplicate

            cursor.execute("""
                INSERT INTO events (id, timestamp, kind, channel, severity, message, icon, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event["id"],
                event["timestamp"],
                event["kind"],
                event["channel"],
                event["severity"],
                event["message"],
                event.get("icon", "📢"),
                json.dumps(event.get("details", {}), default=str),
            ))

            conn.commit()
            conn.close()

        # Broadcast event via SSE
        self._sse.broadcast("event", event)
        self._metrics["sse_broadcasts"] += 1
        return True

    def get_events(
        self,
        limit: int = 50,
        channel: Optional[str] = None,
        kind: Optional[str] = None,
        since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events with optional filtering."""
        self._metrics["requests_total"] += 1

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            query = "SELECT id, timestamp, kind, channel, severity, message, icon, details FROM events WHERE 1=1"
            params = []

            if channel:
                query += " AND channel = ?"
                params.append(channel)
            if kind:
                query += " AND kind = ?"
                params.append(kind)
            if since:
                query += " AND timestamp > ?"
                params.append(since)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            events = []
            for row in rows:
                events.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "kind": row[2],
                    "channel": row[3],
                    "severity": row[4],
                    "message": row[5],
                    "icon": row[6] or "📢",
                    "details": json.loads(row[7]) if row[7] else {},
                })

            return list(reversed(events))  # Return oldest-first

    # =========================================================================
    # HISTORY
    # =========================================================================

    def get_history(
        self,
        section: str,
        limit: int = 100,
        since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get historical snapshots for a section."""
        self._metrics["requests_total"] += 1

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            if since:
                cursor.execute("""
                    SELECT timestamp, data FROM history
                    WHERE section = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (section, since, limit))
            else:
                cursor.execute("""
                    SELECT timestamp, data FROM history
                    WHERE section = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (section, limit))

            rows = cursor.fetchall()
            conn.close()

            return [
                {"timestamp": row[0], "data": json.loads(row[1])}
                for row in reversed(rows)
            ]

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        uptime = datetime.now() - datetime.fromisoformat(self._metrics["started_at"])

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM events")
            event_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM history")
            history_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM workers")
            worker_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM workers WHERE status = 'stale'")
            stale_workers = cursor.fetchone()[0]

            conn.close()

        return {
            **self._metrics,
            "uptime_seconds": int(uptime.total_seconds()),
            "sse_clients": self._sse.client_count,
            "event_count": event_count,
            "history_count": history_count,
            "worker_count": worker_count,
            "stale_workers": stale_workers,
            "schema_version": SCHEMA_VERSION,
        }

    def record_error(self):
        """Record an error for metrics."""
        self._metrics["errors_total"] += 1


class RealmStateHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RealmState service."""

    # Class variables - set by server
    db: RealmStateDB = None
    sse: SSEManager = None

    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Schema-Version", str(SCHEMA_VERSION))
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _send_error_json(self, message: str, status: int = 400):
        """Send error response."""
        self.db.record_error()
        self._send_json({"error": message, "schema_version": SCHEMA_VERSION}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            # Health check
            if path == "/health":
                self._send_json({
                    "status": "ok",
                    "service": "realm",
                    "schema_version": SCHEMA_VERSION,
                    "sse_clients": self.sse.client_count,
                })

            # Metrics
            elif path == "/metrics":
                self._send_json(self.db.get_metrics())

            # SSE stream
            elif path == "/api/stream":
                self._handle_sse()

            # Complete state
            elif path == "/api/state":
                state = self.db.get_all_state()
                events = self.db.get_events(50)
                self._send_json({
                    "state": state,
                    "events": events,
                    "timestamp": datetime.now().isoformat(),
                    "schema_version": SCHEMA_VERSION,
                })

            # Specific section
            elif path.startswith("/api/state/"):
                section = path.split("/")[-1]
                data = self.db.get_section(section)
                if data:
                    self._send_json(data)
                else:
                    self._send_error_json(f"Section not found: {section}", 404)

            # Workers
            elif path == "/api/workers":
                workers = self.db.get_all_workers()
                self._send_json({"workers": workers})

            elif path.startswith("/api/workers/"):
                worker_id = path.split("/")[-1]
                worker = self.db.get_worker(worker_id)
                if worker:
                    self._send_json(worker)
                else:
                    self._send_error_json(f"Worker not found: {worker_id}", 404)

            # Events with filtering
            elif path == "/api/events":
                limit = int(query.get("limit", ["50"])[0])
                channel = query.get("channel", [None])[0]
                kind = query.get("kind", [None])[0]
                since = query.get("since", [None])[0]
                events = self.db.get_events(limit, channel, kind, since)
                self._send_json({"events": events})

            # History
            elif path.startswith("/api/history/"):
                section = path.split("/")[-1]
                limit = int(query.get("limit", ["100"])[0])
                since = query.get("since", [None])[0]
                history = self.db.get_history(section, limit, since)
                self._send_json({"history": history, "section": section})

            else:
                self._send_error_json(f"Not found: {path}", 404)

        except Exception as e:
            logger.error(f"GET error: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            # Update section
            if path.startswith("/api/update/"):
                section = path.split("/")[-1]
                record_history = data.pop("_record_history", False)
                self.db.update_section(section, data, record_history=record_history)
                self._send_json({"success": True, "section": section})

            # Atomic worker update
            elif path.startswith("/api/worker/"):
                worker_id = path.split("/")[-1]
                self.db.update_worker(worker_id, **data)
                self._send_json({"success": True, "worker_id": worker_id})

            # Add event
            elif path == "/api/event":
                event = {
                    "id": data.get("id", f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{data.get('kind', 'event')}"),
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "kind": data.get("kind", "event"),
                    "channel": data.get("channel", "system"),
                    "severity": data.get("severity", "info"),
                    "message": data.get("message", ""),
                    "icon": data.get("icon", "📢"),
                    "details": data.get("details", {}),
                }
                added = self.db.add_event(event)
                self._send_json({"success": True, "event": event, "was_duplicate": not added})

            else:
                self._send_error_json(f"Not found: {path}", 404)

        except Exception as e:
            logger.error(f"POST error: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            # Remove worker
            if path.startswith("/api/worker/"):
                worker_id = path.split("/")[-1]
                self.db.remove_worker(worker_id)
                self._send_json({"success": True, "worker_id": worker_id})

            else:
                self._send_error_json(f"Not found: {path}", 404)

        except Exception as e:
            logger.error(f"DELETE error: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def _handle_sse(self):
        """Handle Server-Sent Events connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Schema-Version", str(SCHEMA_VERSION))
        self.end_headers()

        client_id, message_queue = self.sse.add_client()

        # Send initial state
        try:
            state = self.db.get_all_state()
            events = self.db.get_events(50)
            initial = f"event: init\ndata: {json.dumps({'state': state, 'events': events}, default=str)}\n\n"
            self.wfile.write(initial.encode())
            self.wfile.flush()
        except Exception as e:
            logger.error(f"SSE init error: {e}")
            self.sse.remove_client(client_id)
            return

        # Stream messages
        try:
            while True:
                try:
                    message = message_queue.get(timeout=30)
                    self.wfile.write(message.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(": keepalive\n\n".encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
        finally:
            self.sse.remove_client(client_id)

    def log_message(self, format, *args):
        """Override to use logger."""
        # Only log non-SSE requests at debug level
        if "/api/stream" not in args[0]:
            logger.debug(f"{self.address_string()} - {format % args}")


class ThreadedHTTPServer(HTTPServer):
    """HTTPServer with threading support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon_threads = True

    def process_request(self, request, client_address):
        """Handle each request in a new thread."""
        thread = threading.Thread(target=self.process_request_thread, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def process_request_thread(self, request, client_address):
        """Process request in thread."""
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class RealmStateServer:
    """Realm State HTTP server with SSE support."""

    def __init__(self, db_path: Path, host: str = "localhost", port: int = 8866):
        self.db_path = db_path
        self.host = host
        self.port = port
        self.sse = SSEManager()
        self.db = RealmStateDB(db_path, self.sse)
        self.server = None

    def start(self):
        """Start the server."""
        # Set class variables so handler can access DB and SSE
        RealmStateHandler.db = self.db
        RealmStateHandler.sse = self.sse

        self.server = ThreadedHTTPServer((self.host, self.port), RealmStateHandler)
        logger.info(f"RealmState service starting on {self.host}:{self.port}")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Schema version: {SCHEMA_VERSION}")
        logger.info(f"Endpoints:")
        logger.info(f"  GET  /health")
        logger.info(f"  GET  /metrics")
        logger.info(f"  GET  /api/stream (SSE)")
        logger.info(f"  GET  /api/state")
        logger.info(f"  GET  /api/state/<section>")
        logger.info(f"  GET  /api/workers")
        logger.info(f"  GET  /api/workers/<id>")
        logger.info(f"  GET  /api/events?channel=X&kind=Y&since=T&limit=N")
        logger.info(f"  GET  /api/history/<section>?since=T&limit=N")
        logger.info(f"  POST /api/update/<section>")
        logger.info(f"  POST /api/worker/<id>")
        logger.info(f"  POST /api/event")
        logger.info(f"  DELETE /api/worker/<id>")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Realm State Service")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8866, help="Port to bind to")
    parser.add_argument("--db", help="Database path (default: data/realm_state.db)")

    args = parser.parse_args()

    # Determine DB path
    if args.db:
        db_path = Path(args.db)
    else:
        try:
            from core.paths import get_base_dir
            db_path = get_base_dir() / "data" / "realm_state.db"
        except ImportError:
            db_path = Path(__file__).parent.parent / "data" / "realm_state.db"

    # Start server
    server = RealmStateServer(db_path, args.host, args.port)
    server.start()


if __name__ == "__main__":
    main()
