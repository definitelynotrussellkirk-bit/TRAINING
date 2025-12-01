"""
Realm State Service - HTTP server backed by SQLite.

Single source of truth for all Realm state, accessible over the network.

Usage:
    python3 -m realm.server --port 8866
    python3 -m realm.server --port 8866 --host 0.0.0.0  # Listen on all interfaces
"""

import argparse
import json
import logging
import sqlite3
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RealmStateDB:
    """SQLite backend for realm state."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
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

            # History table - state snapshots (optional, for time-series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    section TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_section_time ON history(section, timestamp DESC)")

            conn.commit()
            conn.close()

    def update_section(self, section: str, data: Dict[str, Any]):
        """Update a state section."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get existing data
            cursor.execute("SELECT data FROM state WHERE section = ?", (section,))
            row = cursor.fetchone()

            if row:
                # Merge with existing
                existing = json.loads(row[0])
                existing.update(data)
                existing["updated_at"] = datetime.now().isoformat()
                data_json = json.dumps(existing, default=str)
            else:
                # New section
                data["updated_at"] = datetime.now().isoformat()
                data_json = json.dumps(data, default=str)

            # Upsert
            cursor.execute("""
                INSERT OR REPLACE INTO state (section, data, updated_at)
                VALUES (?, ?, ?)
            """, (section, data_json, datetime.now().isoformat()))

            conn.commit()
            conn.close()

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Get a state section."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM state WHERE section = ?", (section,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return json.loads(row[0])
            return None

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state sections."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT section, data FROM state")
            rows = cursor.fetchall()
            conn.close()

            state = {}
            for section, data_json in rows:
                state[section] = json.loads(data_json)

            return state

    def add_event(self, event: Dict[str, Any]):
        """Add an event to the log."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

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

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, kind, channel, severity, message, icon, details
                FROM events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
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

    def add_history_snapshot(self, section: str, data: Dict[str, Any]):
        """Add a state snapshot to history (for time-series)."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO history (timestamp, section, data)
                VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), section, json.dumps(data, default=str)))

            conn.commit()
            conn.close()


class RealmStateHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RealmState service."""

    # Class variable - will be set by server
    db: RealmStateDB = None

    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _send_error_json(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/health":
                self._send_json({"status": "ok", "service": "realm"})

            elif path == "/api/state":
                # Get complete state
                state = self.db.get_all_state()
                events = self.db.get_events(50)
                self._send_json({
                    "state": state,
                    "events": events,
                    "timestamp": datetime.now().isoformat(),
                })

            elif path.startswith("/api/state/"):
                # Get specific section
                section = path.split("/")[-1]
                data = self.db.get_section(section)
                if data:
                    self._send_json(data)
                else:
                    self._send_error_json(f"Section not found: {section}", 404)

            elif path == "/api/events":
                # Get events
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["50"])[0])
                events = self.db.get_events(limit)
                self._send_json({"events": events})

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

            if path.startswith("/api/update/"):
                # Update specific section
                section = path.split("/")[-1]
                self.db.update_section(section, data)
                self._send_json({"success": True, "section": section})

            elif path == "/api/event":
                # Add event
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
                self.db.add_event(event)
                self._send_json({"success": True, "event": event})

            else:
                self._send_error_json(f"Not found: {path}", 404)

        except Exception as e:
            logger.error(f"POST error: {e}", exc_info=True)
            self._send_error_json(str(e), 500)

    def log_message(self, format, *args):
        """Override to use logger."""
        logger.debug(f"{self.address_string()} - {format % args}")


class RealmStateServer:
    """Realm State HTTP server."""

    def __init__(self, db_path: Path, host: str = "localhost", port: int = 8866):
        self.db_path = db_path
        self.host = host
        self.port = port
        self.db = RealmStateDB(db_path)
        self.server = None

    def start(self):
        """Start the server."""
        # Set class variable so handler can access DB
        RealmStateHandler.db = self.db

        self.server = HTTPServer((self.host, self.port), RealmStateHandler)
        logger.info(f"RealmState service starting on {self.host}:{self.port}")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Endpoints:")
        logger.info(f"  GET  /health")
        logger.info(f"  GET  /api/state")
        logger.info(f"  GET  /api/state/<section>")
        logger.info(f"  GET  /api/events?limit=N")
        logger.info(f"  POST /api/update/<section>")
        logger.info(f"  POST /api/event")

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
