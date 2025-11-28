#!/usr/bin/env python3
"""
The Tavern - Game UI Server for Realm of Training

The Tavern is where adventurers gather to:
    - View their hero's progress
    - Check quest status
    - Monitor guild skills
    - Access the vault

Start the Tavern:
    python3 tavern/server.py --port 8888

Then visit: http://localhost:8888

Process Management:
    - Writes PID to .pids/tavern.pid
    - Handles SIGTERM/SIGINT gracefully
    - Cleans up on exit
    - Use --daemon flag for background mode

RPG Flavor:
    The Tavern sits at the crossroads of the realm, a warm haven
    where travelers check their maps, count their gold, and plan
    their next adventure. The barkeep (this server) serves up
    fresh data from all corners of the realm.
"""

import argparse
import atexit
import json
import logging
import mimetypes
import os
import signal
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# Add parent to path for imports
TAVERN_DIR = Path(__file__).parent
BASE_DIR = TAVERN_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from watchtower.chronicle import get_chronicle

# Import events system
try:
    from events import get_broadcaster, get_recent, subscribe, Event
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False

# Import Saga reader (lazy to avoid circular imports)
_saga_reader = None

def get_saga_reader():
    """Get or create the Saga reader."""
    global _saga_reader
    if _saga_reader is None:
        try:
            from guild.saga import SagaReader
            _saga_reader = SagaReader(BASE_DIR)
        except ImportError:
            pass
    return _saga_reader


# Skill loader (lazy)
_skill_cache = None

def get_skills_data():
    """Load skill data from YAML configs + curriculum state."""
    global _skill_cache

    try:
        import yaml

        skills_dir = BASE_DIR / "configs" / "skills"
        curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"

        # Load curriculum state for levels/accuracy
        curriculum = {}
        if curriculum_file.exists():
            with open(curriculum_file) as f:
                curriculum = json.load(f)

        skills = []
        for yaml_file in skills_dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # Skip templates

            with open(yaml_file) as f:
                config = yaml.safe_load(f)

            skill_id = config.get("id", yaml_file.stem)

            # Handle ID mapping (YAML uses short IDs, curriculum uses full names)
            id_mapping = {
                "sy": "syllo",
                "bin": "binary",
            }
            curriculum_id = id_mapping.get(skill_id, skill_id)
            skill_state = curriculum.get("skills", {}).get(curriculum_id, {})

            # current_level = level being TRAINED on (not mastered)
            # mastered = training - 1 (minimum 0)
            max_level = config.get("max_level", 30)
            training = skill_state.get("current_level", 1)
            training = min(training, max_level)
            mastered = max(0, training - 1)

            # Get recent accuracy from history
            history = skill_state.get("accuracy_history", [])
            recent_acc = 0
            if history:
                # Average of last 3 evals
                recent = history[-3:]
                recent_acc = sum(r.get("accuracy", 0) for r in recent) / len(recent) * 100

            # Count evals at current training level
            at_level = [r for r in history if r.get("training_level", r.get("level")) == training]

            skills.append({
                "id": skill_id,
                "name": config.get("name", skill_id),
                "rpg_name": config.get("rpg_name", config.get("name", skill_id)),
                "rpg_description": config.get("rpg_description", config.get("description", "")),
                "icon": config.get("display", {}).get("icon", "⚔️"),
                "color": config.get("display", {}).get("color", "#888"),
                "short_name": config.get("display", {}).get("short_name", skill_id.upper()),
                "max_level": max_level,
                "mastered_level": mastered,
                "training_level": training,
                "accuracy": round(recent_acc, 1),
                "eval_count": len(at_level),
                "category": config.get("category", "general"),
                "description": config.get("description", ""),
            })

        # Sort by name
        skills.sort(key=lambda s: s["name"])

        return {
            "skills": skills,
            "active_skill": curriculum.get("active_skill", ""),
            "total_mastered": sum(s["mastered_level"] for s in skills),
        }

    except Exception as e:
        logger.error(f"Failed to load skills: {e}")
        return {"skills": [], "error": str(e)}


def get_vault_assets():
    """Load vault assets - checkpoints, sizes, etc."""
    from datetime import datetime

    assets = {
        "base_model": None,
        "checkpoints": [],
        "checkpoint_count": 0,
        "total_size_gb": 0,
        "last_updated": datetime.now().isoformat(),
    }

    try:
        # Load config for base model info
        config_path = BASE_DIR / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            model_path = Path(config.get("model_path", ""))
            if model_path.exists():
                size_bytes = sum(
                    f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                )
                assets["base_model"] = {
                    "name": config.get("model_name", "unknown"),
                    "display_name": config.get("model_display_name", "Base Model"),
                    "path": str(model_path),
                    "size_gb": round(size_bytes / (1024**3), 2),
                    "locked": config.get("locked", {}),
                }
                assets["total_size_gb"] += assets["base_model"]["size_gb"]

        # Scan for checkpoints
        current_model_dir = BASE_DIR / "current_model"
        if current_model_dir.exists():
            for item in sorted(current_model_dir.iterdir(), reverse=True):
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.split("-")[1])
                        size_bytes = sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        )
                        size_gb = round(size_bytes / (1024**3), 2)
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)

                        assets["checkpoints"].append({
                            "name": item.name,
                            "step": step,
                            "size_gb": size_gb,
                            "modified": mtime.isoformat(),
                            "age_hours": (datetime.now() - mtime).total_seconds() / 3600,
                            "is_latest": len(assets["checkpoints"]) == 0,
                            "is_champion": False,
                        })
                        assets["total_size_gb"] += size_gb
                    except (ValueError, OSError):
                        continue

        assets["checkpoint_count"] = len(assets["checkpoints"])

    except Exception as e:
        logger.error(f"Failed to load vault assets: {e}")

    return assets


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(log_to_file: bool = False) -> logging.Logger:
    """Configure logging for the Tavern."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_to_file:
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "tavern.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
    )
    return logging.getLogger("tavern")

logger = logging.getLogger("tavern")


# ==============================================================================
# PROCESS MANAGEMENT
# ==============================================================================

class ProcessManager:
    """Manages PID file and graceful shutdown."""

    PID_DIR = BASE_DIR / ".pids"
    PID_FILE = PID_DIR / "tavern.pid"

    _server: HTTPServer = None
    _running: bool = False

    @classmethod
    def write_pid(cls) -> None:
        """Write current PID to file."""
        cls.PID_DIR.mkdir(exist_ok=True)

        # Check for stale PID
        if cls.PID_FILE.exists():
            try:
                old_pid = int(cls.PID_FILE.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)
                # If we get here, process exists
                logger.warning(f"Tavern already running with PID {old_pid}")
                logger.warning("Use 'kill {old_pid}' to stop it first, or remove .pids/tavern.pid")
                sys.exit(1)
            except (ValueError, ProcessLookupError, PermissionError):
                # Stale PID file, remove it
                cls.PID_FILE.unlink(missing_ok=True)

        # Write our PID
        pid = os.getpid()
        cls.PID_FILE.write_text(str(pid))
        logger.info(f"PID {pid} written to {cls.PID_FILE}")

    @classmethod
    def remove_pid(cls) -> None:
        """Remove PID file on exit."""
        if cls.PID_FILE.exists():
            try:
                # Only remove if it's our PID
                stored_pid = int(cls.PID_FILE.read_text().strip())
                if stored_pid == os.getpid():
                    cls.PID_FILE.unlink()
                    logger.info("PID file removed")
            except (ValueError, FileNotFoundError):
                pass

    @classmethod
    def setup_signals(cls, server: HTTPServer) -> None:
        """Setup signal handlers for graceful shutdown."""
        cls._server = server
        cls._running = True

        def shutdown_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, shutting down gracefully...")
            cls._running = False
            if cls._server:
                # Shutdown in a thread to avoid blocking
                import threading
                threading.Thread(target=cls._server.shutdown).start()

        # Handle common termination signals
        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

        # SIGHUP - traditionally used for config reload, but we'll treat as shutdown
        try:
            signal.signal(signal.SIGHUP, shutdown_handler)
        except (AttributeError, ValueError):
            pass  # Windows doesn't have SIGHUP

        logger.info("Signal handlers registered (SIGTERM, SIGINT, SIGHUP)")

    @classmethod
    def register_cleanup(cls) -> None:
        """Register cleanup on exit."""
        atexit.register(cls.remove_pid)

    @classmethod
    def is_running(cls) -> bool:
        """Check if server should keep running."""
        return cls._running


class TavernHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the Tavern game UI."""

    # Upstream API (monitoring server or unified API)
    API_HOST = "localhost"
    API_PORT = 8081

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Route handling
        if path == "/" or path == "/game" or path == "/game.html":
            self._serve_template("game.html")

        elif path.startswith("/static/"):
            self._serve_static(path[8:])  # Remove /static/ prefix

        elif path == "/chronicle":
            self._serve_chronicle(query)

        elif path == "/chronicle/stats":
            self._serve_chronicle_stats()

        elif path == "/saga":
            self._serve_saga(query)

        elif path == "/saga/stats":
            self._serve_saga_stats()

        elif path == "/skills":
            self._serve_skills()

        elif path.startswith("/skill/"):
            skill_id = path.replace("/skill/", "").strip("/")
            if skill_id:
                self._serve_skill_page(skill_id)
            else:
                self._send_error(400, "Missing skill ID")

        elif path.startswith("/skill-data/"):
            skill_id = path.replace("/skill-data/", "").strip("/")
            if skill_id:
                self._serve_skill_data(skill_id)
            else:
                self._send_error(400, "Missing skill ID")

        elif path == "/settings" or path == "/settings.html":
            self._serve_template("settings.html")

        elif path == "/vault" or path == "/vault.html":
            self._serve_template("vault.html")

        elif path == "/config":
            self._serve_config()

        elif path == "/vault/assets":
            self._serve_vault_assets()

        elif path == "/vault/zones":
            self._serve_vault_zones()

        elif path == "/vault/zones/refresh":
            self._serve_vault_zones(refresh=True)

        # Ledger API - checkpoint stats and history
        elif path == "/ledger":
            self._serve_ledger_list(query)
        elif path == "/ledger/summary":
            self._serve_ledger_summary()
        elif path == "/ledger/best":
            self._serve_ledger_best(query)
        elif path.startswith("/ledger/"):
            step_str = path.replace("/ledger/", "")
            self._serve_ledger_checkpoint(step_str)

        # Checkpoint detail page
        elif path.startswith("/checkpoint/"):
            step_str = path.replace("/checkpoint/", "").strip("/")
            if step_str and step_str != "data":
                self._serve_template("checkpoint.html")
            else:
                self._send_error(400, "Missing checkpoint step")

        elif path.startswith("/checkpoint-data/"):
            step_str = path.replace("/checkpoint-data/", "").strip("/")
            if step_str:
                self._serve_checkpoint_data(step_str)
            else:
                self._send_error(400, "Missing checkpoint step")

        # Oracle - Talk to DIO (inference interface)
        elif path == "/oracle" or path == "/oracle.html":
            self._serve_template("oracle.html")
        elif path == "/oracle/hosts":
            self._serve_oracle_hosts()
        elif path == "/oracle/status":
            self._serve_oracle_status()

        # Guild - Skills and Passives
        elif path == "/guild" or path == "/guild.html" or path == "/guildhall.html":
            self._serve_template("guild.html")
        elif path == "/api/skills":
            self._serve_skills()
        elif path == "/api/curriculum":
            self._serve_curriculum()
        elif path == "/api/passives/definitions":
            self._serve_passives_definitions()
        elif path == "/api/passives/summary":
            self._serve_passives_summary()

        # Mantra - System prompt that's injected into all training
        elif path == "/mantra":
            self._serve_mantra()

        # Scheduler - Curriculum schedule configuration
        elif path == "/scheduler" or path == "/scheduler.html":
            self._serve_template("scheduler.html")
        elif path == "/api/scheduler":
            self._serve_scheduler_status()
        elif path == "/api/scheduler/presets":
            self._serve_scheduler_presets()

        # Quests - Training queue management
        elif path == "/quests" or path == "/quests.html":
            self._serve_template("quests.html")
        elif path == "/api/quests":
            self._serve_quests_data()
        elif path.startswith("/api/quests/preview/"):
            filename = path.replace("/api/quests/preview/", "")
            self._serve_quest_preview(filename, query)

        # Generators - Data generation control
        elif path == "/api/generators":
            self._serve_generators_status()

        # Skill APIs status
        elif path == "/api/skill-apis/status":
            self._serve_skill_apis_status()

        # Training daemon status
        elif path == "/api/daemon/status":
            self._serve_daemon_status()

        # Weaver (orchestrator) status
        elif path == "/api/weaver/status":
            self._serve_weaver_status()

        # Fresh game data API (replaces stale /api/unified)
        elif path == "/api/game":
            self._serve_game_data()

        # Eval results API
        elif path.startswith("/api/evals/"):
            skill_id = path.replace("/api/evals/", "").strip("/")
            level = query.get("level", [None])[0]
            self._serve_eval_results(skill_id, level)

        # Validation set API - get test problems for a level
        elif path.startswith("/api/validation/"):
            skill_id = path.replace("/api/validation/", "").strip("/")
            level = query.get("level", ["1"])[0]
            self._serve_validation_set(skill_id, level)

        # Events API - Global announcement channel
        elif path == "/api/events":
            self._serve_events_list(query)
        elif path == "/api/events/stream":
            self._serve_events_stream()
        elif path == "/api/events/stats":
            self._serve_events_stats()

        elif path.startswith("/api/"):
            self._proxy_api(path)

        elif path == "/health":
            self._send_json({"status": "open", "tavern": "welcoming travelers"})

        else:
            self._send_error(404, "Page not found")

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/config":
            self._save_config()
        elif path == "/config/save-default":
            self._save_default()
        elif path == "/config/restore-default":
            self._restore_default()
        # Oracle - Model loading and chat
        elif path == "/oracle/load":
            self._handle_oracle_load()
        elif path == "/oracle/chat":
            self._handle_oracle_chat()
        # Mantra - Save system prompt
        elif path == "/mantra":
            self._save_mantra()
        # Scheduler - Apply preset or update config
        elif path == "/api/scheduler/preset":
            self._apply_scheduler_preset()
        elif path == "/api/scheduler/config":
            self._save_scheduler_config()
        elif path.startswith("/api/scheduler/skill/"):
            self._toggle_skill()
        # Daemon control
        elif path == "/api/daemon/control":
            self._daemon_control()
        # Quests - Queue management
        elif path == "/api/quests/priority":
            self._change_quest_priority()
        elif path == "/api/quests/delete":
            self._delete_quest()
        elif path == "/api/quests/retry":
            self._retry_quest()
        # Generators - Toggle on/off
        elif path == "/api/generators/toggle":
            self._toggle_generator()
        else:
            self._send_error(404, "Endpoint not found")

    def _serve_template(self, template_name: str):
        """Serve an HTML template."""
        template_path = TAVERN_DIR / "templates" / template_name

        if not template_path.exists():
            self._send_error(404, f"Template not found: {template_name}")
            return

        content = template_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_static(self, static_path: str):
        """Serve a static file (CSS, JS, images)."""
        file_path = TAVERN_DIR / "static" / static_path

        if not file_path.exists() or not file_path.is_file():
            self._send_error(404, f"Static file not found: {static_path}")
            return

        # Security: ensure path is within static directory
        try:
            file_path.resolve().relative_to((TAVERN_DIR / "static").resolve())
        except ValueError:
            self._send_error(403, "Access denied")
            return

        content = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "max-age=3600")  # Cache static files
        self.end_headers()
        self.wfile.write(content)

    def _proxy_api(self, path: str):
        """Proxy API requests to the monitoring server."""
        api_url = f"http://{self.API_HOST}:{self.API_PORT}{path}"

        try:
            with urllib.request.urlopen(api_url, timeout=10) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type", "application/json")

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)

        except urllib.error.URLError as e:
            logger.warning(f"API proxy failed: {e}")
            self._send_json({
                "error": "API unavailable",
                "detail": str(e),
            }, 502)

        except Exception as e:
            logger.error(f"API proxy error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        """Send an error response."""
        self._send_json({"error": message, "status": status}, status)

    # =========================================
    # Events API - Global Announcement Channel
    # =========================================

    def _serve_events_list(self, query: dict):
        """Serve recent events as JSON list.

        Reads directly from the events.jsonl file to see events
        from all processes (daemon, data_manager, etc.).
        """
        try:
            limit = int(query.get("limit", [50])[0])

            # Read directly from events file for cross-process visibility
            events_file = BASE_DIR / "status" / "events.jsonl"
            events = []

            if events_file.exists():
                with open(events_file, "r") as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-limit:]
                    for line in reversed(recent_lines):  # Most recent first
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass

            self._send_json({
                "events": events,
                "count": len(events),
            })
        except Exception as e:
            logger.error(f"Events list error: {e}")
            self._send_json({"events": [], "error": str(e)})

    def _serve_events_stream(self):
        """Serve Server-Sent Events stream for real-time updates."""
        if not EVENTS_AVAILABLE:
            self._send_error(503, "Events system not available")
            return

        # Limit concurrent SSE connections to prevent thread explosion
        try:
            broadcaster = get_broadcaster()
            if broadcaster.subscriber_count() > 20:
                logger.warning(f"Too many SSE subscribers ({broadcaster.subscriber_count()}), rejecting")
                self._send_error(503, "Too many connections, try again later")
                return
        except Exception:
            pass  # Continue if we can't check

        try:
            # Send SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Stream events (5s timeout for faster disconnect detection)
            for event in subscribe(timeout=5):
                if event is None:
                    # Keepalive
                    self.wfile.write(b": keepalive\n\n")
                else:
                    # Send event in SSE format
                    self.wfile.write(event.to_sse().encode("utf-8"))
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected
            logger.debug("SSE client disconnected")
        except Exception as e:
            logger.error(f"Events stream error: {e}")

    def _serve_events_stats(self):
        """Serve event system statistics."""
        if not EVENTS_AVAILABLE:
            self._send_json({"error": "Events system not available"}, 503)
            return

        try:
            broadcaster = get_broadcaster()
            self._send_json(broadcaster.stats())
        except Exception as e:
            logger.error(f"Events stats error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_chronicle(self, query: dict):
        """Serve chronicle entries."""
        try:
            limit = int(query.get("limit", [50])[0])
            chronicle = get_chronicle()
            entries = chronicle.recent(limit=limit)
            self._send_json({"entries": entries, "count": len(entries)})
        except Exception as e:
            logger.error(f"Chronicle error: {e}")
            self._send_json({"entries": [], "error": str(e)})

    def _serve_chronicle_stats(self):
        """Serve chronicle statistics."""
        try:
            chronicle = get_chronicle()
            stats = chronicle.get_stats()
            self._send_json(stats)
        except Exception as e:
            logger.error(f"Chronicle stats error: {e}")
            self._send_json({"error": str(e)})

    def _serve_saga(self, query: dict):
        """Serve saga tales (narrative log)."""
        try:
            reader = get_saga_reader()
            if reader is None:
                self._send_json({
                    "tales": [],
                    "error": "Saga module not available"
                })
                return

            limit = int(query.get("limit", [50])[0])
            category = query.get("category", [None])[0]
            search = query.get("q", [None])[0]

            # Get tales based on query
            if search:
                tales = reader.search(search, limit=limit)
            elif category:
                tales = reader.by_category(category, limit=limit)
            else:
                tales = reader.recent(limit=limit)

            # Format for JSON
            formatted = []
            for tale in tales:
                formatted.append({
                    "time": tale.timestamp.strftime("%H:%M:%S"),
                    "date": tale.timestamp.strftime("%Y-%m-%d"),
                    "icon": tale.icon,
                    "message": tale.message,
                    "type": tale.event_type,
                    "category": tale.category,
                    "display": tale.format_display(),
                    "data": tale.data,
                })

            self._send_json({
                "tales": formatted,
                "count": len(formatted),
            })

        except Exception as e:
            logger.error(f"Saga error: {e}")
            self._send_json({"tales": [], "error": str(e)})

    def _serve_saga_stats(self):
        """Serve saga statistics."""
        try:
            reader = get_saga_reader()
            if reader is None:
                self._send_json({"error": "Saga module not available"})
                return

            stats = reader.stats()
            self._send_json(stats)

        except Exception as e:
            logger.error(f"Saga stats error: {e}")
            self._send_json({"error": str(e)})

    def _serve_game_data(self):
        """
        Serve unified game data fresh from status files.

        This replaces the stale /api/unified proxy with direct file reads.
        All data is read fresh on each request.
        """
        try:
            import subprocess

            response = {
                "timestamp": datetime.now().isoformat(),
                "training": None,
                "gpu": None,
                "curriculum": None,
                "vault": None,
                "comparison": None,
            }

            # 1. Training status
            training_file = BASE_DIR / "status" / "training_status.json"
            if training_file.exists():
                try:
                    with open(training_file) as f:
                        training_data = json.load(f)

                    # Calculate progress_percent from batch_step / batch_total_steps
                    batch_step = training_data.get("batch_step", 0)
                    batch_total = training_data.get("batch_total_steps", 0)
                    if batch_total > 0:
                        training_data["progress_percent"] = (batch_step / batch_total) * 100
                    else:
                        training_data["progress_percent"] = 0

                    response["training"] = training_data
                except Exception as e:
                    logger.warning(f"Failed to read training status: {e}")

            # 2. GPU stats (nvidia-smi)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 4:
                        response["gpu"] = {
                            "utilization_pct": float(parts[0]),
                            "vram_used_gb": float(parts[1]) / 1024,
                            "vram_total_gb": float(parts[2]) / 1024,
                            "temperature_c": float(parts[3]),
                        }
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")

            # 3. Curriculum state
            curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
            if curriculum_file.exists():
                try:
                    with open(curriculum_file) as f:
                        curriculum = json.load(f)

                    # Transform to expected format
                    skills_data = {}
                    for skill_id, skill_state in curriculum.get("skills", {}).items():
                        skills_data[skill_id] = {
                            "current_level": skill_state.get("current_level", 0),
                            "recent_accuracy": skill_state.get("recent_accuracy", 0),
                            "eval_count": len(skill_state.get("accuracy_history", [])),
                        }

                    response["curriculum"] = {
                        "active_skill": curriculum.get("active_skill"),
                        "skills": skills_data,
                    }
                except Exception as e:
                    logger.warning(f"Failed to read curriculum: {e}")

            # 4. Vault data (checkpoints)
            try:
                vault_data = get_vault_assets()
                response["vault"] = {
                    "checkpoint_count": vault_data.get("checkpoint_count", 0),
                    "total_size_gb": vault_data.get("total_size_gb", 0),
                    "checkpoints": vault_data.get("checkpoints", []),
                }
            except Exception as e:
                logger.warning(f"Failed to get vault data: {e}")

            # 5. Best checkpoint (from ledger)
            try:
                from core.checkpoint_ledger import get_ledger
                ledger = get_ledger()
                best = ledger.get_best(metric="train_loss")
                if best:
                    response["comparison"] = {
                        "best_checkpoint": f"checkpoint-{best.step}",
                        "best_step": best.step,
                        "best_loss": best.train_loss,
                    }
            except Exception as e:
                logger.warning(f"Failed to get best checkpoint: {e}")

            self._send_json(response)

        except Exception as e:
            logger.error(f"Game data error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _serve_validation_set(self, skill_id: str, level: str):
        """
        Serve validation set (test problems) for a skill level.

        Serves from static validation files (5 fixed problems per level).
        Auto-detects validation file from skill ID.
        """
        try:
            # Auto-detect validation file: data/validation/{skill_id}_validation.json
            # Also check common aliases
            aliases = {
                "syllo": "sy",
                "binary": "bin",
            }
            normalized_id = aliases.get(skill_id, skill_id)
            val_file = BASE_DIR / "data" / "validation" / f"{normalized_id}_validation.json"

            if not val_file.exists():
                self._send_json({
                    "error": f"Validation set not found. Run: python3 scripts/generate_validation_sets.py {normalized_id}",
                    "skill": skill_id,
                }, 404)
                return

            level_num = int(level) if level else 1
            level_key = str(level_num)

            with open(val_file) as f:
                validation = json.load(f)

            problems = validation.get(level_key, [])

            self._send_json({
                "skill": skill_id,
                "level": level_num,
                "count": len(problems),
                "problems": problems,
                "static": True,  # Indicates these are fixed problems
            })

        except Exception as e:
            logger.error(f"Validation set error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_eval_results(self, skill_id: str, level: Optional[str] = None):
        """
        Serve eval results for a skill.

        Returns latest eval and history from eval_results_history.json
        """
        try:
            # Map skill IDs
            id_mapping = {"sy": "syllo", "bin": "binary", "syllo": "syllo", "binary": "binary"}
            skill_name = id_mapping.get(skill_id, skill_id)

            response = {
                "skill": skill_id,
                "skill_name": skill_name,
                "level": int(level) if level else None,
                "latest": None,
                "history": [],
            }

            # Load current eval results
            eval_file = BASE_DIR / "status" / "curriculum_eval.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    eval_data = json.load(f)

                last_eval = eval_data.get("last_eval", {})
                if last_eval.get("skill") == skill_name:
                    # Filter by level if specified
                    if level is None or last_eval.get("level") == int(level):
                        response["latest"] = {
                            "level": last_eval.get("level"),
                            "level_name": last_eval.get("level_name"),
                            "accuracy": last_eval.get("accuracy"),
                            "correct": last_eval.get("correct"),
                            "total": last_eval.get("total"),
                            "timestamp": last_eval.get("timestamp"),
                            "step": last_eval.get("step"),
                            "results": last_eval.get("results", []),
                        }

            # Load eval history (last 5 per level)
            history_file = BASE_DIR / "status" / "eval_results_history.json"
            if history_file.exists():
                try:
                    with open(history_file) as f:
                        history_data = json.load(f)

                    skill_history = history_data.get(skill_name, {})
                    if level:
                        # Get history for specific level
                        level_key = str(level)
                        response["history"] = skill_history.get(level_key, [])
                    else:
                        # Get all levels
                        response["history_by_level"] = skill_history
                except Exception as e:
                    logger.warning(f"Failed to load eval history: {e}")

            self._send_json(response)

        except Exception as e:
            logger.error(f"Eval results error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_skills(self):
        """Serve skill data from YAML configs."""
        try:
            data = get_skills_data()
            self._send_json(data)
        except Exception as e:
            logger.error(f"Skills error: {e}")
            self._send_json({"skills": [], "error": str(e)})

    def _serve_skill_page(self, skill_id: str):
        """Serve the skill detail page."""
        self._serve_template("skill.html")

    def _serve_skill_data(self, skill_id: str):
        """Serve full skill data from YAML + curriculum state."""
        try:
            import yaml

            # Find the skill YAML
            skills_dir = BASE_DIR / "configs" / "skills"
            yaml_file = skills_dir / f"{skill_id}.yaml"

            if not yaml_file.exists():
                self._send_json({"error": f"Skill '{skill_id}' not found"}, 404)
                return

            # Load full YAML config
            with open(yaml_file) as f:
                config = yaml.safe_load(f)

            # Load curriculum state
            curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
            curriculum = {}
            if curriculum_file.exists():
                with open(curriculum_file) as f:
                    curriculum = json.load(f)

            # Map skill ID to curriculum ID
            id_mapping = {"sy": "syllo", "bin": "binary"}
            curriculum_id = id_mapping.get(skill_id, skill_id)
            skill_state = curriculum.get("skills", {}).get(curriculum_id, {})

            # Build response
            # current_level = highest level MASTERED (0 = nothing mastered)
            # training_level = mastered + 1
            max_level = config.get("max_level", 30)
            mastered = skill_state.get("current_level", 0)
            training = min(mastered + 1, max_level)

            # Fill in missing levels in level_progression (extrapolate from last defined)
            level_prog = config.get("level_progression", {})
            if level_prog and len(level_prog) < max_level:
                # Find highest defined level (keys may be int or str from YAML)
                defined_levels = [int(k) for k in level_prog.keys() if str(k).isdigit()]
                if defined_levels:
                    last_level = max(defined_levels)
                    # Try both int and str keys
                    last_data = level_prog.get(last_level, level_prog.get(str(last_level), {}))
                    # Fill in missing levels
                    for lvl in range(last_level + 1, max_level + 1):
                        level_prog[str(lvl)] = {
                            "name": f"Level {lvl}",
                            "desc": f"Beyond L{last_level} - extrapolated",
                            **{k: v for k, v in last_data.items() if k != "name"}
                        }
                    config["level_progression"] = level_prog

            # Get accuracy history
            history = skill_state.get("accuracy_history", [])

            # Recent accuracy
            recent_acc = 0
            if history:
                recent = history[-3:]
                recent_acc = sum(r.get("accuracy", 0) for r in recent) / len(recent) * 100

            # Count evals at training level
            at_level = [r for r in history if r.get("training_level", r.get("level")) == training]

            response = {
                "id": skill_id,
                "config": config,
                "state": {
                    "mastered_level": mastered,
                    "training_level": training,
                    "accuracy": round(recent_acc, 1),
                    "eval_count": len(at_level),
                    "total_evals": len(history),
                    "accuracy_history": history[-20:],  # Last 20 evals
                },
                "is_active": curriculum.get("active_skill") == curriculum_id,
            }

            self._send_json(response)

        except Exception as e:
            logger.error(f"Skill data error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _serve_config(self):
        """Serve config.json."""
        try:
            config_path = BASE_DIR / "config.json"
            if not config_path.exists():
                self._send_json({"error": "Config not found"}, 404)
                return

            with open(config_path) as f:
                config = json.load(f)

            self._send_json(config)
        except Exception as e:
            logger.error(f"Config read error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # SCHEDULER API - Curriculum schedule configuration
    # =========================================================================

    def _serve_scheduler_status(self):
        """Get current scheduler status."""
        try:
            from guild.dispatch.scheduler import get_scheduler
            scheduler = get_scheduler()
            status = scheduler.get_status()
            self._send_json(status)
        except ImportError as e:
            self._send_json({"error": f"Scheduler not available: {e}"}, 500)
        except Exception as e:
            logger.error(f"Scheduler status error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_scheduler_presets(self):
        """Get available scheduler presets."""
        try:
            import yaml
            config_path = BASE_DIR / "configs" / "schedule.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                presets = config.get("presets", {})
                self._send_json({
                    "presets": [{
                        "id": k,
                        "description": v.get("description", k),
                        "strategy": v.get("strategy"),
                    } for k, v in presets.items()]
                })
            else:
                self._send_json({"presets": []})
        except Exception as e:
            logger.error(f"Scheduler presets error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _apply_scheduler_preset(self):
        """Apply a scheduler preset."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())
            preset_name = body.get("preset")

            if not preset_name:
                self._send_json({"success": False, "error": "Missing preset name"}, 400)
                return

            from guild.dispatch.scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.apply_preset(preset_name)

            self._send_json({
                "success": True,
                "preset": preset_name,
                "status": scheduler.get_status(),
            })
        except ValueError as e:
            self._send_json({"success": False, "error": str(e)}, 400)
        except Exception as e:
            logger.error(f"Apply preset error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _save_scheduler_config(self):
        """Save scheduler configuration."""
        try:
            import yaml
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            config_path = BASE_DIR / "configs" / "schedule.yaml"

            # Load current config
            current_config = {}
            if config_path.exists():
                with open(config_path) as f:
                    current_config = yaml.safe_load(f) or {}

            # Update with new values
            if "strategy" in body:
                current_config["strategy"] = body["strategy"]

            if "skills" in body:
                if "skills" not in current_config:
                    current_config["skills"] = {}
                for skill_id, skill_config in body["skills"].items():
                    if skill_id not in current_config["skills"]:
                        current_config["skills"][skill_id] = {}
                    current_config["skills"][skill_id].update(skill_config)

            if "settings" in body:
                if "settings" not in current_config:
                    current_config["settings"] = {}
                current_config["settings"].update(body["settings"])

            # Save
            with open(config_path, "w") as f:
                yaml.dump(current_config, f, default_flow_style=False)

            # Reload scheduler
            from guild.dispatch.scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.config = scheduler._load_config()

            self._send_json({
                "success": True,
                "config": current_config,
            })
        except Exception as e:
            logger.error(f"Save scheduler config error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _toggle_skill(self):
        """Toggle a skill's enabled status."""
        try:
            import yaml
            # Extract skill_id from path: /api/scheduler/skill/sy -> sy
            skill_id = self.path.split("/")[-1]

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())
            enabled = body.get("enabled", True)

            config_path = BASE_DIR / "configs" / "schedule.yaml"

            # Load current config
            current_config = {}
            if config_path.exists():
                with open(config_path) as f:
                    current_config = yaml.safe_load(f) or {}

            # Update skill enabled status
            if "skills" not in current_config:
                current_config["skills"] = {}
            if skill_id not in current_config["skills"]:
                current_config["skills"][skill_id] = {}
            current_config["skills"][skill_id]["enabled"] = enabled

            # Save
            with open(config_path, "w") as f:
                yaml.dump(current_config, f, default_flow_style=False)

            # Reload scheduler
            from guild.dispatch.scheduler import get_scheduler
            scheduler = get_scheduler()
            scheduler.config = scheduler._load_config()

            self._send_json({
                "success": True,
                "skill_id": skill_id,
                "enabled": enabled,
            })
        except Exception as e:
            logger.error(f"Toggle skill error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    # ==========================================
    # TRAINING DAEMON Handlers
    # ==========================================

    def _serve_daemon_status(self):
        """Get training daemon status."""
        try:
            import subprocess

            # Check if daemon process is running
            pid_file = BASE_DIR / ".daemon.pid"
            daemon_running = False
            daemon_pid = None

            if pid_file.exists():
                try:
                    daemon_pid = int(pid_file.read_text().strip())
                    # Check if process exists
                    import os
                    os.kill(daemon_pid, 0)
                    daemon_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    daemon_running = False

            # Get training status from status file
            status_file = BASE_DIR / "status" / "training_status.json"
            training_status = "unknown"
            current_step = 0
            current_file = None
            last_update = None

            if status_file.exists():
                try:
                    with open(status_file) as f:
                        status_data = json.load(f)
                    training_status = status_data.get("status", "unknown")
                    current_step = status_data.get("current_step", 0)
                    current_file = status_data.get("current_file")
                    last_update = status_data.get("timestamp")
                except:
                    pass

            # Get control signals
            control_dir = BASE_DIR / "control"
            signals = {
                "pause": (control_dir / ".pause").exists(),
                "stop": (control_dir / ".stop").exists(),
                "resume": (control_dir / ".resume").exists(),
            }

            self._send_json({
                "daemon_running": daemon_running,
                "daemon_pid": daemon_pid,
                "training_status": training_status,
                "current_step": current_step,
                "current_file": current_file,
                "last_update": last_update,
                "signals": signals,
            })
        except Exception as e:
            logger.error(f"Daemon status error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_weaver_status(self):
        """Get Weaver (daemon orchestrator) status."""
        try:
            pid_file = BASE_DIR / ".pids" / "weaver.pid"
            weaver_running = False
            weaver_pid = None

            if pid_file.exists():
                try:
                    weaver_pid = int(pid_file.read_text().strip())
                    os.kill(weaver_pid, 0)
                    weaver_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    weaver_running = False

            # Get thread status from weaver if it's running
            threads = {}
            if weaver_running:
                try:
                    from weaver.weaver import Weaver
                    w = Weaver(str(BASE_DIR))
                    tapestry = w.check_tapestry()
                    for name, info in tapestry.items():
                        threads[name] = {
                            "name": info["name"],
                            "alive": info["alive"],
                            "restarts": info.get("restarts", 0)
                        }
                except Exception as e:
                    logger.debug(f"Could not get thread status: {e}")

            self._send_json({
                "weaver_running": weaver_running,
                "weaver_pid": weaver_pid,
                "threads": threads,
                "healthy_count": sum(1 for t in threads.values() if t.get("alive", False)),
                "total_threads": len(threads)
            })
        except Exception as e:
            logger.error(f"Weaver status error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _daemon_control(self):
        """Control training daemon (start/stop/pause/resume)."""
        try:
            import subprocess

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())
            action = body.get("action")

            if action not in ["start", "stop", "pause", "resume", "kill"]:
                self._send_json({"success": False, "error": f"Invalid action: {action}"}, 400)
                return

            control_dir = BASE_DIR / "control"
            control_dir.mkdir(exist_ok=True)

            if action == "start":
                # Start daemon in background
                daemon_script = BASE_DIR / "core" / "training_daemon.py"
                log_file = BASE_DIR / "logs" / "training_daemon.log"
                subprocess.Popen(
                    ["nohup", "python3", str(daemon_script)],
                    stdout=open(log_file, "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(BASE_DIR)
                )
                self._send_json({"success": True, "action": "start", "message": "Daemon starting..."})

            elif action == "stop":
                # Create stop signal
                (control_dir / ".stop").touch()
                # Remove pause if exists
                (control_dir / ".pause").unlink(missing_ok=True)
                self._send_json({"success": True, "action": "stop", "message": "Stop signal sent"})

            elif action == "pause":
                (control_dir / ".pause").touch()
                (control_dir / ".resume").unlink(missing_ok=True)
                self._send_json({"success": True, "action": "pause", "message": "Pause signal sent"})

            elif action == "resume":
                (control_dir / ".resume").touch()
                (control_dir / ".pause").unlink(missing_ok=True)
                self._send_json({"success": True, "action": "resume", "message": "Resume signal sent"})

            elif action == "kill":
                # Force kill daemon
                pid_file = BASE_DIR / ".daemon.pid"
                if pid_file.exists():
                    try:
                        pid = int(pid_file.read_text().strip())
                        import os
                        os.kill(pid, 9)  # SIGKILL
                        pid_file.unlink()
                        self._send_json({"success": True, "action": "kill", "message": f"Killed daemon (PID {pid})"})
                    except Exception as e:
                        self._send_json({"success": False, "error": str(e)}, 500)
                else:
                    self._send_json({"success": False, "error": "No daemon PID file found"}, 404)

        except Exception as e:
            logger.error(f"Daemon control error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    # ==========================================
    # QUESTS (Training Queue) Handlers
    # ==========================================

    def _serve_quests_data(self):
        """Serve quest/queue data for the Quests page."""
        try:
            from core.training_queue import TrainingQueue
            from datetime import datetime

            queue = TrainingQueue(BASE_DIR)
            status = queue.get_queue_status()

            # Get queued files
            queued = queue.list_queue()

            # Get processing files
            processing = []
            processing_dir = BASE_DIR / "queue" / "processing"
            if processing_dir.exists():
                for f in processing_dir.glob("*.jsonl"):
                    processing.append({
                        "file": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "started_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        "progress": 0,  # Would need to read training status for actual progress
                    })

            # Get recently completed (last 100)
            recently_completed = []
            completed_dir = BASE_DIR / "queue" / "recently_completed"
            if completed_dir.exists():
                files = sorted(completed_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:100]
                for f in files:
                    recently_completed.append({
                        "file": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "completed_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        "priority": "completed",
                    })

            # Get failed files
            failed = []
            failed_dir = BASE_DIR / "queue" / "failed"
            if failed_dir.exists():
                for f in sorted(failed_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
                    failed.append({
                        "file": f.name,
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "failed_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        "priority": "failed",
                    })

            self._send_json({
                "status": status,
                "queued": queued,
                "processing": processing,
                "recently_completed": recently_completed,
                "failed": failed,
            })
        except Exception as e:
            logger.error(f"Quests data error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_quest_preview(self, filename: str, query: dict):
        """Serve preview examples from a queue file."""
        import random
        from urllib.parse import unquote

        try:
            filename = unquote(filename)  # Handle URL-encoded filenames
            count = min(int(query.get("count", ["5"])[0]), 20)

            # Find the file in queue directories
            file_path = None
            for location in ("high", "normal", "low", "processing", "recently_completed", "failed"):
                candidate = BASE_DIR / "queue" / location / filename
                if candidate.exists():
                    file_path = candidate
                    break

            # Also check inbox
            if not file_path:
                candidate = BASE_DIR / "inbox" / filename
                if candidate.exists():
                    file_path = candidate

            if not file_path:
                self._send_json({"error": f"File not found: {filename}", "filename": filename}, 404)
                return

            # Read all examples
            examples = []
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not examples:
                self._send_json({"error": "No valid examples in file", "filename": filename, "total_examples": 0}, 400)
                return

            # Select random examples
            selected = random.sample(examples, min(count, len(examples)))

            # Extract prompt and response for display
            previews = []
            for ex in selected:
                # Handle different formats (messages array or prompt/response)
                if "messages" in ex:
                    messages = ex["messages"]
                    prompt = None
                    response = None
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user" and not prompt:
                            prompt = content
                        elif role == "assistant":
                            response = content
                    previews.append({
                        "prompt": prompt or "(no prompt)",
                        "response": response or "(no response)",
                        "format": "messages"
                    })
                elif "prompt" in ex:
                    previews.append({
                        "prompt": ex.get("prompt", "(no prompt)"),
                        "response": ex.get("response", ex.get("completion", "(no response)")),
                        "format": "prompt_response"
                    })
                else:
                    # Unknown format - show raw
                    previews.append({
                        "prompt": str(ex)[:500],
                        "response": "(unknown format)",
                        "format": "raw"
                    })

            self._send_json({
                "filename": filename,
                "total_examples": len(examples),
                "preview_count": len(previews),
                "previews": previews
            })
        except Exception as e:
            logger.error(f"Quest preview error: {e}")
            self._send_json({"error": str(e), "filename": filename}, 500)

    def _change_quest_priority(self):
        """Change priority of a queued quest."""
        try:
            from core.training_queue import TrainingQueue

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            filename = body.get("filename")
            new_priority = body.get("priority")

            if not filename or not new_priority:
                self._send_json({"success": False, "error": "Missing filename or priority"}, 400)
                return

            if new_priority not in ("high", "normal", "low"):
                self._send_json({"success": False, "error": "Invalid priority"}, 400)
                return

            queue = TrainingQueue(BASE_DIR)
            success = queue.change_priority(filename, new_priority)

            if success:
                self._send_json({"success": True})
            else:
                self._send_json({"success": False, "error": "File not found in queue"}, 404)

        except Exception as e:
            logger.error(f"Change priority error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _delete_quest(self):
        """Delete a quest from the queue."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            filename = body.get("filename")
            if not filename:
                self._send_json({"success": False, "error": "Missing filename"}, 400)
                return

            # Search in all queue directories
            queue_dirs = [
                BASE_DIR / "queue" / "high",
                BASE_DIR / "queue" / "normal",
                BASE_DIR / "queue" / "low",
            ]

            for queue_dir in queue_dirs:
                file_path = queue_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted quest: {filename}")
                    self._send_json({"success": True})
                    return

            self._send_json({"success": False, "error": "File not found in queue"}, 404)

        except Exception as e:
            logger.error(f"Delete quest error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _retry_quest(self):
        """Retry a failed quest by moving it back to the queue."""
        try:
            import shutil

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            filename = body.get("filename")
            priority = body.get("priority", "normal")

            if not filename:
                self._send_json({"success": False, "error": "Missing filename"}, 400)
                return

            failed_path = BASE_DIR / "queue" / "failed" / filename
            if not failed_path.exists():
                self._send_json({"success": False, "error": "File not found in failed queue"}, 404)
                return

            # Move to target priority queue
            target_dir = BASE_DIR / "queue" / priority
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            shutil.move(str(failed_path), str(target_path))
            logger.info(f"Retried quest: {filename} -> {priority}")

            self._send_json({"success": True})

        except Exception as e:
            logger.error(f"Retry quest error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    # ==========================================
    # GENERATORS (Data Forge) Handlers
    # ==========================================

    def _serve_generators_status(self):
        """Serve status of all data generators."""
        try:
            config_path = BASE_DIR / "config.json"
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            generators = {}

            # Auto-generate (skill training)
            auto_gen = config.get("auto_generate", {})
            generators["auto_generate"] = {
                "enabled": auto_gen.get("enabled", False),
                "last_run": None,  # Would need to track this separately
                "count": auto_gen.get("count", 100),
            }

            # Self-correction
            self_corr = config.get("self_correction", {})
            self_corr_status = BASE_DIR / "status" / "self_correction.json"
            last_self_corr = None
            if self_corr_status.exists():
                try:
                    with open(self_corr_status) as f:
                        sc_data = json.load(f)
                        last_self_corr = sc_data.get("last_run")
                except:
                    pass
            generators["self_correction"] = {
                "enabled": self_corr.get("enabled", False),
                "last_run": last_self_corr,
            }

            # Curriculum
            curriculum = config.get("curriculum", {})
            generators["curriculum"] = {
                "enabled": curriculum.get("enabled", False),
                "last_run": None,
            }

            # Discrimination generator (separate status file)
            discrim_status = BASE_DIR / "status" / "discrimination_generator.json"
            last_discrim = None
            if discrim_status.exists():
                try:
                    with open(discrim_status) as f:
                        disc_data = json.load(f)
                        last_discrim = disc_data.get("last_run")
                except:
                    pass
            generators["discrimination"] = {
                "enabled": True,  # No config toggle currently
                "last_run": last_discrim,
            }

            self._send_json(generators)

        except Exception as e:
            logger.error(f"Generators status error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _toggle_generator(self):
        """Toggle a generator on/off."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            generator_id = body.get("generator")
            enabled = body.get("enabled", False)

            if not generator_id:
                self._send_json({"success": False, "error": "Missing generator id"}, 400)
                return

            config_path = BASE_DIR / "config.json"
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            # Map generator IDs to config keys
            config_mapping = {
                "auto_generate": "auto_generate",
                "self_correction": "self_correction",
                "curriculum": "curriculum",
            }

            if generator_id not in config_mapping:
                self._send_json({"success": False, "error": f"Unknown generator: {generator_id}"}, 400)
                return

            config_key = config_mapping[generator_id]
            if config_key not in config:
                config[config_key] = {}
            config[config_key]["enabled"] = enabled

            # Save config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Toggled generator {generator_id} to {enabled}")
            self._send_json({"success": True, "generator": generator_id, "enabled": enabled})

        except Exception as e:
            logger.error(f"Toggle generator error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _serve_skill_apis_status(self):
        """Check health of skill API servers (SY on 8080, BIN on 8090)."""
        import socket

        def check_api(host: str, port: int, timeout: float = 3.0) -> dict:
            """Quick check if API is responding."""
            try:
                # First check if port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()

                if result != 0:
                    return {"online": False, "error": "Port closed"}

                # Try HTTP health check
                import urllib.request
                url = f"http://{host}:{port}/health"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        return {"online": True, "data": data}
                    return {"online": False, "error": f"Status {resp.status}"}
            except Exception as e:
                return {"online": False, "error": str(e)}

        status = {
            "sy": check_api("localhost", 8080),
            "bin": check_api("localhost", 8090),
        }

        self._send_json(status)

    def _serve_mantra(self):
        """Serve the MANTRA (system prompt) that's injected into all training."""
        try:
            prompts_path = BASE_DIR / "core" / "prompts.py"

            # Read current values
            base_prompt = ""
            template = ""

            if prompts_path.exists():
                content = prompts_path.read_text()
                # Extract BASE_PROMPT
                import re
                base_match = re.search(r'BASE_PROMPT\s*=\s*["\'](.+?)["\']', content)
                template_match = re.search(r'BASE_PROMPT_TEMPLATE\s*=\s*["\'](.+?)["\']', content)

                if base_match:
                    base_prompt = base_match.group(1)
                if template_match:
                    template = template_match.group(1)

            # Get formatted version with today's date
            from datetime import datetime
            formatted = template.replace("{date}", datetime.now().strftime("%Y-%m-%d"))

            self._send_json({
                "base_prompt": base_prompt,
                "template": template,
                "formatted": formatted,
                "file": str(prompts_path),
                "description": "The MANTRA is auto-injected as system prompt into EVERY training example.",
            })
        except Exception as e:
            logger.error(f"Mantra read error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _save_mantra(self):
        """Save the MANTRA (system prompt)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            new_prompt = body.get("base_prompt", "").strip()
            if not new_prompt:
                self._send_json({"success": False, "error": "base_prompt is required"}, 400)
                return

            prompts_path = BASE_DIR / "core" / "prompts.py"

            if not prompts_path.exists():
                self._send_json({"success": False, "error": "prompts.py not found"}, 404)
                return

            # Create backup
            backup_path = prompts_path.with_suffix(".py.bak")
            import shutil
            shutil.copy2(prompts_path, backup_path)

            # Read current content
            content = prompts_path.read_text()

            # Update BASE_PROMPT
            import re
            content = re.sub(
                r'(BASE_PROMPT\s*=\s*)["\'](.+?)["\']',
                f'\\1"{new_prompt}"',
                content
            )

            # Update BASE_PROMPT_TEMPLATE (add date prefix)
            new_template = f"Today is {{date}}. {new_prompt}"
            content = re.sub(
                r'(BASE_PROMPT_TEMPLATE\s*=\s*)["\'](.+?)["\']',
                f'\\1"{new_template}"',
                content
            )

            # Write back
            prompts_path.write_text(content)

            logger.info(f"Mantra updated: {new_prompt}")
            self._send_json({
                "success": True,
                "base_prompt": new_prompt,
                "template": new_template,
                "backup": str(backup_path),
            })
        except Exception as e:
            logger.error(f"Mantra save error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _serve_vault_assets(self):
        """Serve vault assets - base model, checkpoints, etc."""
        try:
            import os
            from datetime import datetime

            assets = {
                "base_model": None,
                "checkpoints": [],
                "total_size_gb": 0,
                "last_updated": datetime.now().isoformat(),
            }

            # Load config for base model info
            config_path = BASE_DIR / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                model_path = Path(config.get("model_path", ""))
                if model_path.exists():
                    # Calculate size
                    size_bytes = sum(
                        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                    )
                    assets["base_model"] = {
                        "name": config.get("model_name", "unknown"),
                        "display_name": config.get("model_display_name", "Base Model"),
                        "path": str(model_path),
                        "size_gb": round(size_bytes / (1024**3), 2),
                        "locked": config.get("locked", {}),
                    }
                    assets["total_size_gb"] += assets["base_model"]["size_gb"]

            # Scan for checkpoints
            current_model_dir = BASE_DIR / "current_model"
            if current_model_dir.exists():
                for item in sorted(current_model_dir.iterdir(), reverse=True):
                    if item.is_dir() and item.name.startswith("checkpoint-"):
                        try:
                            step = int(item.name.split("-")[1])
                            # Get size
                            size_bytes = sum(
                                f.stat().st_size for f in item.rglob("*") if f.is_file()
                            )
                            size_gb = round(size_bytes / (1024**3), 2)
                            # Get modification time
                            mtime = datetime.fromtimestamp(item.stat().st_mtime)

                            # Get eval count from curriculum_state.json
                            eval_count = 0
                            try:
                                curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
                                if curriculum_file.exists():
                                    with open(curriculum_file) as cf:
                                        curriculum = json.load(cf)
                                    # Count evals at or near this step (within 500 steps)
                                    for skill_id, skill_data in curriculum.get("skills", {}).items():
                                        for eval_entry in skill_data.get("accuracy_history", []):
                                            eval_step = eval_entry.get("step", 0)
                                            if abs(eval_step - step) <= 500:
                                                eval_count += 1
                            except Exception:
                                pass

                            assets["checkpoints"].append({
                                "name": item.name,
                                "step": step,
                                "path": str(item),
                                "size_gb": size_gb,
                                "created": mtime.isoformat(),
                                "age_hours": round((datetime.now() - mtime).total_seconds() / 3600, 1),
                                "eval_count": eval_count,
                            })
                            assets["total_size_gb"] += size_gb
                        except (ValueError, OSError) as e:
                            logger.warning(f"Error processing {item}: {e}")

            # Round total
            assets["total_size_gb"] = round(assets["total_size_gb"], 2)
            assets["checkpoint_count"] = len(assets["checkpoints"])

            # Mark latest and find best from status file
            if assets["checkpoints"]:
                assets["checkpoints"][0]["is_latest"] = True

            # Mark best checkpoint from ledger
            try:
                from core.checkpoint_ledger import get_ledger
                ledger = get_ledger()
                best_record = ledger.get_best(metric="train_loss")
                if best_record:
                    for cp in assets["checkpoints"]:
                        if cp.get("step") == best_record.step:
                            cp["is_champion"] = True
                            break
            except Exception:
                pass

            self._send_json(assets)

        except Exception as e:
            logger.error(f"Vault assets error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_vault_zones(self, refresh: bool = False):
        """Serve zone status by querying Zone Wardens."""
        try:
            from vault.warden_client import get_zone_summary

            # Query all zone wardens (parallel, with timeout)
            zone_summary = get_zone_summary()

            # Map role to type for backward compatibility
            role_to_type = {
                "trainer": "central",
                "inference": "inference",
                "storage": "storage",
            }

            zones = []
            for zone in zone_summary:
                zone_data = {
                    "zone_id": zone["zone_id"],
                    "name": zone["name"],
                    "type": role_to_type.get(zone.get("role"), zone.get("role", "unknown")),
                    "host": zone["host"],
                    "port": zone["port"],
                    "status": zone["status"],
                    "reachable": zone.get("reachable", False),
                }

                # Add service summary if available
                if "services" in zone:
                    s = zone["services"]
                    zone_data["services_online"] = s.get("online", 0)
                    zone_data["services_total"] = s.get("total", 0)
                    zone_data["services_offline"] = s.get("offline", 0)

                # Add capabilities based on role
                zone_data["can_train"] = zone.get("role") == "trainer"
                zone_data["can_infer"] = zone.get("role") == "inference"

                # Add warden info
                if zone.get("warden_version"):
                    zone_data["warden_version"] = zone["warden_version"]
                if zone.get("last_patrol"):
                    zone_data["last_patrol"] = zone["last_patrol"]
                if zone.get("response_ms"):
                    zone_data["response_ms"] = zone["response_ms"]
                if zone.get("error"):
                    zone_data["error"] = zone["error"]

                zones.append(zone_data)

            self._send_json({
                "zones": zones,
                "count": len(zones),
                "online": sum(1 for z in zones if z["status"] == "online"),
                "reachable": sum(1 for z in zones if z.get("reachable", False)),
            })

        except ImportError as e:
            logger.warning(f"Warden client not available: {e}")
            # Return default zones without live status
            self._send_json({
                "zones": [
                    {"zone_id": "4090", "name": "Training Server", "status": "unknown", "type": "central"},
                    {"zone_id": "3090", "name": "Inference Server", "status": "unknown", "type": "inference"},
                    {"zone_id": "nas", "name": "Synology NAS", "status": "unknown", "type": "storage"},
                ],
                "count": 3,
                "online": 0,
                "note": "Warden client not available - start zone wardens for live status"
            })
        except Exception as e:
            logger.error(f"Zone status error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # LEDGER API - Checkpoint history and stats
    # =========================================================================

    def _serve_ledger_list(self, query: dict):
        """List all checkpoints with ledger stats, including base model."""
        try:
            from core.checkpoint_ledger import get_ledger
            from datetime import datetime

            ledger = get_ledger()
            limit = int(query.get("limit", [50])[0])
            skill = query.get("skill", [None])[0]
            include_base = query.get("include_base", ["true"])[0].lower() == "true"

            if skill:
                records = ledger.list_by_skill(skill)[:limit]
            else:
                records = ledger.list_all(limit=limit)

            checkpoints = []

            # Add base model as step 0 if requested
            if include_base and not skill:
                base_model_path = BASE_DIR / "models" / "Qwen3-0.6B"
                if base_model_path.exists():
                    # Calculate base model size
                    try:
                        size_bytes = sum(
                            f.stat().st_size for f in base_model_path.rglob("*") if f.is_file()
                        )
                        size_gb = round(size_bytes / (1024**3), 2)

                        # Get model config for more info
                        config_file = base_model_path / "config.json"
                        model_info = {}
                        if config_file.exists():
                            with open(config_file) as f:
                                model_info = json.load(f)

                        checkpoints.append({
                            "step": 0,
                            "canonical_name": "base-model",
                            "display_name": "Qwen3-0.6B (Base)",
                            "timestamp": datetime.fromtimestamp(base_model_path.stat().st_mtime).isoformat(),
                            "train_loss": None,  # Base model has no training loss
                            "val_loss": None,
                            "learning_rate": None,
                            "skill_name": None,
                            "skill_level": None,
                            "size_gb": size_gb,
                            "age_hours": None,
                            "path": str(base_model_path),
                            "is_base": True,
                            "model_type": model_info.get("model_type", "qwen2"),
                            "hidden_size": model_info.get("hidden_size"),
                            "num_layers": model_info.get("num_hidden_layers"),
                            "vocab_size": model_info.get("vocab_size"),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get base model info: {e}")

            for r in records:
                checkpoints.append({
                    "step": r.step,
                    "canonical_name": r.canonical_name,
                    "timestamp": r.timestamp,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "learning_rate": r.learning_rate,
                    "skill_name": r.skill_name,
                    "skill_level": r.skill_level,
                    "size_gb": r.size_gb,
                    "age_hours": round(r.age_hours, 1),
                    "path": r.path,
                    "is_base": False,
                })

            self._send_json({
                "checkpoints": checkpoints,
                "count": len(checkpoints),
            })

        except ImportError:
            # Ledger not available, return empty
            self._send_json({"checkpoints": [], "count": 0, "error": "Ledger not initialized"})
        except Exception as e:
            logger.error(f"Ledger list error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_ledger_summary(self):
        """Get ledger summary statistics."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            summary = ledger.get_summary()
            self._send_json(summary)

        except ImportError:
            self._send_json({"error": "Ledger not initialized"}, 500)
        except Exception as e:
            logger.error(f"Ledger summary error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_ledger_best(self, query: dict):
        """Get the best checkpoint by a metric."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            metric = query.get("metric", ["train_loss"])[0]
            lower_is_better = query.get("lower", ["true"])[0].lower() == "true"

            record = ledger.get_best(metric=metric, lower_is_better=lower_is_better)

            if record:
                self._send_json({
                    "step": record.step,
                    "canonical_name": record.canonical_name,
                    "metric": metric,
                    "value": getattr(record, metric, None),
                    "train_loss": record.train_loss,
                    "val_loss": record.val_loss,
                    "skill_name": record.skill_name,
                    "path": record.path,
                })
            else:
                self._send_json({"error": "No checkpoints with that metric"}, 404)

        except ImportError:
            self._send_json({"error": "Ledger not initialized"}, 500)
        except Exception as e:
            logger.error(f"Ledger best error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_ledger_checkpoint(self, step_str: str):
        """Get specific checkpoint info by step."""
        try:
            from core.checkpoint_ledger import get_ledger

            step = int(step_str)
            ledger = get_ledger()
            record = ledger.get(step)

            if record:
                self._send_json(record.to_dict())
            else:
                self._send_json({"error": f"Checkpoint {step} not found"}, 404)

        except ValueError:
            self._send_json({"error": f"Invalid step: {step_str}"}, 400)
        except ImportError:
            self._send_json({"error": "Ledger not initialized"}, 500)
        except Exception as e:
            logger.error(f"Ledger checkpoint error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_checkpoint_data(self, step_str: str):
        """
        Serve aggregated checkpoint data from multiple sources.

        Combines:
        - Ledger data (stats at save time)
        - Model comparison scores (if available)
        - Deployment history (if available)
        - Physical file info
        - Evaluation results (skill evals, passives)
        """
        try:
            step = int(step_str)

            response = {
                "step": step,
                "ledger": None,
                "comparison": None,
                "deployment": None,
                "physical": None,
                "evaluations": None,
                "found": False,
            }

            # Special handling for base model (step 0)
            if step == 0:
                base_model_path = BASE_DIR / "models" / "Qwen3-0.6B"
                if base_model_path.exists():
                    # Get model config
                    config_file = base_model_path / "config.json"
                    model_info = {}
                    if config_file.exists():
                        with open(config_file) as f:
                            model_info = json.load(f)

                    # Calculate size
                    total_size = sum(
                        f.stat().st_size for f in base_model_path.rglob("*") if f.is_file()
                    )

                    # File list
                    files = []
                    for f in base_model_path.iterdir():
                        if f.is_file():
                            size = f.stat().st_size
                            files.append({
                                "name": f.name,
                                "size_bytes": size,
                                "size_mb": round(size / (1024**2), 2),
                            })

                    response["ledger"] = {
                        "step": 0,
                        "canonical_name": "base-model",
                        "display_name": "Qwen3-0.6B (Base)",
                        "is_base": True,
                        "model_type": model_info.get("model_type", "qwen3"),
                        "hidden_size": model_info.get("hidden_size"),
                        "num_layers": model_info.get("num_hidden_layers"),
                        "vocab_size": model_info.get("vocab_size"),
                        "max_position_embeddings": model_info.get("max_position_embeddings"),
                        "train_loss": None,
                        "val_loss": None,
                    }

                    response["physical"] = {
                        "path": str(base_model_path),
                        "name": "Qwen3-0.6B",
                        "exists": True,
                        "total_size_gb": round(total_size / (1024**3), 2),
                        "file_count": len(files),
                        "files": sorted(files, key=lambda x: -x["size_bytes"]),
                        "has_optimizer": False,
                        "has_scheduler": False,
                        "modified": datetime.fromtimestamp(base_model_path.stat().st_mtime).isoformat(),
                    }

                    # Load base model eval results if any
                    response["evaluations"] = self._get_checkpoint_evals(0)

                    response["found"] = True
                    self._send_json(response)
                    return
                else:
                    self._send_json({"error": "Base model not found"}, 404)
                    return

            # 1. Ledger data (primary source)
            try:
                from core.checkpoint_ledger import get_ledger
                ledger = get_ledger()
                record = ledger.get(step)
                if record:
                    response["ledger"] = record.to_dict()
                    response["found"] = True
            except ImportError:
                pass

            # 2. Check if this is the best checkpoint (from ledger)
            try:
                best_record = ledger.get_best(metric="train_loss")
                if best_record and best_record.step == step:
                    response["is_best"] = True
            except Exception:
                pass

            # 3. Deployment history
            deployment_file = BASE_DIR / "status" / "deployment_status.json"
            if deployment_file.exists():
                try:
                    with open(deployment_file) as f:
                        deployments = json.load(f)

                    # Find deployments of this checkpoint
                    checkpoint_deployments = []
                    for dep in deployments if isinstance(deployments, list) else [deployments]:
                        dep_step = dep.get("checkpoint_step") or dep.get("step")
                        if dep_step == step:
                            checkpoint_deployments.append({
                                "host": dep.get("host", dep.get("target_host")),
                                "deployed_at": dep.get("deployed_at", dep.get("timestamp")),
                                "status": dep.get("status"),
                            })

                    if checkpoint_deployments:
                        response["deployment"] = {
                            "deployed": True,
                            "deployments": checkpoint_deployments,
                            "last_deployed": checkpoint_deployments[0].get("deployed_at"),
                        }
                except Exception as e:
                    logger.warning(f"Failed to load deployment status: {e}")

            # 4. Physical file info
            # Try to find the actual checkpoint directory
            checkpoints_dir = BASE_DIR / "current_model"
            if checkpoints_dir.exists():
                # Look for matching checkpoint
                for item in checkpoints_dir.iterdir():
                    if item.is_dir() and f"-{step}" in item.name:
                        try:
                            # Get file list
                            files = []
                            total_size = 0
                            for f in item.iterdir():
                                if f.is_file():
                                    size = f.stat().st_size
                                    files.append({
                                        "name": f.name,
                                        "size_bytes": size,
                                        "size_mb": round(size / (1024**2), 2),
                                    })
                                    total_size += size

                            response["physical"] = {
                                "path": str(item),
                                "name": item.name,
                                "exists": True,
                                "total_size_gb": round(total_size / (1024**3), 2),
                                "file_count": len(files),
                                "files": sorted(files, key=lambda x: -x["size_bytes"]),
                                "has_optimizer": any(f["name"] == "optimizer.pt" for f in files),
                                "has_scheduler": any(f["name"] == "scheduler.pt" for f in files),
                                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                            }
                            response["found"] = True
                        except Exception as e:
                            logger.warning(f"Failed to read checkpoint files: {e}")
                        break

            if not response["found"]:
                self._send_json({"error": f"Checkpoint {step} not found"}, 404)
                return

            # 5. Evaluation results (skill evals, passives)
            response["evaluations"] = self._get_checkpoint_evals(step)

            self._send_json(response)

        except ValueError:
            self._send_json({"error": f"Invalid step: {step_str}"}, 400)
        except Exception as e:
            logger.error(f"Checkpoint data error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _get_checkpoint_evals(self, step: int) -> dict:
        """
        Get evaluation results for a checkpoint from multiple sources.

        Returns:
            dict with skill_evals, passives, curriculum results
        """
        evals = {
            "skill_evals": [],
            "passives": [],
            "curriculum": None,
            "has_data": False,
        }

        try:
            # 1. Skill evaluations (from evaluation_ledger)
            try:
                from core.evaluation_ledger import get_eval_ledger
                eval_ledger = get_eval_ledger()
                skill_results = eval_ledger.get_by_checkpoint(step)
                if skill_results:
                    evals["skill_evals"] = [{
                        "skill": r.skill,
                        "level": r.level,
                        "accuracy": r.accuracy,
                        "correct": r.correct,
                        "total": r.total,
                        "evaluated_at": r.evaluated_at,
                    } for r in skill_results]
                    evals["has_data"] = True
            except (ImportError, AttributeError):
                pass

            # 2. Passive evaluations
            try:
                from core.passives import get_passives_ledger
                passives_ledger = get_passives_ledger()
                passive_results = passives_ledger.get_by_checkpoint(step)
                if passive_results:
                    evals["passives"] = [{
                        "passive_id": r.passive_id,
                        "mode": r.mode,
                        "accuracy": r.accuracy,
                        "correct": r.correct,
                        "total": r.total,
                        "evaluated_at": r.evaluated_at,
                        "version": r.version,
                    } for r in passive_results]
                    evals["has_data"] = True
            except (ImportError, AttributeError):
                pass

            # 3. Curriculum eval status from curriculum_state.json
            curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
            if curriculum_file.exists():
                try:
                    with open(curriculum_file) as f:
                        curr_data = json.load(f)
                    # Find evals near this step (within 500 steps)
                    curriculum_evals = []
                    for skill_id, skill_data in curr_data.get("skills", {}).items():
                        for eval_entry in skill_data.get("accuracy_history", []):
                            eval_step = eval_entry.get("step", 0)
                            if abs(eval_step - step) <= 500:
                                curriculum_evals.append({
                                    "skill": skill_id,
                                    "level": eval_entry.get("level") or eval_entry.get("metadata", {}).get("level", 1),
                                    "accuracy": eval_entry.get("accuracy", 0),
                                    "correct": eval_entry.get("metadata", {}).get("correct", 0),
                                    "total": eval_entry.get("metadata", {}).get("problems", 20),
                                    "step": eval_step,
                                    "evaluated_at": eval_entry.get("timestamp"),
                                })
                    if curriculum_evals:
                        evals["skill_evals"] = curriculum_evals
                        evals["has_data"] = True
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Failed to load evals for checkpoint {step}: {e}")

        return evals

    # =========================================================================
    # ORACLE API - Talk to DIO (inference interface)
    # =========================================================================

    # Host registry integration
    _host_registry = None
    _host_registry_loaded = False
    _inference_api_key = None
    _inference_api_key_loaded = False

    @classmethod
    def _get_inference_api_key(cls):
        """Get inference API key from environment or config."""
        if not cls._inference_api_key_loaded:
            import os
            # Try environment variable first
            key = os.environ.get("INFERENCE_ADMIN_KEY", "")
            if not key:
                # Try config file
                secrets_file = BASE_DIR / ".secrets" / "inference.json"
                if secrets_file.exists():
                    try:
                        with open(secrets_file) as f:
                            secrets = json.load(f)
                        key = secrets.get("admin_key", "")
                    except Exception as e:
                        logger.warning(f"Failed to load inference secrets: {e}")
            cls._inference_api_key = key
            cls._inference_api_key_loaded = True
            if key:
                logger.info("Inference API key loaded")
            else:
                logger.warning("No inference API key configured (set INFERENCE_ADMIN_KEY)")
        return cls._inference_api_key

    def _make_inference_request(self, url: str, method: str = "GET",
                                 data: Optional[bytes] = None, timeout: int = 30):
        """Make an authenticated request to the inference server."""
        headers = {}
        api_key = self._get_inference_api_key()
        if api_key:
            headers["X-API-Key"] = api_key
        if data:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout)

    @classmethod
    def _get_host_registry(cls):
        """Get host registry (lazy load)."""
        if not cls._host_registry_loaded:
            try:
                from core.hosts import get_registry
                cls._host_registry = get_registry()
                logger.info("Host registry loaded successfully")
            except ImportError as e:
                logger.warning(f"Host registry not available: {e}")
                cls._host_registry = None
            cls._host_registry_loaded = True
        return cls._host_registry

    @classmethod
    def _load_inference_hosts(cls):
        """Load inference hosts from host registry or fallback to config."""
        # Try host registry first
        registry = cls._get_host_registry()
        if registry:
            hosts = {}
            for host in registry.list_with_service("inference"):
                svc = host.services.get("inference")
                if svc:
                    hosts[host.host_id] = {
                        "name": host.name,
                        "host": host.host,
                        "port": svc.port,
                        "type": host.role.value,
                        "models_dir": host.models_dir,
                        "checkpoints_dir": host.checkpoints_dir,
                    }
            if hosts:
                return hosts

        # Fallback to config file
        hosts_file = BASE_DIR / "config" / "hosts.json"
        default_hosts = {
            "3090": {
                "name": "RTX 3090 (Inference)",
                "host": "192.168.x.x",
                "port": 8765,
                "type": "inference",
                "models_dir": "/path/to/models",
            },
        }

        if hosts_file.exists():
            try:
                with open(hosts_file) as f:
                    config = json.load(f)
                return config.get("inference_hosts", default_hosts)
            except Exception as e:
                logger.warning(f"Failed to load hosts.json: {e}")

        return default_hosts

    @property
    def INFERENCE_HOSTS(self):
        """Get inference hosts (cached after first load)."""
        if not hasattr(self.__class__, '_inference_hosts_cache'):
            self.__class__._inference_hosts_cache = self._load_inference_hosts()
        return self.__class__._inference_hosts_cache

    def _get_host_service_url(self, host_id: str, service: str) -> Optional[str]:
        """Get a service URL for a host using registry if available."""
        registry = self._get_host_registry()
        if registry:
            host = registry.get(host_id)
            if host:
                return host.get_service_url(service)
        # Fallback to INFERENCE_HOSTS
        if host_id in self.INFERENCE_HOSTS:
            host = self.INFERENCE_HOSTS[host_id]
            return f"http://{host['host']}:{host['port']}"
        return None

    def _serve_oracle_hosts(self):
        """List available inference hosts."""
        hosts = []
        for host_id, config in self.INFERENCE_HOSTS.items():
            host_info = {
                "id": host_id,
                "name": config["name"],
                "host": config["host"],
                "port": config["port"],
                "type": config["type"],
                "status": "unknown",
                "loaded_checkpoint": None,
            }

            # Check if host is online and get loaded model
            try:
                url = f"http://{config['host']}:{config['port']}/models/info"
                with self._make_inference_request(url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    host_info["status"] = "online"
                    # Extract loaded model info
                    if data.get("pool"):
                        for model_id, info in data["pool"].items():
                            if info.get("is_active"):
                                host_info["loaded_checkpoint"] = {
                                    "model_id": model_id,
                                    "step": info.get("checkpoint_step"),
                                    "loaded_at": info.get("loaded_at"),
                                }
                                break
            except Exception as e:
                host_info["status"] = "offline"
                host_info["error"] = str(e)

            hosts.append(host_info)

        self._send_json({"hosts": hosts})

    def _serve_oracle_status(self):
        """Get current oracle status - what's loaded where."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            latest = ledger.get_latest()
            best = ledger.get_best(metric="train_loss")

            status = {
                "latest_checkpoint": {
                    "step": latest.step,
                    "canonical_name": latest.canonical_name,
                    "train_loss": latest.train_loss,
                } if latest else None,
                "best_checkpoint": {
                    "step": best.step,
                    "canonical_name": best.canonical_name,
                    "train_loss": best.train_loss,
                } if best else None,
                "hosts": {},
            }

            # Check each host - use /models/info for ACCURATE loaded models
            for host_id, config in self.INFERENCE_HOSTS.items():
                try:
                    import re
                    # Get ACTUAL loaded models from /models/info (not stale /health)
                    models_url = f"http://{config['host']}:{config['port']}/models/info"
                    with self._make_inference_request(models_url, timeout=5) as response:
                        models_data = json.loads(response.read().decode())

                    loaded_models = []
                    for model in models_data.get("models", []):
                        model_id = model.get("model_id", "")
                        step = None
                        ledger_info = None

                        # Check if it's the base model
                        if model_id in ["Qwen3-0.6B-base", "Qwen3-0.6B", "base"]:
                            step = 0
                            ledger_info = {
                                "train_loss": None,
                                "skill_name": None,
                                "canonical_name": "Qwen3-0.6B (Base)",
                            }
                        else:
                            # Extract step from model_id
                            match = re.search(r'(?:checkpoint-?|step)(\d+)(k)?', model_id, re.IGNORECASE)
                            if match:
                                step = int(match.group(1))
                                if match.group(2):  # 'k' suffix means thousands
                                    step *= 1000
                                record = ledger.get(step)
                                if record:
                                    ledger_info = {
                                        "train_loss": record.train_loss,
                                        "skill_name": record.skill_name,
                                        "canonical_name": record.canonical_name,
                                    }

                        loaded_models.append({
                            "model_id": model_id,
                            "step": step,
                            "ledger_info": ledger_info,
                        })

                    status["hosts"][host_id] = {
                        "status": "online",
                        "loaded_models": loaded_models,
                        "model_count": len(loaded_models),
                    }
                except Exception as e:
                    status["hosts"][host_id] = {
                        "status": "offline",
                        "error": str(e),
                    }

            self._send_json(status)

        except Exception as e:
            logger.error(f"Oracle status error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # SKILLS API - Active skills and curriculum
    # =========================================================================

    def _serve_skills(self):
        """Get all skills with current level and progress from curriculum state."""
        try:
            import yaml
            from pathlib import Path

            skills_dir = BASE_DIR / "configs" / "skills"
            curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"

            # Load curriculum state
            curriculum_state = {}
            if curriculum_file.exists():
                with open(curriculum_file) as f:
                    curriculum_state = json.load(f)

            skills = []
            for yaml_file in skills_dir.glob("*.yaml"):
                if yaml_file.name.startswith("_"):
                    continue  # Skip template

                with open(yaml_file) as f:
                    config = yaml.safe_load(f)

                skill_id = config.get("id")
                # Map skill IDs (syllo -> sy in some places)
                state_key = skill_id
                if skill_id == "sy" and "sy" not in curriculum_state.get("skills", {}):
                    state_key = "syllo"
                elif skill_id == "bin" and "bin" not in curriculum_state.get("skills", {}):
                    state_key = "binary"

                skill_state = curriculum_state.get("skills", {}).get(state_key, {})
                accuracy_history = skill_state.get("accuracy_history", [])

                # Get latest accuracy
                latest_accuracy = 0.0
                if accuracy_history:
                    latest_accuracy = accuracy_history[-1].get("accuracy", 0.0)

                # current_level = mastered level (0 = hasn't mastered anything yet)
                # training_level = current_level + 1 (what they're working on)
                mastered_level = skill_state.get("current_level", 0)
                training_level = mastered_level + 1

                # Count evals at training level
                evals_at_level = sum(
                    1 for h in accuracy_history[-10:]  # Last 10 evals
                    if h.get("level") == training_level
                )

                # Calculate progress (3 passing evals at 80%+ = level up)
                passing_evals = sum(
                    1 for h in accuracy_history[-10:]
                    if h.get("level") == training_level and h.get("accuracy", 0) >= 0.8
                )
                progress_pct = min(100, int((passing_evals / 3) * 100))

                display = config.get("display", {})
                skills.append({
                    "id": skill_id,
                    "name": config.get("name", skill_id.upper()),
                    "short_name": display.get("short_name", skill_id.upper()),
                    "rpg_name": config.get("name", skill_id.upper()),
                    "description": config.get("description", ""),
                    "icon": display.get("icon", "⚔️"),
                    "color": display.get("color", "#7c3aed"),
                    "current_level": mastered_level,  # Mastered level (0 = none)
                    "mastered_level": mastered_level,  # Alias for frontend
                    "training_level": training_level,  # What they're working on
                    "max_level": config.get("levels", {}).get("max", 30),
                    "accuracy": latest_accuracy * 100,  # Convert to percentage
                    "evals_at_level": min(evals_at_level, 3),
                    "eval_count": len(accuracy_history),
                    "progress_pct": progress_pct,
                })

            # Calculate total mastered
            total_mastered = sum(s.get("mastered_level", 0) for s in skills)

            # Return in format frontend expects
            self._send_json({
                "skills": skills,
                "total_mastered": total_mastered
            })

        except Exception as e:
            logger.error(f"Skills API error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _serve_curriculum(self):
        """Get curriculum state including last eval time."""
        try:
            curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"

            if not curriculum_file.exists():
                self._send_json({"error": "Curriculum state not found"}, 404)
                return

            with open(curriculum_file) as f:
                state = json.load(f)

            # Find last eval time across all skills
            last_eval_time = None
            for skill_data in state.get("skills", {}).values():
                for h in skill_data.get("accuracy_history", []):
                    ts = h.get("timestamp")
                    if ts and (not last_eval_time or ts > last_eval_time):
                        last_eval_time = ts

            self._send_json({
                "skills": state.get("skills", {}),
                "last_eval_time": last_eval_time,
            })

        except Exception as e:
            logger.error(f"Curriculum API error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # PASSIVES API - Transfer learning evaluations
    # =========================================================================

    def _serve_passives_definitions(self):
        """Get all passive definitions from modular system."""
        try:
            from guild.passives import get_passive_configs

            configs = get_passive_configs()
            definitions = [{
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "version": c.version,
                "lite_count": c.lite_count,
                "full_count": c.full_count,
            } for c in configs]

            self._send_json(definitions)

        except ImportError:
            # Fall back to core.passives
            try:
                from core.passives import get_passive_definitions
                defs = get_passive_definitions()
                self._send_json([{
                    "id": d.id,
                    "name": d.name,
                    "category": d.category,
                    "description": d.description,
                    "version": "1.0.0",
                    "lite_count": d.lite_count,
                    "full_count": d.full_count,
                } for d in defs])
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        except Exception as e:
            logger.error(f"Passives definitions error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_passives_summary(self):
        """Get passives summary statistics."""
        try:
            from core.passives import get_passives_ledger

            ledger = get_passives_ledger()
            summary = ledger.summary()
            self._send_json(summary)

        except ImportError:
            self._send_json({
                "total_results": 0,
                "by_passive": {},
                "message": "Passives ledger not available"
            })

        except Exception as e:
            logger.error(f"Passives summary error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_oracle_load(self):
        """Load a checkpoint on an inference host."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            step = body.get("step")
            host_id = body.get("host", "3090")
            force_sync = body.get("sync", False)  # Force sync even if might exist

            if step is None:
                self._send_json({"success": False, "error": "Missing 'step'"}, 400)
                return

            if host_id not in self.INFERENCE_HOSTS:
                self._send_json({"success": False, "error": f"Unknown host: {host_id}"}, 400)
                return

            host_config = self.INFERENCE_HOSTS[host_id]
            models_dir = host_config.get("models_dir", "/path/to/models")

            # Special handling for base model (step 0)
            if int(step) == 0:
                checkpoint_name = "Qwen3-0.6B-base"
                local_path = BASE_DIR / "models" / "Qwen3-0.6B"
                remote_path = f"{models_dir}/Qwen3-0.6B-base"

                # Check if base model exists on remote
                if not self._check_remote_checkpoint(host_config, checkpoint_name):
                    # Sync base model
                    logger.info(f"Syncing base model to {host_id}...")
                    sync_result = self._sync_checkpoint_to_host(str(local_path), host_config, checkpoint_name)
                    if not sync_result["success"]:
                        self._send_json({
                            "success": False,
                            "error": f"Sync failed: {sync_result.get('error', 'Unknown error')}",
                        }, 500)
                        return

                # Load base model
                url = f"http://{host_config['host']}:{host_config['port']}/models/reload"
                req_data = json.dumps({"model_path": remote_path}).encode()

                with self._make_inference_request(url, method="POST", data=req_data, timeout=60) as response:
                    result = json.loads(response.read().decode())

                self._send_json({
                    "success": True,
                    "step": 0,
                    "host": host_id,
                    "checkpoint_name": "Qwen3-0.6B (Base)",
                    "remote_path": remote_path,
                    "result": result,
                })
                return

            # Get checkpoint path from ledger
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            record = ledger.get(int(step))

            if not record:
                self._send_json({"success": False, "error": f"Checkpoint {step} not in ledger"}, 404)
                return

            # Use simple checkpoint name (checkpoint-{step}), not canonical with date
            checkpoint_name = f"checkpoint-{step}"
            remote_path = f"{models_dir}/{checkpoint_name}"

            # Check if checkpoint exists on remote (or force sync requested)
            if force_sync or not self._check_remote_checkpoint(host_config, checkpoint_name):
                # Need to sync - check if local exists
                local_path = record.path
                if not Path(local_path).exists():
                    self._send_json({
                        "success": False,
                        "error": f"Checkpoint {step} not found locally at {local_path}",
                        "needs_sync": False,
                    }, 404)
                    return

                # Sync to remote
                logger.info(f"Syncing checkpoint-{step} to {host_id}...")
                sync_result = self._sync_checkpoint_to_host(local_path, host_config, checkpoint_name)

                if not sync_result["success"]:
                    self._send_json({
                        "success": False,
                        "error": f"Sync failed: {sync_result.get('error', 'Unknown error')}",
                        "sync_attempted": True,
                    }, 500)
                    return

                logger.info(f"Sync completed in {sync_result.get('duration', 0):.1f}s")

            # Request model load
            url = f"http://{host_config['host']}:{host_config['port']}/models/reload"
            req_data = json.dumps({"model_path": remote_path}).encode()

            with self._make_inference_request(url, method="POST", data=req_data, timeout=60) as response:
                result = json.loads(response.read().decode())

            self._send_json({
                "success": True,
                "step": step,
                "host": host_id,
                "checkpoint_name": checkpoint_name,
                "remote_path": remote_path,
                "result": result,
            })

        except urllib.error.URLError as e:
            self._send_json({"success": False, "error": f"Host unreachable: {e}"}, 502)
        except Exception as e:
            logger.error(f"Oracle load error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"success": False, "error": str(e)}, 500)

    def _check_remote_checkpoint(self, host_config: dict, checkpoint_name: str) -> bool:
        """Check if a checkpoint exists on the remote host."""
        try:
            # Use SSH to check if directory exists
            import subprocess
            host = host_config["host"]
            models_dir = host_config.get("models_dir", "/path/to/models")
            remote_path = f"{models_dir}/{checkpoint_name}"

            result = subprocess.run(
                ["ssh", host, f"test -d {remote_path} && echo exists"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "exists" in result.stdout
        except Exception as e:
            logger.warning(f"Failed to check remote checkpoint: {e}")
            return False  # Assume doesn't exist, will try to sync

    def _sync_checkpoint_to_host(self, local_path: str, host_config: dict, checkpoint_name: str) -> dict:
        """Sync a checkpoint to a remote host using rsync."""
        import subprocess
        import time

        host = host_config["host"]
        models_dir = host_config.get("models_dir", "/path/to/models")
        remote_target = f"{host}:{models_dir}/{checkpoint_name}/"

        cmd = [
            "rsync",
            "-avz",
            "--delete",
            "--checksum",
            str(local_path) + "/",  # Trailing slash = sync contents
            remote_target
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            duration = time.time() - start

            if result.returncode == 0:
                return {"success": True, "duration": duration}
            else:
                return {"success": False, "error": result.stderr, "duration": duration}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Sync timeout (5 min)", "duration": time.time() - start}
        except Exception as e:
            return {"success": False, "error": str(e), "duration": time.time() - start}

    def _handle_oracle_chat(self):
        """Chat with DIO via inference host."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            message = body.get("message")
            host_id = body.get("host", "3090")
            system_prompt = body.get("system_prompt", "You are DIO, a helpful AI assistant.")
            max_tokens = body.get("max_tokens", 512)

            if not message:
                self._send_json({"success": False, "error": "Missing 'message'"}, 400)
                return

            if host_id not in self.INFERENCE_HOSTS:
                self._send_json({"success": False, "error": f"Unknown host: {host_id}"}, 400)
                return

            host_config = self.INFERENCE_HOSTS[host_id]

            # Determine which model to use
            # Query the inference server for available models and match by step
            step = body.get("step")
            model_name = None

            # Get list of available models from inference server
            available_model_ids = set()
            try:
                models_url = f"http://{host_config['host']}:{host_config['port']}/models/info"
                with self._make_inference_request(models_url, timeout=5) as resp:
                    models_info = json.loads(resp.read().decode())
                    # models_info has format: {"loaded": true, "models": [{"model_id": "...", ...}, ...]}
                    for model in models_info.get("models", []):
                        available_model_ids.add(model.get("model_id", ""))
            except Exception as e:
                logger.warning(f"Could not get models list: {e}")

            if step is not None:
                step = int(step)
                if step == 0:
                    # Base model - try various naming patterns
                    base_patterns = ["Qwen3-0.6B-base", "base", "Qwen3-0.6B"]
                    for pattern in base_patterns:
                        if pattern in available_model_ids:
                            model_name = pattern
                            break
                else:
                    # Find model matching this step - try various patterns
                    step_patterns = [
                        f"checkpoint-{step}",          # canonical
                        f"Qwen3-0.6B-step{step//1000}k",  # legacy naming
                        f"step{step}",
                        f"checkpoint_{step}",
                    ]
                    for pattern in step_patterns:
                        if pattern in available_model_ids:
                            model_name = pattern
                            break

                    # If no exact match, FAIL - never silently use wrong model
                    if not model_name:
                        logger.error(f"STRICT: No model found for step {step}. Available: {list(available_model_ids)}")
                        self._send_json({
                            "success": False,
                            "error": f"Step {step} not loaded. Available: {sorted(available_model_ids)}",
                            "available_models": sorted(available_model_ids),
                            "requested_step": step,
                        }, 404)
                        return

            # NO FALLBACK - step is REQUIRED for chat
            if not model_name:
                self._send_json({
                    "success": False,
                    "error": "Step parameter is required. No fallback allowed.",
                }, 400)
                return

            logger.info(f"Oracle chat: step={step}, resolved model={model_name}")

            # Build chat completion request (OpenAI-compatible format)
            chat_request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            url = f"http://{host_config['host']}:{host_config['port']}/v1/chat/completions"
            req_data = json.dumps(chat_request).encode()

            with self._make_inference_request(url, method="POST", data=req_data, timeout=120) as response:
                result = json.loads(response.read().decode())

            # VERSION VERIFICATION: Confirm server used the model we requested
            actual_model = result.get("model", "")
            if model_name and actual_model and actual_model != model_name:
                logger.warning(f"Model mismatch! Requested '{model_name}' but server used '{actual_model}'")
                self._send_json({
                    "success": False,
                    "error": f"Model mismatch: requested '{model_name}' but server used '{actual_model}'",
                    "requested_model": model_name,
                    "actual_model": actual_model,
                }, 409)  # 409 Conflict
                return

            # Extract response
            assistant_message = ""
            if result.get("choices"):
                assistant_message = result["choices"][0].get("message", {}).get("content", "")

            # Include verified model info in response
            self._send_json({
                "success": True,
                "response": assistant_message,
                "host": host_id,
                "model": actual_model,  # Server-confirmed model
                "model_path": result.get("model_path", ""),  # Full path on server
                "usage": result.get("usage", {}),
            })

        except urllib.error.URLError as e:
            self._send_json({"success": False, "error": f"Host unreachable: {e}"}, 502)
        except Exception as e:
            logger.error(f"Oracle chat error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _save_config(self):
        """Save config.json with backup."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            new_config = json.loads(body.decode("utf-8"))

            config_path = BASE_DIR / "config.json"
            backup_path = BASE_DIR / "config.json.bak"

            # Create backup
            if config_path.exists():
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Config backup created: {backup_path}")

            # Validate - ensure critical fields aren't removed
            if "model_name" not in new_config:
                self._send_json({"success": False, "error": "Missing model_name"}, 400)
                return

            # Write new config
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)

            logger.info("Config saved successfully")
            self._send_json({"success": True, "message": "Config saved"})

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self._send_json({"success": False, "error": f"Invalid JSON: {e}"}, 400)
        except Exception as e:
            logger.error(f"Config save error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _save_default(self):
        """Save current config as default."""
        try:
            config_path = BASE_DIR / "config.json"
            default_path = BASE_DIR / "config.defaults.json"

            if not config_path.exists():
                self._send_json({"success": False, "error": "No config to save"}, 404)
                return

            import shutil
            shutil.copy2(config_path, default_path)
            logger.info(f"Config saved as default: {default_path}")
            self._send_json({"success": True, "message": "Saved as default"})

        except Exception as e:
            logger.error(f"Save default error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _restore_default(self):
        """Restore config from default."""
        try:
            config_path = BASE_DIR / "config.json"
            default_path = BASE_DIR / "config.defaults.json"
            backup_path = BASE_DIR / "config.json.bak"

            if not default_path.exists():
                self._send_json({"success": False, "error": "No defaults saved yet"}, 404)
                return

            # Backup current before restoring
            if config_path.exists():
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Current config backed up: {backup_path}")

            # Restore from default
            import shutil
            shutil.copy2(default_path, config_path)
            logger.info(f"Config restored from default")

            # Return the restored config
            with open(config_path) as f:
                config = json.load(f)

            self._send_json({"success": True, "message": "Restored from default", "config": config})

        except Exception as e:
            logger.error(f"Restore default error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)


def run_tavern(
    port: int = 8888,
    api_host: str = "localhost",
    api_port: int = 8081,
    log_to_file: bool = False,
):
    """Start the Tavern server with process management."""
    global logger
    logger = setup_logging(log_to_file=log_to_file)

    TavernHandler.API_HOST = api_host
    TavernHandler.API_PORT = api_port

    # Process management
    ProcessManager.write_pid()
    ProcessManager.register_cleanup()

    # Use ThreadingHTTPServer to handle SSE streams without blocking
    server = ThreadingHTTPServer(("0.0.0.0", port), TavernHandler)
    ProcessManager.setup_signals(server)

    logger.info("=" * 60)
    logger.info("  THE TAVERN IS OPEN FOR BUSINESS!")
    logger.info("=" * 60)
    logger.info(f"  Visit: http://localhost:{port}")
    logger.info(f"  API proxied from: http://{api_host}:{api_port}")
    logger.info(f"  PID file: {ProcessManager.PID_FILE}")
    logger.info("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass  # Signal handler will take care of it
    finally:
        logger.info("Tavern closing for the night...")
        server.shutdown()
        ProcessManager.remove_pid()


def main():
    parser = argparse.ArgumentParser(
        description="The Tavern - Game UI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 tavern/server.py                    # Default port 8888
    python3 tavern/server.py --port 9000        # Custom port
    python3 tavern/server.py --api-port 8081    # Custom API port
    python3 tavern/server.py --log-to-file      # Also log to logs/tavern.log

Process Management:
    - PID written to .pids/tavern.pid
    - Stop with: kill $(cat .pids/tavern.pid)
    - Or use SIGTERM/SIGINT for graceful shutdown
        """
    )
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--api-host", type=str, default="localhost", help="API host")
    parser.add_argument("--api-port", type=int, default=8081, help="API port")
    parser.add_argument("--log-to-file", action="store_true", help="Also log to logs/tavern.log")

    args = parser.parse_args()
    run_tavern(
        port=args.port,
        api_host=args.api_host,
        api_port=args.api_port,
        log_to_file=args.log_to_file,
    )


if __name__ == "__main__":
    main()
