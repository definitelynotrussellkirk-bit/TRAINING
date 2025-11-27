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
from http.server import HTTPServer, SimpleHTTPRequestHandler
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
        elif path == "/api/passives/definitions":
            self._serve_passives_definitions()
        elif path == "/api/passives/summary":
            self._serve_passives_summary()

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
                        response["training"] = json.load(f)
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

            # 5. Model comparison (best checkpoint)
            comparison_file = BASE_DIR / "status" / "model_comparisons.json"
            if comparison_file.exists():
                try:
                    with open(comparison_file) as f:
                        comparisons = json.load(f)
                    response["comparison"] = {
                        "best_checkpoint": comparisons.get("latest_summary", {}).get("best_checkpoint"),
                        "total_compared": comparisons.get("latest_summary", {}).get("total_compared", 0),
                    }
                except Exception as e:
                    logger.warning(f"Failed to read comparisons: {e}")

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
            # current_level = level being TRAINED on (not mastered)
            max_level = config.get("max_level", 30)
            training = skill_state.get("current_level", 1)
            training = min(training, max_level)
            mastered = max(0, training - 1)

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

                            assets["checkpoints"].append({
                                "name": item.name,
                                "step": step,
                                "path": str(item),
                                "size_gb": size_gb,
                                "created": mtime.isoformat(),
                                "age_hours": round((datetime.now() - mtime).total_seconds() / 3600, 1),
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

            # Try to get best checkpoint from model comparison status
            comparison_file = BASE_DIR / "status" / "model_comparisons.json"
            if comparison_file.exists():
                try:
                    with open(comparison_file) as f:
                        comparisons = json.load(f)
                    best = comparisons.get("latest_summary", {}).get("best_checkpoint", "")
                    for cp in assets["checkpoints"]:
                        if cp["name"] == best or f"checkpoint-{cp['step']}" == best:
                            cp["is_champion"] = True
                            break
                except Exception:
                    pass

            self._send_json(assets)

        except Exception as e:
            logger.error(f"Vault assets error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_vault_zones(self, refresh: bool = False):
        """Serve zone status from VaultKeeper API."""
        try:
            from vault.zones import get_zone_registry, ZoneClient, ZoneStatus

            registry = get_zone_registry()

            # Optionally refresh all statuses
            if refresh:
                registry.check_all_status()

            zones = []
            for zone in registry.list():
                zone_data = {
                    "zone_id": zone.zone_id,
                    "name": zone.name,
                    "type": zone.zone_type.value,
                    "host": zone.host,
                    "port": zone.port,
                    "status": zone.status.value,
                    "asset_count": zone.asset_count,
                    "total_size_gb": zone.total_size_gb,
                    "disk_free_gb": zone.disk_free_gb,
                    "can_train": zone.can_train,
                    "can_infer": zone.can_infer,
                    "last_checked": zone.last_checked.isoformat() if zone.last_checked else None,
                }

                # For non-central zones, try to get live status
                if zone.zone_type.value != "central":
                    client = ZoneClient(zone)
                    live = client.get_status()
                    if live:
                        zone_data["status"] = "online"
                        zone_data["live"] = live
                        zone_data["asset_count"] = live.get("catalog", {}).get("total_assets", 0)
                        zone_data["disk_free_gb"] = live.get("disk", {}).get("free_gb", 0)
                else:
                    zone_data["status"] = "online"  # Central is always online

                zones.append(zone_data)

            self._send_json({
                "zones": zones,
                "count": len(zones),
                "online": sum(1 for z in zones if z["status"] == "online"),
            })

        except ImportError as e:
            logger.warning(f"Zone module not available: {e}")
            # Return default zones without live status
            self._send_json({
                "zones": [
                    {"zone_id": "4090", "name": "Training Server", "status": "online", "type": "central"},
                    {"zone_id": "3090", "name": "Inference Server", "status": "unknown", "type": "inference"},
                    {"zone_id": "nas", "name": "Synology NAS", "status": "unknown", "type": "storage"},
                ],
                "count": 3,
                "online": 1,
            })
        except Exception as e:
            logger.error(f"Zone status error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # LEDGER API - Checkpoint history and stats
    # =========================================================================

    def _serve_ledger_list(self, query: dict):
        """List all checkpoints with ledger stats."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            limit = int(query.get("limit", [50])[0])
            skill = query.get("skill", [None])[0]

            if skill:
                records = ledger.list_by_skill(skill)[:limit]
            else:
                records = ledger.list_all(limit=limit)

            checkpoints = []
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
        """
        try:
            step = int(step_str)

            response = {
                "step": step,
                "ledger": None,
                "comparison": None,
                "deployment": None,
                "physical": None,
                "found": False,
            }

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

            # 2. Model comparison scores
            comparison_file = BASE_DIR / "status" / "model_comparisons.json"
            if comparison_file.exists():
                try:
                    with open(comparison_file) as f:
                        comparisons = json.load(f)

                    # Find this checkpoint in comparisons
                    for comp in comparisons.get("comparisons", []):
                        if comp.get("checkpoint_step") == step:
                            response["comparison"] = {
                                "composite_score": comp.get("composite_score"),
                                "ranking": comp.get("ranking"),
                                "metrics": comp.get("metrics", {}),
                                "is_best": comp.get("is_best", False),
                            }
                            break

                    # Check if it's the current best
                    best = comparisons.get("latest_summary", {}).get("best_checkpoint", "")
                    if f"checkpoint-{step}" in best or f"-{step}-" in best:
                        if response["comparison"]:
                            response["comparison"]["is_best"] = True
                        else:
                            response["comparison"] = {"is_best": True}
                except Exception as e:
                    logger.warning(f"Failed to load model comparisons: {e}")

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

            self._send_json(response)

        except ValueError:
            self._send_json({"error": f"Invalid step: {step_str}"}, 400)
        except Exception as e:
            logger.error(f"Checkpoint data error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # ORACLE API - Talk to DIO (inference interface)
    # =========================================================================

    # Host registry integration
    _host_registry = None
    _host_registry_loaded = False

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
                with urllib.request.urlopen(url, timeout=5) as response:
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

            # Check each host
            for host_id, config in self.INFERENCE_HOSTS.items():
                try:
                    url = f"http://{config['host']}:{config['port']}/models/info"
                    with urllib.request.urlopen(url, timeout=5) as response:
                        data = json.loads(response.read().decode())
                        loaded_step = None
                        for model_id, info in data.get("pool", {}).items():
                            if info.get("is_active"):
                                loaded_step = info.get("checkpoint_step")
                                break

                        # Get ledger info for loaded checkpoint
                        ledger_info = None
                        if loaded_step:
                            record = ledger.get(loaded_step)
                            if record:
                                ledger_info = {
                                    "train_loss": record.train_loss,
                                    "skill_name": record.skill_name,
                                    "canonical_name": record.canonical_name,
                                }

                        status["hosts"][host_id] = {
                            "status": "online",
                            "loaded_step": loaded_step,
                            "ledger_info": ledger_info,
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

            if not step:
                self._send_json({"success": False, "error": "Missing 'step'"}, 400)
                return

            if host_id not in self.INFERENCE_HOSTS:
                self._send_json({"success": False, "error": f"Unknown host: {host_id}"}, 400)
                return

            # Get checkpoint path from ledger
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            record = ledger.get(int(step))

            if not record:
                self._send_json({"success": False, "error": f"Checkpoint {step} not in ledger"}, 404)
                return

            # Determine path on remote host using host config
            host_config = self.INFERENCE_HOSTS[host_id]
            models_dir = host_config.get("models_dir", "/path/to/models")
            remote_path = f"{models_dir}/{record.canonical_name}"

            # Request model load
            url = f"http://{host_config['host']}:{host_config['port']}/models/reload"
            req_data = json.dumps({"model_path": remote_path}).encode()
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())

            self._send_json({
                "success": True,
                "step": step,
                "host": host_id,
                "canonical_name": record.canonical_name,
                "remote_path": remote_path,
                "result": result,
            })

        except urllib.error.URLError as e:
            self._send_json({"success": False, "error": f"Host unreachable: {e}"}, 502)
        except Exception as e:
            logger.error(f"Oracle load error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

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

            # Build chat completion request (OpenAI-compatible format)
            chat_request = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            url = f"http://{host_config['host']}:{host_config['port']}/v1/chat/completions"
            req_data = json.dumps(chat_request).encode()
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())

            # Extract response
            assistant_message = ""
            if result.get("choices"):
                assistant_message = result["choices"][0].get("message", {}).get("content", "")

            self._send_json({
                "success": True,
                "response": assistant_message,
                "host": host_id,
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

    server = HTTPServer(("0.0.0.0", port), TavernHandler)
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
