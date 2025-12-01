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

# Import API modules (extracted from this file)
from tavern.api import heroes as heroes_api
from tavern.api import analysis as analysis_api
from tavern.api import skills as skills_api
from tavern.api import vault as vault_api
from tavern.api import jobs as jobs_api
from tavern.api import momentum as momentum_api
from tavern.api import train as train_api
from tavern.api import generate as generate_api
from tavern.api import setup as setup_api
from tavern.api import run_context as run_context_api
from tavern.api import temple as temple_api

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


def get_active_hero_id() -> str:
    """
    Get the active hero_id from the current campaign.

    Returns the hero_id from control/active_campaign.json, or None if no campaign is active.
    This replaces hardcoded "dio-qwen3-0.6b" defaults throughout the codebase.
    """
    try:
        campaign_file = BASE_DIR / "control" / "active_campaign.json"
        if campaign_file.exists():
            with open(campaign_file) as f:
                data = json.load(f)
                return data.get("hero_id")
    except Exception as e:
        logger.warning(f"Failed to get active hero_id: {e}")
    return None


# Skill loader - uses SkillEngine
def get_skills_data():
    """Load skill data from SkillEngine (single source of truth)."""
    try:
        from guild.skills import get_engine
        from guild.skills.loader import load_skill_config

        engine = get_engine()
        skills = []

        for skill_id in engine.list_skills():
            try:
                config = load_skill_config(skill_id)
                state = engine.get_state(skill_id)

                # current_level = level being TRAINED on (not mastered)
                # mastered = training - 1 (minimum 0)
                training = state.level
                training = min(training, config.max_level)
                mastered = max(0, training - 1)

                # Get accuracy from state
                recent_acc = 0
                if state.last_eval_accuracy is not None:
                    recent_acc = state.last_eval_accuracy * 100

                skills.append({
                    "id": config.id,
                    "name": config.name,
                    "rpg_name": config.rpg_name or config.name,
                    "rpg_description": config.rpg_description or config.description or "",
                    "icon": config.display.icon if config.display else "⚔️",
                    "color": config.display.color if config.display else "#888",
                    "short_name": config.display.short_name if config.display else config.id.upper(),
                    "max_level": config.max_level,
                    "mastered_level": mastered,
                    "training_level": training,
                    "accuracy": round(recent_acc, 1),
                    "eval_count": state.total_evals,
                    "category": config.category.value if hasattr(config.category, 'value') else str(config.category),
                    "description": config.description or "",
                })
            except Exception as e:
                logger.warning(f"Failed to load skill {skill_id}: {e}")
                continue

        # Sort by name
        skills.sort(key=lambda s: s["name"])

        return {
            "skills": skills,
            "active_skill": "",  # TODO: Get from training status
            "total_mastered": sum(s["mastered_level"] for s in skills),
        }

    except Exception as e:
        logger.error(f"Failed to load skills from engine: {e}")
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
            skills_api.serve_skills(self)

        elif path.startswith("/skill/"):
            skill_id = path.replace("/skill/", "").strip("/")
            if skill_id:
                self._serve_template("skill.html")  # Page template
            else:
                self._send_error(400, "Missing skill ID")

        elif path.startswith("/skill-data/"):
            skill_id = path.replace("/skill-data/", "").strip("/")
            if skill_id:
                skills_api.serve_skill_data(self, skill_id)
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

        # Ledger API - checkpoint stats and history (uses vault_api)
        elif path == "/ledger":
            vault_api.serve_ledger_list(self, query)
        elif path == "/ledger/summary":
            vault_api.serve_ledger_summary(self)
        elif path == "/ledger/best":
            vault_api.serve_ledger_best(self, query)
        elif path.startswith("/ledger/"):
            step_str = path.replace("/ledger/", "")
            vault_api.serve_ledger_checkpoint(self, step_str)

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
                vault_api.serve_checkpoint_data(self, step_str)
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
        elif path == "/api/hero":
            heroes_api.serve_hero_info(self)
        elif path.startswith("/api/hero-config/"):
            hero_id = path.replace("/api/hero-config/", "")
            heroes_api.serve_hero_config(self, hero_id)
        elif path == "/api/hero-model-info":
            heroes_api.serve_hero_model_info(self)
        elif path == "/api/skills":
            skills_api.serve_skills(self)
        elif path == "/api/titles":
            skills_api.serve_titles(self)
        elif path == "/api/curriculum":
            skills_api.serve_curriculum(self)
        elif path == "/api/passives/definitions":
            self._serve_passives_definitions()
        elif path == "/api/passives/summary":
            self._serve_passives_summary()

        # Skill Engine - Primitives API (uses skills_api)
        elif path == "/api/engine/health":
            skills_api.serve_engine_health(self)
        elif path == "/api/engine/primitives":
            self._serve_all_primitives()
        elif path == "/api/engine/primitive-summary":
            skills_api.serve_primitive_summary(self)
        elif path.startswith("/api/engine/skill/") and path.endswith("/primitives"):
            skill_id = path.replace("/api/engine/skill/", "").replace("/primitives", "")
            skills_api.serve_skill_primitives(self, skill_id)
        elif path.startswith("/api/engine/skill/") and path.endswith("/state"):
            skill_id = path.replace("/api/engine/skill/", "").replace("/state", "")
            skills_api.serve_skill_state(self, skill_id)

        # Evals - Evaluation ledger page and API
        elif path == "/evals" or path == "/evals.html":
            self._serve_template("evals.html")
        elif path == "/api/evals":
            self._serve_evals_list(query)
        elif path == "/api/evals/summary":
            self._serve_evals_summary()
        elif path.startswith("/api/evals/checkpoint/"):
            checkpoint_step = path.replace("/api/evals/checkpoint/", "")
            self._serve_evals_by_checkpoint(checkpoint_step, query)
        elif path.startswith("/api/evals/skill/"):
            skill_id = path.replace("/api/evals/skill/", "")
            self._serve_evals_by_skill(skill_id, query)
        elif path.startswith("/api/evals/job/"):
            job_id = path.replace("/api/evals/job/", "")
            self._serve_eval_by_job(job_id)

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

        # Weaver (orchestrator) status and control
        elif path == "/api/weaver/status":
            self._serve_weaver_status()
        elif path == "/api/weaver/start":
            self._weaver_start()
        elif path == "/api/weaver/stop":
            self._weaver_stop()

        # Task Master (background task scheduler) status
        elif path == "/api/task-master":
            self._serve_task_master_status()

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

        # Battle Log - MMO-style event stream
        elif path == "/battle-log" or path == "/battle_log" or path == "/battlelog":
            self._serve_template("battle_log.html")

        # Jobs API - Distributed job execution (uses jobs_api)
        elif path == "/jobs" or path == "/jobs.html":
            self._serve_template("jobs.html")
        elif path == "/api/jobs":
            jobs_api.serve_jobs_list(self, query)
        elif path == "/api/jobs/stats":
            jobs_api.serve_jobs_stats(self)
        elif path == "/api/jobs/warnings":
            jobs_api.serve_jobs_warnings(self, query)
        elif path.startswith("/api/jobs/"):
            job_id = path.replace("/api/jobs/", "")
            jobs_api.serve_job(self, job_id)

        # Cluster Dashboard - Heterogeneous cluster status
        elif path == "/cluster" or path == "/cluster.html":
            self._serve_template("cluster.html")

        # Forge - Data validation and queue health
        elif path == "/forge" or path == "/forge.html":
            self._serve_template("forge.html")
        elif path == "/api/forge/status":
            self._serve_forge_status()
        elif path == "/api/forge/rejected":
            self._serve_forge_rejected()

        # Temple - Diagnostic rituals
        elif path == "/temple" or path == "/temple.html":
            self._serve_template("temple.html")
        elif path == "/api/temple/rituals":
            temple_api.serve_rituals(self)

        # Campaign - Hero/Campaign management
        elif path == "/campaign" or path == "/campaign.html":
            self._serve_template("campaign.html")
        elif path == "/graph-test" or path == "/graph-test.html":
            self._serve_template("graph_test.html")
        elif path == "/api/campaigns":
            heroes_api.serve_campaigns_data(self)
        elif path == "/api/campaigns/active":
            heroes_api.serve_active_campaign(self)
        elif path == "/api/heroes":
            heroes_api.serve_heroes_data(self)

        # Momentum Engine - Forward progress + blockers
        elif path == "/api/momentum":
            momentum_api.serve_momentum(self)

        # Training API - Start training from UI
        elif path == "/api/train":
            if self.command == "POST":
                train_api.serve_train_post(self)
            else:
                train_api.serve_train_status(self)

        # Generate API - Generate training data from UI
        elif path == "/api/generate":
            if self.command == "POST":
                generate_api.serve_generate_post(self)
            else:
                generate_api.serve_generate_status(self)

        # Setup API - First-run detection and onboarding
        elif path == "/api/setup/status":
            setup_api.serve_setup_status(self)

        # Run Context API - Single source of truth for current training run
        elif path == "/api/run-context":
            run_context_api.serve_run_context(self)

        # World State API - Single authoritative snapshot of the Realm
        elif path == "/api/world-state":
            self._serve_world_state()
        elif path == "/api/realm" or path == "/api/realm-state":
            self._serve_realm_state()
        elif path == "/api/realm-mode":
            self._serve_realm_mode()

        # Cluster State API - Host registry and cluster management
        elif path == "/api/cluster":
            self._serve_cluster_state()
        elif path == "/api/cluster/summary":
            self._serve_cluster_summary()
        elif path.startswith("/api/cluster/host/"):
            host_id = path.replace("/api/cluster/host/", "").strip("/")
            self._serve_cluster_host(host_id)

        elif path == "/api/battle-log":
            self._serve_battle_log(query)
        elif path == "/api/battle_log":
            self._serve_battle_log_v2(query)

        # Analysis - Model Archaeology (uses analysis_api module)
        elif path == "/analysis" or path == "/analysis.html":
            self._serve_template("analysis.html")
        elif path.startswith("/api/analysis/") and "/layer_stats/" in path:
            # /api/analysis/{campaign_id}/layer_stats/{checkpoint}
            parts = path.replace("/api/analysis/", "").split("/layer_stats/")
            if len(parts) == 2:
                campaign_id, checkpoint_name = parts
                analysis_api.serve_layer_stats_detail(self, campaign_id, checkpoint_name, query)
            else:
                self._send_error(400, "Invalid path")
        elif path.startswith("/api/analysis/") and path.endswith("/layer_stats"):
            # /api/analysis/{campaign_id}/layer_stats
            campaign_id = path.replace("/api/analysis/", "").replace("/layer_stats", "")
            analysis_api.serve_layer_stats_list(self, campaign_id, query)
        elif path.startswith("/api/analysis/") and path.endswith("/drift_timeline"):
            # /api/analysis/{campaign_id}/drift_timeline
            campaign_id = path.replace("/api/analysis/", "").replace("/drift_timeline", "")
            analysis_api.serve_drift_timeline(self, campaign_id, query)
        elif path.startswith("/api/analysis/") and path.endswith("/top_movers"):
            # /api/analysis/{campaign_id}/top_movers
            campaign_id = path.replace("/api/analysis/", "").replace("/top_movers", "")
            analysis_api.serve_top_movers(self, campaign_id, query)

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
        # Campaign - Create, activate, archive
        elif path == "/api/campaigns":
            self._create_campaign()
        elif path == "/api/campaigns/activate":
            self._activate_campaign()
        elif path == "/api/campaigns/archive":
            self._archive_campaign()
        # Vault - Delete checkpoints
        elif path == "/api/vault/delete":
            self._delete_checkpoints()
        # Jobs - Submit jobs
        elif path == "/api/jobs":
            self._submit_job()
        elif path == "/api/jobs/eval":
            self._submit_eval_job()
        elif path.endswith("/cancel"):
            job_id = path.replace("/api/jobs/", "").replace("/cancel", "")
            self._cancel_job(job_id)
        # Training API - Start training from UI
        elif path == "/api/train":
            train_api.serve_train_post(self)
        # Generate API - Generate training data from UI
        elif path == "/api/generate":
            generate_api.serve_generate_post(self)
        # Setup API - Quick start for first-run
        elif path == "/api/setup/quick-start":
            setup_api.serve_quick_start(self)
        # Temple API - Run diagnostic rituals
        elif path == "/api/temple/run":
            temple_api.serve_run_ritual(self)
        # Reset API - Clear stale state
        elif path == "/api/reset":
            self._handle_reset()
        # Realm Mode API - Set global mode (TRAINING/IDLE/etc)
        elif path == "/api/realm-mode":
            self._set_realm_mode()
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
    # World State API - Single Authoritative Snapshot
    # =========================================

    def _serve_world_state(self):
        """Serve complete world state snapshot."""
        try:
            from core.world_state import get_world_state
            state = get_world_state()
            self._send_json(state)
        except ImportError:
            self._send_json({
                "error": "World state system not available",
                "realm_mode": "unknown",
                "health": "unknown",
            }, 503)
        except Exception as e:
            logger.error(f"World state error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # CLUSTER STATE API
    # =========================================================================

    def _serve_cluster_state(self):
        """Serve full cluster state (all hosts)."""
        try:
            from core.cluster_state import get_cluster_state
            cluster = get_cluster_state()
            self._send_json(cluster.to_dict())
        except ImportError:
            self._send_json({
                "error": "Cluster state system not available",
                "hosts": {},
            }, 503)
        except Exception as e:
            logger.error(f"Cluster state error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_cluster_summary(self):
        """Serve cluster summary (counts and overview)."""
        try:
            from core.cluster_state import get_cluster_summary
            summary = get_cluster_summary()
            self._send_json(summary)
        except ImportError:
            self._send_json({
                "error": "Cluster state system not available",
                "total_hosts": 0,
            }, 503)
        except Exception as e:
            logger.error(f"Cluster summary error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_cluster_host(self, host_id: str):
        """Serve details for a specific host."""
        try:
            from core.cluster_state import get_host
            host = get_host(host_id)
            if host:
                self._send_json(host.to_dict())
            else:
                self._send_json({"error": f"Host not found: {host_id}"}, 404)
        except ImportError:
            self._send_json({"error": "Cluster state system not available"}, 503)
        except Exception as e:
            logger.error(f"Cluster host error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_realm_state(self):
        """
        Serve unified realm state from RealmState service.

        This is the NEW single source of truth API. Combines:
        - Training state (step, loss, speed, etc.)
        - Queue state (depth, priorities)
        - Worker states (heartbeats)
        - Hero state (level, xp)
        - Recent events (battle log)
        - Mode state

        Reads from RealmState HTTP service (port 8866) instead of file.
        """
        try:
            from core.realm_store import get_realm_state
            data = get_realm_state()

            if data and "state" in data:
                # Return service data directly
                self._send_json(data)
            else:
                # Fall back to legacy sources if service unavailable
                logger.warning("RealmState service returned no data, using legacy fallback")
                self._serve_realm_state_legacy()

        except Exception as e:
            logger.error(f"Realm state error: {e}")
            # Fall back to legacy
            self._serve_realm_state_legacy()

    def _serve_realm_state_legacy(self):
        """Serve realm state from legacy sources (training_status.json, etc.)."""
        try:
            from core.world_state import get_world_state
            state = get_world_state()
            self._send_json(state)
        except Exception as e:
            logger.error(f"Legacy realm state error: {e}")
            self._send_json({
                "error": "Realm store not available",
                "state": {},
                "events": [],
            }, 503)

    def _populate_store_from_legacy(self, store):
        """Populate realm store from legacy sources during transition."""
        try:
            # Pull from training_status.json
            status_file = BASE_DIR / "status" / "training_status.json"
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)
                store.update_training(
                    status=status.get("status", "idle"),
                    step=status.get("current_step", 0),
                    total_steps=status.get("total_steps", 0),
                    loss=status.get("loss"),
                    learning_rate=status.get("learning_rate"),
                    file=status.get("current_file"),
                )

            # Pull from queue
            try:
                from core.training_queue import TrainingQueue
                queue = TrainingQueue(str(BASE_DIR / "queue"))
                depth, breakdown = queue.get_depth_with_breakdown()
                store.update_queue(
                    depth=depth,
                    high_priority=breakdown.get("high", 0),
                    normal_priority=breakdown.get("normal", 0),
                    low_priority=breakdown.get("low", 0),
                    status="ok" if depth > 5 else ("low" if depth > 0 else "empty"),
                )
            except Exception:
                pass

            # Pull from heartbeats
            heartbeat_dir = BASE_DIR / "status" / "heartbeats"
            if heartbeat_dir.exists():
                for hb_file in heartbeat_dir.glob("*.json"):
                    try:
                        with open(hb_file) as f:
                            hb = json.load(f)
                        store.update_worker(
                            worker_id=hb.get("worker_id", hb_file.stem),
                            role=hb.get("role", "unknown"),
                            status=hb.get("status", "unknown"),
                            device=hb.get("device"),
                            current_job=hb.get("current_job"),
                        )
                    except Exception:
                        pass

            # Pull from battle_log
            try:
                from core.battle_log import get_battle_logger
                bl = get_battle_logger()
                events = bl.get_events(limit=30)
                for e in events:
                    store.emit_event(
                        kind=e.source or "legacy",
                        message=e.message,
                        channel=e.channel,
                        severity=e.severity,
                        details=e.details,
                    )
            except Exception:
                pass

            store.flush()

        except Exception as e:
            logger.error(f"Failed to populate store from legacy: {e}")

    def _serve_realm_mode(self):
        """Serve current realm mode."""
        try:
            from core.realm_state import get_realm_state, RealmMode
            state = get_realm_state()
            self._send_json({
                "mode": state.mode.value,
                "description": state.mode.description,
                "changed_at": state.changed_at,
                "changed_by": state.changed_by,
                "reason": state.reason,
                "allows_training": state.mode.allows_training,
                "allows_evals": state.mode.allows_evals,
                "available_modes": [m.value for m in RealmMode],
            })
        except ImportError:
            self._send_json({
                "mode": "training",
                "description": "Realm state system not available",
                "error": "System not installed",
            }, 503)
        except Exception as e:
            logger.error(f"Realm mode error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _set_realm_mode(self):
        """Set realm mode via POST."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            mode_str = data.get("mode")
            reason = data.get("reason", "set via tavern")

            if not mode_str:
                self._send_json({"error": "Missing 'mode' field"}, 400)
                return

            from core.realm_state import set_realm_mode, RealmMode

            try:
                mode = RealmMode(mode_str)
            except ValueError:
                self._send_json({
                    "error": f"Invalid mode: {mode_str}",
                    "valid_modes": [m.value for m in RealmMode],
                }, 400)
                return

            new_state = set_realm_mode(mode, changed_by="tavern", reason=reason)
            self._send_json({
                "success": True,
                "mode": new_state.mode.value,
                "description": new_state.mode.description,
                "changed_at": new_state.changed_at,
            })

        except ImportError:
            self._send_json({"error": "Realm state system not available"}, 503)
        except Exception as e:
            logger.error(f"Set realm mode error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_battle_log(self, query: dict):
        """Serve formatted battle log entries."""
        try:
            from core.events import get_battle_log
            limit = int(query.get("limit", [50])[0])
            since_minutes = query.get("since_minutes", [None])[0]
            if since_minutes:
                since_minutes = int(since_minutes)

            log = get_battle_log(limit=limit, since_minutes=since_minutes)
            self._send_json({
                "entries": log,
                "count": len(log),
            })
        except ImportError:
            self._send_json({
                "entries": [],
                "error": "Battle log system not available",
            }, 503)
        except Exception as e:
            logger.error(f"Battle log error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_battle_log_v2(self, query: dict):
        """Serve battle log with full event objects (for battle_log.html UI)."""
        try:
            from core.battle_log import get_battle_logger, CHANNEL_ICONS

            logger_inst = get_battle_logger()
            limit = int(query.get("limit", [100])[0])
            since = query.get("since", [None])[0]

            # Get events
            events = logger_inst.get_events(limit=limit, since=since)

            # Format for UI
            formatted_events = []
            channel_counts = {}

            for e in events:
                channel = e.channel
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

                formatted_events.append({
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "channel": channel,
                    "severity": e.severity,
                    "message": e.message,
                    "source": e.source,
                    "icon": CHANNEL_ICONS.get(channel, "📢"),
                    "details": e.details,
                })

            # Get next_since for pagination
            next_since = None
            if formatted_events:
                next_since = formatted_events[0]["timestamp"]

            self._send_json({
                "events": formatted_events,
                "channel_counts": channel_counts,
                "next_since": next_since,
                "total": len(formatted_events),
            })

        except ImportError as e:
            logger.warning(f"Battle log not available: {e}")
            self._send_json({
                "events": [],
                "channel_counts": {},
                "next_since": None,
                "total": 0,
                "error": "Battle log system not available",
            })
        except Exception as e:
            logger.error(f"Battle log v2 error: {e}")
            self._send_json({"error": str(e)}, 500)

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

            # 3. Curriculum state (campaign-specific if available)
            curriculum_file = None
            try:
                from core.hero import get_active_campaign
                campaign = get_active_campaign()
                if campaign.get("campaign_path"):
                    campaign_curriculum = BASE_DIR / campaign["campaign_path"] / "status" / "curriculum_state.json"
                    if campaign_curriculum.exists():
                        curriculum_file = campaign_curriculum
            except Exception:
                pass

            # Fallback to global curriculum
            if curriculum_file is None:
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

            raw_problems = validation.get(level_key, [])

            # Transform messages format to prompt/expected format for frontend
            problems = []
            for prob in raw_problems:
                messages = prob.get("messages", [])
                prompt = ""
                expected = ""

                # Extract user message (prompt) and assistant message (expected answer)
                for msg in messages:
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        expected = msg.get("content", "")

                problems.append({
                    "prompt": prompt,
                    "expected": expected,
                    "metadata": prob.get("metadata", {})
                })

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
        Serve eval results for a skill from EvaluationLedger (single source of truth).
        """
        try:
            from core.evaluation_ledger import get_eval_ledger

            # Normalize skill IDs - ledger uses short names (bin, sy)
            # but may also have legacy records with long names (binary, syllo)
            id_mapping = {
                "binary": "bin",
                "syllo": "sy",
                # Short names pass through as-is
                "bin": "bin",
                "sy": "sy",
            }
            skill_name = id_mapping.get(skill_id, skill_id)

            response = {
                "skill": skill_id,
                "skill_name": skill_name,
                "level": int(level) if level else None,
                "latest": None,
                "history": [],
            }

            # Load from EvaluationLedger
            ledger = get_eval_ledger(BASE_DIR)

            # Force reload to get fresh data (fixes stale eval results)
            ledger.reload()

            # Alternate name for fallback lookups
            alt_names = {"bin": "binary", "sy": "syllo", "binary": "bin", "syllo": "sy"}
            alt_name = alt_names.get(skill_name)

            # Get latest eval for this skill (try both short and long names)
            latest_record = ledger.get_latest(skill_name)
            if not latest_record and alt_name:
                latest_record = ledger.get_latest(alt_name)

            if latest_record:
                # Filter by level if specified
                if level is None or latest_record.level == int(level):
                    # Convert problems to expected format
                    results = []
                    for prob in latest_record.problems:
                        results.append({
                            "problem_id": prob.get("problem_id") or f"problem-{prob.get('problem_idx', 0)}",
                            "correct": prob.get("correct", False),
                            "partial_score": prob.get("partial_score", 0.0),
                            "expected": prob.get("expected", ""),
                            "model_answer": prob.get("got", prob.get("model_answer", "")),
                        })

                    response["latest"] = {
                        "level": latest_record.level,
                        "level_name": None,  # TODO: Fetch from skill config if needed
                        "accuracy": latest_record.accuracy,
                        "correct": latest_record.correct,
                        "total": latest_record.total,
                        "timestamp": latest_record.timestamp,
                        "step": latest_record.checkpoint_step,
                        "results": results,
                    }

            # Get history (last 10 evals for this skill) - try both names
            if level:
                history_records = ledger.get_by_skill(skill_name, level=int(level))
                if not history_records and alt_name:
                    history_records = ledger.get_by_skill(alt_name, level=int(level))
            else:
                history_records = ledger.get_by_skill(skill_name)
                if not history_records and alt_name:
                    history_records = ledger.get_by_skill(alt_name)

            # Convert to history format
            history = []
            for record in history_records[-10:]:  # Last 10
                history.append({
                    "level": record.level,
                    "accuracy": record.accuracy,
                    "correct": record.correct,
                    "total": record.total,
                    "timestamp": record.timestamp,
                    "step": record.checkpoint_step,
                })
            response["history"] = history

            self._send_json(response)

        except Exception as e:
            logger.error(f"Eval results error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _serve_hero_info(self):
        """Serve active hero information from campaign/hero config."""
        try:
            from core.hero import get_active_hero
            hero = get_active_hero()
            self._send_json(hero)
        except Exception as e:
            logger.error(f"Hero info error: {e}")
            # Fallback to generic hero
            self._send_json({
                "name": "Hero",
                "rpg_name": "The Apprentice",
                "icon": "🦸",
                "model_name": "Unknown",
                "hero_id": "",
                "campaign_id": "",
                "error": str(e)
            })

    def _serve_hero_config(self, hero_id: str):
        """Serve hero configuration YAML as JSON for a specific hero."""
        try:
            import yaml
            hero_file = BASE_DIR / "configs" / "heroes" / f"{hero_id}.yaml"

            if not hero_file.exists():
                self._send_json({"error": f"Hero config not found: {hero_id}"}, 404)
                return

            with open(hero_file) as f:
                config = yaml.safe_load(f)

            self._send_json(config)
        except Exception as e:
            logger.error(f"Hero config error for {hero_id}: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_skills(self):
        """Serve skill data from YAML configs."""
        try:
            data = get_skills_data()
            self._send_json(data)
        except Exception as e:
            logger.error(f"Skills error: {e}")
            self._send_json({"skills": [], "error": str(e)})

    def _serve_titles(self):
        """Serve hero titles based on current training state."""
        try:
            from guild.titles import get_titles

            # Get training status for total steps
            status_file = BASE_DIR / "status" / "training_status.json"
            total_steps = 0
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)
                    total_steps = status.get("global_step", 0)

            # Get skill states from curriculum
            skill_states = {}
            skills_data = get_skills_data()
            total_level = 0
            for skill in skills_data.get("skills", []):
                skill_id = skill.get("id")
                if skill_id:
                    mastered = skill.get("mastered_level", 0)
                    total_level += mastered
                    skill_states[skill_id] = {
                        "level": skill.get("training_level", 1),
                        "accuracy": skill.get("recent_accuracy", 0) / 100.0,  # Convert from percent
                        "primitive_accuracy": {},  # Could load from skill state
                    }

            # Get titles
            result = get_titles(total_steps, total_level, skill_states)

            # Format response
            response = {
                "primary": {
                    "id": result.primary.id,
                    "name": result.primary.name,
                    "description": result.primary.description,
                    "category": result.primary.category,
                } if result.primary else None,
                "skill_titles": {
                    k: {"id": v.id, "name": v.name, "description": v.description}
                    for k, v in result.skill_titles.items()
                },
                "warnings": [
                    {"id": w.id, "name": w.name, "description": w.description, "icon": w.icon}
                    for w in result.warnings
                ],
                "achievements": [
                    {"id": a.id, "name": a.name, "icon": a.icon}
                    for a in result.achievements
                ],
                "total_count": len(result.all_titles),
                "total_steps": total_steps,
                "total_level": total_level,
            }
            self._send_json(response)
        except Exception as e:
            logger.error(f"Titles error: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({
                "primary": {"id": "unknown", "name": "Unknown", "description": ""},
                "skill_titles": {},
                "warnings": [],
                "achievements": [],
                "error": str(e)
            })

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

    def _weaver_start(self):
        """Start the Weaver daemon."""
        try:
            import subprocess

            pid_file = BASE_DIR / ".pids" / "weaver.pid"

            # Check if already running
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                    self._send_json({
                        "success": False,
                        "message": f"Weaver already running (PID {pid})"
                    })
                    return
                except (ValueError, ProcessLookupError, PermissionError):
                    # Stale PID file, remove it
                    pid_file.unlink(missing_ok=True)

            # Start Weaver daemon
            log_file = BASE_DIR / "logs" / "weaver.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as log:
                proc = subprocess.Popen(
                    ["python3", str(BASE_DIR / "weaver" / "weaver.py"), "--daemon"],
                    cwd=str(BASE_DIR),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )

            # Wait briefly for startup
            import time
            time.sleep(2)

            # Check if started successfully
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                self._send_json({
                    "success": True,
                    "message": f"Weaver started (PID {pid})",
                    "pid": pid
                })
            else:
                self._send_json({
                    "success": False,
                    "message": "Weaver failed to start - check logs/weaver.log"
                })

        except Exception as e:
            logger.error(f"Weaver start error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _weaver_stop(self):
        """Stop the Weaver daemon."""
        try:
            pid_file = BASE_DIR / ".pids" / "weaver.pid"

            if not pid_file.exists():
                self._send_json({
                    "success": False,
                    "message": "Weaver not running (no PID file)"
                })
                return

            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)

                # Wait for graceful shutdown
                import time
                for _ in range(10):
                    time.sleep(0.5)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    # Force kill if still running
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

                pid_file.unlink(missing_ok=True)
                self._send_json({
                    "success": True,
                    "message": f"Weaver stopped (was PID {pid})"
                })

            except ProcessLookupError:
                pid_file.unlink(missing_ok=True)
                self._send_json({
                    "success": True,
                    "message": "Weaver was not running (stale PID file removed)"
                })
            except PermissionError:
                self._send_json({
                    "success": False,
                    "message": "Permission denied stopping Weaver"
                }, 403)

        except Exception as e:
            logger.error(f"Weaver stop error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _serve_task_master_status(self):
        """Get Task Master (background task scheduler) status."""
        try:
            status_file = BASE_DIR / "status" / "task_master.json"

            if not status_file.exists():
                self._send_json({
                    "available": False,
                    "message": "Task Master has not run yet",
                    "gpus": {},
                    "last_task": None,
                    "stats": {}
                })
                return

            with open(status_file) as f:
                status = json.load(f)

            # Check if Task Master daemon is running
            pid_file = BASE_DIR / ".pids" / "task_master.pid"
            daemon_running = False
            daemon_pid = None

            if pid_file.exists():
                try:
                    daemon_pid = int(pid_file.read_text().strip())
                    os.kill(daemon_pid, 0)
                    daemon_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    daemon_running = False

            # Enrich with daemon status
            status["daemon_running"] = daemon_running
            status["daemon_pid"] = daemon_pid
            status["available"] = True

            self._send_json(status)
        except Exception as e:
            logger.error(f"Task Master status error: {e}")
            self._send_json({"error": str(e), "available": False}, 500)

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
                log_file.parent.mkdir(parents=True, exist_ok=True)

                # Set PYTHONPATH so daemon can import trainer, guild, etc.
                env = os.environ.copy()
                env["PYTHONPATH"] = str(BASE_DIR)

                subprocess.Popen(
                    ["nohup", "python3", str(daemon_script)],
                    stdout=open(log_file, "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(BASE_DIR),
                    env=env
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

    # ==========================================================================
    # CAMPAIGN SYSTEM - Hero and Campaign Management
    # ==========================================================================

    def _serve_heroes_data(self):
        """Serve list of available heroes."""
        try:
            from guild.heroes import list_heroes, get_hero

            hero_ids = list_heroes()
            heroes = []
            for hero_id in hero_ids:
                hero = get_hero(hero_id)
                heroes.append({
                    "id": hero.id,
                    "name": hero.name,
                    "rpg_name": hero.rpg_name,
                    "description": hero.description,
                    "model": {
                        "hf_name": hero.model.hf_name,
                        "family": hero.model.family,
                        "size_b": hero.model.size_b,
                    },
                    "display": {
                        "color": hero.display.color,
                        "emoji": hero.display.emoji,
                    },
                    "skills_affinity": hero.skills_affinity,
                })

            self._send_json(heroes)

        except ImportError as e:
            logger.error(f"Hero module not available: {e}")
            self._send_json({"error": "Hero system not installed"}, 503)
        except Exception as e:
            logger.error(f"Heroes data error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_campaigns_data(self):
        """Serve campaigns list and active campaign."""
        try:
            from guild.campaigns import CampaignManager

            mgr = CampaignManager(BASE_DIR)
            active = mgr.get_active()

            # Get all campaigns by hero
            campaigns = {}
            for hero_id in mgr.list_heroes():
                hero_campaigns = mgr.list_campaigns(hero_id, include_archived=True)
                campaigns[hero_id] = [c.to_dict() for c in hero_campaigns]

            result = {
                "active": active.to_dict() if active else None,
                "campaigns": campaigns,
            }

            self._send_json(result)

        except ImportError as e:
            logger.error(f"Campaign module not available: {e}")
            self._send_json({"error": "Campaign system not installed"}, 503)
        except Exception as e:
            logger.error(f"Campaigns data error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_active_campaign(self):
        """Serve only the active campaign."""
        try:
            from guild.campaigns import get_active_campaign

            active = get_active_campaign(BASE_DIR)
            if active:
                self._send_json(active.to_dict())
            else:
                self._send_json(None)

        except ImportError as e:
            logger.error(f"Campaign module not available: {e}")
            self._send_json({"error": "Campaign system not installed"}, 503)
        except Exception as e:
            logger.error(f"Active campaign error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # ANALYSIS API - Model Archaeology
    # =========================================================================

    def _get_analysis_dir(self, campaign_id: str, hero_id: str = None) -> Path:
        """Get the analysis directory for a campaign."""
        if not hero_id:
            hero_id = get_active_hero_id()  # Use active campaign's hero
        if not hero_id:
            logger.warning("No active hero, cannot resolve analysis dir")
            return BASE_DIR / "campaigns" / "unknown" / campaign_id / "analysis"
        return BASE_DIR / "campaigns" / hero_id / campaign_id / "analysis"

    def _serve_analysis_layer_stats_list(self, campaign_id: str, query: dict):
        """List available layer stats for a campaign."""
        hero_id = query.get("hero_id", [get_active_hero_id()])[0]
        analysis_dir = self._get_analysis_dir(campaign_id, hero_id) / "layer_stats"

        if not analysis_dir.exists():
            self._send_json({"stats": [], "count": 0, "campaign_id": campaign_id})
            return

        try:
            stats = []
            for f in sorted(analysis_dir.glob("*.layer_stats.json")):
                # Read summary fields only
                with open(f) as fp:
                    data = json.load(fp)

                stats.append({
                    "checkpoint_step": data.get("checkpoint_step", 0),
                    "created_at": data.get("created_at"),
                    "has_drift": bool(data.get("drift_stats")),
                    "has_activations": bool(data.get("activation_stats")),
                    "num_layers": len(data.get("weight_stats", {})),
                    "most_changed_layer": (
                        data.get("global_drift_stats", {}).get("most_changed_layer")
                    ),
                    "avg_weight_norm": (
                        data.get("global_weight_stats", {}).get("avg_weight_norm")
                    ),
                    "compute_duration_sec": data.get("compute_duration_sec", 0),
                    "filename": f.name,
                })

            self._send_json({
                "stats": stats,
                "count": len(stats),
                "campaign_id": campaign_id,
                "hero_id": hero_id,
            })

        except Exception as e:
            logger.error(f"Layer stats list error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_analysis_layer_stats_detail(self, campaign_id: str, checkpoint_name: str, query: dict):
        """Get full layer stats for a specific checkpoint."""
        hero_id = query.get("hero_id", [get_active_hero_id()])[0]
        analysis_dir = self._get_analysis_dir(campaign_id, hero_id) / "layer_stats"

        # Handle both "183000" and "ckpt-183000.layer_stats.json" formats
        if checkpoint_name.isdigit():
            filename = f"ckpt-{int(checkpoint_name):06d}.layer_stats.json"
        else:
            filename = checkpoint_name

        filepath = analysis_dir / filename

        if not filepath.exists():
            self._send_json({"error": f"Not found: {filename}"}, 404)
            return

        try:
            with open(filepath) as f:
                data = json.load(f)
            self._send_json(data)

        except Exception as e:
            logger.error(f"Layer stats detail error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_analysis_drift_timeline(self, campaign_id: str, query: dict):
        """
        Get drift time series for visualization.

        Returns per-layer drift over checkpoints for heatmap/chart display.
        """
        hero_id = query.get("hero_id", [get_active_hero_id()])[0]
        layer_filter = query.get("layers", [None])[0]  # Comma-separated layer names
        max_layers = int(query.get("max_layers", [30])[0])

        analysis_dir = self._get_analysis_dir(campaign_id, hero_id) / "layer_stats"

        if not analysis_dir.exists():
            self._send_json({"error": "No analysis data found"}, 404)
            return

        try:
            timeline = {
                "checkpoints": [],
                "layers": {},
            }

            # Collect all layer stats with drift data
            for f in sorted(analysis_dir.glob("*.layer_stats.json")):
                with open(f) as fp:
                    data = json.load(fp)

                if not data.get("drift_stats"):
                    continue

                step = data.get("checkpoint_step", 0)
                timeline["checkpoints"].append(step)

                for layer_name, drift in data["drift_stats"].items():
                    # Apply layer filter if specified
                    if layer_filter:
                        allowed = layer_filter.split(",")
                        if not any(a in layer_name for a in allowed):
                            continue

                    if layer_name not in timeline["layers"]:
                        timeline["layers"][layer_name] = {
                            "name": layer_name,
                            "drift_l2": [],
                            "drift_cosine": [],
                        }

                    timeline["layers"][layer_name]["drift_l2"].append(
                        drift.get("total_l2", 0)
                    )
                    timeline["layers"][layer_name]["drift_cosine"].append(
                        drift.get("avg_cosine", 1.0)
                    )

            # Limit number of layers (sort by total drift, keep top N)
            if len(timeline["layers"]) > max_layers:
                sorted_layers = sorted(
                    timeline["layers"].items(),
                    key=lambda x: sum(x[1]["drift_l2"]),
                    reverse=True
                )
                timeline["layers"] = dict(sorted_layers[:max_layers])

            self._send_json(timeline)

        except Exception as e:
            logger.error(f"Drift timeline error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_analysis_top_movers(self, campaign_id: str, query: dict):
        """Get layers that changed the most across training."""
        hero_id = query.get("hero_id", [get_active_hero_id()])[0]
        top_n = int(query.get("n", [10])[0])

        analysis_dir = self._get_analysis_dir(campaign_id, hero_id) / "layer_stats"

        if not analysis_dir.exists():
            self._send_json({"error": "No analysis data found"}, 404)
            return

        try:
            # Accumulate total drift per layer
            layer_drift = {}
            checkpoint_count = 0

            for f in sorted(analysis_dir.glob("*.layer_stats.json")):
                with open(f) as fp:
                    data = json.load(fp)

                if not data.get("drift_stats"):
                    continue

                checkpoint_count += 1

                for layer_name, drift in data["drift_stats"].items():
                    if layer_name not in layer_drift:
                        layer_drift[layer_name] = 0
                    layer_drift[layer_name] += drift.get("total_l2", 0)

            # Sort by total drift
            sorted_layers = sorted(
                layer_drift.items(),
                key=lambda x: x[1],
                reverse=True
            )

            self._send_json({
                "top_movers": [
                    {"layer": name, "total_drift": drift}
                    for name, drift in sorted_layers[:top_n]
                ],
                "most_stable": [
                    {"layer": name, "total_drift": drift}
                    for name, drift in sorted_layers[-top_n:][::-1]
                ],
                "total_layers": len(layer_drift),
                "checkpoints_analyzed": checkpoint_count,
                "campaign_id": campaign_id,
                "hero_id": hero_id,
            })

        except Exception as e:
            logger.error(f"Top movers error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _create_campaign(self):
        """Create a new campaign."""
        try:
            from guild.campaigns import CampaignManager

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            hero_id = body.get("hero_id")
            name = body.get("name")
            description = body.get("description", "")
            skills_focus = body.get("skills_focus", [])
            config_overrides = body.get("config_overrides", {})

            if not hero_id:
                self._send_json({"success": False, "error": "Missing hero_id"}, 400)
                return
            if not name:
                self._send_json({"success": False, "error": "Missing name"}, 400)
                return

            mgr = CampaignManager(BASE_DIR)
            campaign = mgr.create_campaign(
                hero_id=hero_id,
                name=name,
                description=description,
                skills_focus=skills_focus,
                config_overrides=config_overrides,
            )

            logger.info(f"Created campaign: {campaign.hero_id}/{campaign.id}")
            self._send_json({
                "success": True,
                "campaign": campaign.to_dict(),
            })

        except ImportError as e:
            logger.error(f"Campaign module not available: {e}")
            self._send_json({"error": "Campaign system not installed"}, 503)
        except Exception as e:
            logger.error(f"Create campaign error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _activate_campaign(self):
        """Activate a campaign."""
        try:
            from guild.campaigns import CampaignManager

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            hero_id = body.get("hero_id")
            campaign_id = body.get("campaign_id")

            if not hero_id or not campaign_id:
                self._send_json({"success": False, "error": "Missing hero_id or campaign_id"}, 400)
                return

            mgr = CampaignManager(BASE_DIR)
            campaign = mgr.get_campaign(hero_id, campaign_id)
            mgr.activate(campaign)

            logger.info(f"Activated campaign: {hero_id}/{campaign_id}")
            self._send_json({
                "success": True,
                "campaign": campaign.to_dict(),
            })

        except ImportError as e:
            logger.error(f"Campaign module not available: {e}")
            self._send_json({"error": "Campaign system not installed"}, 503)
        except Exception as e:
            logger.error(f"Activate campaign error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _archive_campaign(self):
        """Archive a campaign to the Hall of Legends."""
        try:
            from guild.campaigns import CampaignManager

            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            hero_id = body.get("hero_id")
            campaign_id = body.get("campaign_id")

            if not hero_id or not campaign_id:
                self._send_json({"success": False, "error": "Missing hero_id or campaign_id"}, 400)
                return

            mgr = CampaignManager(BASE_DIR)
            campaign = mgr.archive(hero_id, campaign_id)

            logger.info(f"Archived campaign: {hero_id}/{campaign_id}")
            self._send_json({
                "success": True,
                "campaign": campaign.to_dict(),
            })

        except ImportError as e:
            logger.error(f"Campaign module not available: {e}")
            self._send_json({"error": "Campaign system not installed"}, 503)
        except Exception as e:
            logger.error(f"Archive campaign error: {e}")
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
        """Serve vault assets - base model, checkpoints, etc. Campaign-aware."""
        try:
            import os
            from datetime import datetime

            # Get active campaign
            active_hero_id = None
            active_campaign_id = None
            try:
                from guild.campaigns import get_active_campaign
                active = get_active_campaign(BASE_DIR)
                if active:
                    active_hero_id = active.hero_id
                    active_campaign_id = active.id
            except Exception:
                pass

            assets = {
                "base_model": None,
                "checkpoints": [],
                "total_size_gb": 0,
                "last_updated": datetime.now().isoformat(),
                "active_hero_id": active_hero_id,
                "active_campaign_id": active_campaign_id,
            }

            # Load ALL hero base models (active one highlighted, others faded)
            assets["base_models"] = []
            seen_paths = set()

            try:
                from guild.heroes import list_heroes, get_hero
                for hero_id in list_heroes():
                    try:
                        hero = get_hero(hero_id)
                        if hero and hero.model:
                            hero_model_path = hero.model.hf_name
                            # Handle relative paths
                            if not hero_model_path.startswith("/"):
                                hero_model_path = BASE_DIR / hero_model_path
                            else:
                                hero_model_path = Path(hero_model_path)

                            # Skip if already seen (same model for multiple heroes)
                            path_str = str(hero_model_path)
                            if path_str in seen_paths:
                                continue
                            seen_paths.add(path_str)

                            if hero_model_path.exists():
                                size_bytes = sum(
                                    f.stat().st_size for f in hero_model_path.rglob("*") if f.is_file()
                                )
                                size_gb = round(size_bytes / (1024**3), 2)
                                is_active = (hero_id == active_hero_id)

                                assets["base_models"].append({
                                    "name": f"{hero.name.lower()}_{hero.model.size_b}b",
                                    "display_name": f"{hero.name}'s Base ({hero.model.size_b}B)",
                                    "hero_id": hero_id,
                                    "hero_name": hero.name,
                                    "path": path_str,
                                    "size_gb": size_gb,
                                    "is_active": is_active,
                                })
                                assets["total_size_gb"] += size_gb
                    except Exception as e:
                        logger.warning(f"Could not load hero {hero_id}: {e}")
            except Exception as e:
                logger.warning(f"Could not list heroes: {e}")

            # Sort: active first, then by size descending
            assets["base_models"].sort(key=lambda x: (not x["is_active"], -x["size_gb"]))

            # Keep backward compat: set base_model to active one
            active_models = [m for m in assets["base_models"] if m["is_active"]]
            if active_models:
                assets["base_model"] = active_models[0]

            def scan_checkpoint_dir(checkpoint_dir, hero_id=None, campaign_id=None):
                """Scan a directory for checkpoints."""
                checkpoints = []
                if not checkpoint_dir.exists():
                    return checkpoints

                for item in sorted(checkpoint_dir.iterdir(), reverse=True):
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

                            checkpoints.append({
                                "name": item.name,
                                "step": step,
                                "path": str(item),
                                "size_gb": size_gb,
                                "created": mtime.isoformat(),
                                "age_hours": round((datetime.now() - mtime).total_seconds() / 3600, 1),
                                "eval_count": 0,
                                "hero_id": hero_id,
                                "campaign_id": campaign_id,
                                "is_active_campaign": (hero_id == active_hero_id and campaign_id == active_campaign_id),
                            })
                        except (ValueError, OSError) as e:
                            logger.warning(f"Error processing {item}: {e}")
                return checkpoints

            # Scan campaign checkpoint directories
            campaigns_dir = BASE_DIR / "campaigns"
            if campaigns_dir.exists():
                for hero_dir in campaigns_dir.iterdir():
                    if hero_dir.is_dir() and hero_dir.name not in ('active', 'archived'):
                        hero_id = hero_dir.name
                        for campaign_dir in hero_dir.iterdir():
                            if campaign_dir.is_dir():
                                campaign_id = campaign_dir.name
                                checkpoint_dir = campaign_dir / "checkpoints"
                                cps = scan_checkpoint_dir(checkpoint_dir, hero_id, campaign_id)
                                assets["checkpoints"].extend(cps)
                                for cp in cps:
                                    assets["total_size_gb"] += cp["size_gb"]

            # Also scan legacy current_model/ directory (for DIO's old checkpoints)
            current_model_dir = BASE_DIR / "current_model"
            if current_model_dir.exists():
                # Assume current_model belongs to DIO unless we know otherwise
                legacy_hero = "dio-qwen3-0.6b"
                legacy_campaign = "campaign-001"
                cps = scan_checkpoint_dir(current_model_dir, legacy_hero, legacy_campaign)
                # Mark as active if DIO campaign-001 is active
                for cp in cps:
                    cp["is_active_campaign"] = (legacy_hero == active_hero_id and legacy_campaign == active_campaign_id)
                assets["checkpoints"].extend(cps)
                for cp in cps:
                    assets["total_size_gb"] += cp["size_gb"]

            # Sort all checkpoints by step descending
            assets["checkpoints"].sort(key=lambda x: x["step"], reverse=True)

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

    def _delete_checkpoints(self):
        """Delete checkpoints from the vault."""
        import shutil

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            paths = body.get("paths", [])
            steps = body.get("steps", [])

            if not paths and not steps:
                self._send_json({"success": False, "error": "No paths or steps provided"}, 400)
                return

            deleted = []
            errors = []

            # If steps provided, find all checkpoint paths matching those steps
            if steps:
                # Scan for checkpoints matching the steps
                checkpoint_dirs = [
                    BASE_DIR / "current_model",
                ]
                # Also scan campaign checkpoint directories
                campaigns_dir = BASE_DIR / "campaigns"
                if campaigns_dir.exists():
                    for hero_dir in campaigns_dir.iterdir():
                        if hero_dir.is_dir() and hero_dir.name not in ('active', 'archived'):
                            for campaign_dir in hero_dir.iterdir():
                                if campaign_dir.is_dir():
                                    cp_dir = campaign_dir / "checkpoints"
                                    if cp_dir.exists():
                                        checkpoint_dirs.append(cp_dir)

                for step in steps:
                    for cp_dir in checkpoint_dirs:
                        if not cp_dir.exists():
                            continue
                        for item in cp_dir.iterdir():
                            if item.is_dir() and item.name.startswith(f"checkpoint-{step}"):
                                paths.append(str(item))

            # Delete each path
            for path_str in paths:
                path = Path(path_str)
                try:
                    if path.exists() and path.is_dir():
                        # Safety check - must be a checkpoint directory
                        if not path.name.startswith("checkpoint-"):
                            errors.append(f"Skipped (not a checkpoint): {path_str}")
                            continue

                        # Safety check - must be under TRAINING directory
                        if not str(path).startswith(str(BASE_DIR)):
                            errors.append(f"Skipped (outside base dir): {path_str}")
                            continue

                        shutil.rmtree(path)
                        deleted.append(path_str)
                        logger.info(f"Deleted checkpoint: {path_str}")
                    else:
                        errors.append(f"Not found: {path_str}")
                except Exception as e:
                    errors.append(f"Error deleting {path_str}: {e}")

            self._send_json({
                "success": True,
                "deleted": deleted,
                "deleted_count": len(deleted),
                "errors": errors,
            })

        except Exception as e:
            logger.error(f"Delete checkpoints error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    # =========================================================================
    # JOBS API - Distributed job execution
    # =========================================================================

    def _get_job_client(self):
        """Get job store client, caching for performance."""
        if not hasattr(self, '_job_client'):
            try:
                from jobs.client import JobStoreClient
                self._job_client = JobStoreClient()
            except ImportError:
                self._job_client = None
        return self._job_client

    def _serve_jobs_list(self, query):
        """List jobs with optional filters."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            status = query.get("status", [None])[0]
            job_type = query.get("type", [None])[0]
            limit = int(query.get("limit", [100])[0])

            jobs = client.list(status=status, job_type=job_type, limit=limit)

            self._send_json({
                "count": len(jobs),
                "jobs": jobs,
            })

        except Exception as e:
            logger.error(f"Jobs list error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_jobs_stats(self):
        """Get job statistics."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            stats = client.stats()
            self._send_json(stats)

        except Exception as e:
            logger.error(f"Jobs stats error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_job(self, job_id: str):
        """Get a specific job."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            job = client.get(job_id)
            if job:
                self._send_json(job)
            else:
                self._send_json({"error": f"Job not found: {job_id}"}, 404)

        except Exception as e:
            logger.error(f"Jobs get error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _submit_job(self):
        """Submit a generic job."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            from guild.job_types import JobSpec
            spec = JobSpec.from_dict(body)
            job_id = client.submit(spec)

            self._send_json({
                "success": True,
                "job_id": job_id,
            }, 201)

        except Exception as e:
            logger.error(f"Submit job error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _submit_eval_job(self):
        """Submit an eval job (convenience endpoint)."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            skill_id = body.get("skill_id", "bin")
            level = int(body.get("level", 1))
            batch_size = int(body.get("batch_size", 100))

            # Get RunContext for model identity anchoring
            from core.run_context import get_run_context
            ctx = get_run_context()

            from jobs import eval_job
            spec = eval_job(
                skill_id,
                level,
                batch_size,
                # Anchor to current run context
                hero_id=ctx.hero_id,
                campaign_id=ctx.campaign_id,
                checkpoint_path=ctx.current_model_dir,
                context_hash=ctx.context_hash(),
            )
            job_id = client.submit(spec)

            self._send_json({
                "success": True,
                "job_id": job_id,
                "skill_id": skill_id,
                "level": level,
                "hero_id": ctx.hero_id,
                "context_hash": ctx.context_hash(),
            }, 201)

        except Exception as e:
            logger.error(f"Submit eval job error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _cancel_job(self, job_id: str):
        """Cancel a job."""
        client = self._get_job_client()
        if not client:
            self._send_json({"error": "Jobs module not available"}, 500)
            return

        try:
            success = client.cancel(job_id)

            if success:
                self._send_json({"success": True, "status": "cancelled"})
            else:
                self._send_json({"success": False, "error": "Cannot cancel job"}, 400)

        except Exception as e:
            logger.error(f"Cancel job error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    def _handle_reset(self):
        """
        Reset training environment - clear stale state while preserving models.

        POST /api/reset
        Body: {"keep_jobs": false}  # optional
        """
        try:
            from core.reset import reset_environment
            from core.paths import get_base_dir

            content_length = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_length > 0:
                body = json.loads(self.rfile.read(content_length).decode())

            keep_jobs = body.get("keep_jobs", False)

            result = reset_environment(
                keep_jobs=keep_jobs,
                base_dir=get_base_dir(),
            )

            self._send_json({
                "success": True,
                "message": "Environment reset complete",
                "results": result.as_counts(),
            })

        except Exception as e:
            logger.error(f"Reset error: {e}")
            self._send_json({"success": False, "error": str(e)}, 500)

    # =========================================================================
    # FORGE - Data Validation and Queue Health
    # =========================================================================

    def _serve_forge_status(self):
        """Get Forge status including queue health and validation stats."""
        try:
            from core.training_queue import TrainingQueue
            from core.paths import get_base_dir

            queue = TrainingQueue(str(get_base_dir()))
            base_status = queue.get_queue_status()

            # Get rejected files
            rejected_dir = queue.queue_dir / "rejected"
            rejected_files = []
            recent_rejections = []

            if rejected_dir.exists():
                for jsonl_file in sorted(rejected_dir.glob("*.jsonl"))[-20:]:
                    rejected_files.append(jsonl_file.name)

                    # Try to get rejection report
                    report_file = rejected_dir / f"{jsonl_file.stem}.rejection.json"
                    if report_file.exists():
                        try:
                            import json
                            with open(report_file) as f:
                                report = json.load(f)
                                recent_rejections.append({
                                    "file": jsonl_file.name,
                                    "rejected_at": report.get("rejected_at", ""),
                                    "errors": report.get("validation", {}).get("errors", [])[:3],
                                    "leakage_count": report.get("validation", {}).get("leakage_count", 0),
                                })
                        except Exception:
                            pass

            # Get eval banks info
            eval_banks = []
            try:
                from forge.leakage import EvalBankManager
                manager = EvalBankManager()
                banks = manager.list_banks()
                for skill_id, counts in banks.items():
                    eval_banks.append({
                        "skill_id": skill_id,
                        "ids": counts["ids"],
                        "prompt_hashes": counts["prompt_hashes"],
                    })
            except Exception:
                pass

            # Get dataset contracts
            contracts = []
            try:
                from forge.contracts import list_contracts
                for contract in list_contracts():
                    contracts.append({
                        "id": contract.id,
                        "version": contract.version,
                        "description": contract.description[:100] if contract.description else "",
                        "skill_id": contract.skill_id,
                        "max_invalid_fraction": contract.max_invalid_fraction,
                        "required_fields": contract.required_fields,
                    })
            except Exception:
                pass

            # Get shard state summary
            datasets = []
            try:
                from forge.state import get_forge_state
                state_mgr = get_forge_state()
                summary = state_mgr.get_summary()
                for dataset_id, info in summary.get("datasets", {}).items():
                    datasets.append({
                        "id": dataset_id,
                        "total_shards": info["total"],
                        "ready": info["by_status"].get("ready", 0),
                        "rejected": info["by_status"].get("rejected", 0),
                        "pending": info["by_status"].get("unknown", 0) + info["by_status"].get("pending", 0),
                    })
            except Exception:
                pass

            self._send_json({
                "queue": {
                    "high": base_status.get("queued", {}).get("high", 0),
                    "normal": base_status.get("queued", {}).get("normal", 0),
                    "low": base_status.get("queued", {}).get("low", 0),
                    "processing": base_status.get("processing", 0),
                    "total": base_status.get("total_queued", 0),
                },
                "rejected": {
                    "count": len(rejected_files),
                    "files": rejected_files[-10:],
                },
                "recent_rejections": recent_rejections[-5:],
                "eval_banks": eval_banks,
                "contracts": contracts,
                "datasets": datasets,
            })

        except Exception as e:
            logger.error(f"Forge status error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_forge_rejected(self):
        """Get list of rejected files with details."""
        try:
            from core.training_queue import TrainingQueue
            from core.paths import get_base_dir
            import json

            queue = TrainingQueue(str(get_base_dir()))
            rejected_dir = queue.queue_dir / "rejected"

            rejected = []
            if rejected_dir.exists():
                for jsonl_file in sorted(rejected_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
                    entry = {
                        "file": jsonl_file.name,
                        "size_bytes": jsonl_file.stat().st_size,
                        "rejected_at": None,
                        "errors": [],
                        "leakage_count": 0,
                    }

                    report_file = rejected_dir / f"{jsonl_file.stem}.rejection.json"
                    if report_file.exists():
                        try:
                            with open(report_file) as f:
                                report = json.load(f)
                                entry["rejected_at"] = report.get("rejected_at")
                                validation = report.get("validation", {})
                                entry["errors"] = validation.get("errors", [])[:5]
                                entry["warnings"] = validation.get("warnings", [])[:5]
                                entry["leakage_count"] = validation.get("leakage_count", 0)
                                entry["stats"] = validation.get("stats", {})
                        except Exception:
                            pass

                    rejected.append(entry)

            self._send_json({"rejected": rejected[:50]})

        except Exception as e:
            logger.error(f"Forge rejected error: {e}")
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

            # 4. Detailed eval results with model responses from eval_results_history.json
            results_file = BASE_DIR / "status" / "eval_results_history.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results_data = json.load(f)
                    # Find results near this step
                    detailed_results = []
                    for skill_id, levels in results_data.items():
                        for level, evals_list in levels.items():
                            for eval_entry in evals_list:
                                eval_step = eval_entry.get("step", 0)
                                if abs(eval_step - step) <= 500:
                                    detailed_results.append({
                                        "skill": skill_id,
                                        "level": int(level),
                                        "step": eval_step,
                                        "accuracy": eval_entry.get("accuracy", 0),
                                        "correct": eval_entry.get("correct", 0),
                                        "total": eval_entry.get("total", 0),
                                        "timestamp": eval_entry.get("timestamp"),
                                        "results": eval_entry.get("results", []),  # Individual problem results
                                    })
                    if detailed_results:
                        evals["detailed_results"] = detailed_results
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

        # No hardcoded defaults - hosts.json is required
        if not hosts_file.exists():
            logger.error(f"Host configuration not found: {hosts_file}")
            logger.error("Create config/hosts.json with host definitions.")
            return {}

        try:
            with open(hosts_file) as f:
                config = json.load(f)
            return config.get("inference_hosts", {})
        except Exception as e:
            logger.warning(f"Failed to load hosts.json: {e}")
            return {}

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

    # =========================================================================
    # Evaluation Ledger API - Query eval results
    # =========================================================================

    def _serve_evals_list(self, query: dict):
        """List recent evals with optional filtering."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            limit = int(query.get("limit", [50])[0])

            # Optional filters
            skill = query.get("skill", [None])[0]
            hero_id = query.get("hero_id", [None])[0]
            campaign_id = query.get("campaign_id", [None])[0]

            if skill:
                level = query.get("level", [None])[0]
                level = int(level) if level else None
                records = ledger.get_by_skill(skill, level)
            elif hero_id:
                records = ledger.get_by_hero_campaign(hero_id, campaign_id)
            else:
                records = ledger.list_all(limit)

            self._send_json({
                "evals": [r.to_dict() for r in records[:limit]],
                "count": len(records),
            })

        except ImportError:
            self._send_json({
                "evals": [],
                "count": 0,
                "message": "Evaluation ledger not available"
            })
        except Exception as e:
            logger.error(f"Evals list error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_evals_summary(self):
        """Get evaluation summary statistics."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            summary = ledger.summary()
            self._send_json(summary)

        except ImportError:
            self._send_json({
                "total_evaluations": 0,
                "by_skill": {},
                "message": "Evaluation ledger not available"
            })
        except Exception as e:
            logger.error(f"Evals summary error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_evals_by_checkpoint(self, checkpoint_step: str, query: dict):
        """Get all evals for a checkpoint."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            step = int(checkpoint_step)

            # Optional hero/campaign filter
            hero_id = query.get("hero_id", [None])[0]
            campaign_id = query.get("campaign_id", [None])[0]

            records = ledger.get_by_checkpoint(step)

            # Filter by hero/campaign if specified
            if hero_id:
                records = [r for r in records if r.hero_id == hero_id]
            if campaign_id:
                records = [r for r in records if r.campaign_id == campaign_id]

            # Also get skills grouped view
            skills_map = ledger.get_checkpoint_skills(step, hero_id, campaign_id)

            self._send_json({
                "checkpoint_step": step,
                "evals": [r.to_dict() for r in records],
                "skills": {k: v.to_dict() for k, v in skills_map.items()},
                "count": len(records),
            })

        except ValueError:
            self._send_json({"error": "Invalid checkpoint step"}, 400)
        except Exception as e:
            logger.error(f"Evals by checkpoint error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_evals_by_skill(self, skill_id: str, query: dict):
        """Get all evals for a skill with optional level filter."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()

            level = query.get("level", [None])[0]
            level = int(level) if level else None
            hero_id = query.get("hero_id", [None])[0]
            campaign_id = query.get("campaign_id", [None])[0]
            limit = int(query.get("limit", [100])[0])

            records = ledger.get_by_skill(skill_id, level)

            # Filter by hero/campaign
            if hero_id:
                records = [r for r in records if r.hero_id == hero_id]
            if campaign_id:
                records = [r for r in records if r.campaign_id == campaign_id]

            # Get skill summary with trends
            skill_summary = ledger.get_by_skill(skill_id)  # Unfiltered for summary

            self._send_json({
                "skill_id": skill_id,
                "level": level,
                "evals": [r.to_dict() for r in records[:limit]],
                "count": len(records),
                "best_accuracy": max((r.accuracy for r in records), default=None),
            })

        except Exception as e:
            logger.error(f"Evals by skill error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_eval_by_job(self, job_id: str):
        """Get eval result for a specific job."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            record = ledger.get_by_job(job_id)

            if record:
                self._send_json({
                    "found": True,
                    "eval": record.to_dict(),
                })
            else:
                self._send_json({
                    "found": False,
                    "job_id": job_id,
                    "message": "No eval found for this job"
                })

        except Exception as e:
            logger.error(f"Eval by job error: {e}")
            self._send_json({"error": str(e)}, 500)

    # =========================================================================
    # Skill Engine API - Primitives and State
    # =========================================================================

    def _serve_engine_health(self):
        """Get Skill Engine health status."""
        try:
            from guild.skills import get_engine
            engine = get_engine()
            health = engine.health_check()
            health["skills_available"] = engine.list_skills()
            self._send_json(health)
        except ImportError as e:
            self._send_json({
                "error": f"Skill Engine not available: {e}",
                "total_skills": 0,
            }, 503)
        except Exception as e:
            logger.error(f"Engine health error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_all_primitives(self):
        """List all primitives across all skills."""
        try:
            from guild.skills import get_engine
            from guild.skills.primitives import PRIMITIVE_CATALOG

            engine = get_engine()
            primitives = {}

            # Get primitives from loaded skills
            for skill_id in engine.list_skills():
                try:
                    skill = engine.get(skill_id)
                    for prim in skill.primitives:
                        primitives[str(prim)] = {
                            "skill_id": skill_id,
                            "name": prim.name,
                            "track": prim.track,
                            "version": prim.version,
                        }
                except Exception as e:
                    logger.warning(f"Could not load skill {skill_id}: {e}")

            # Also include catalog primitives
            catalog_count = sum(len(prims) for prims in PRIMITIVE_CATALOG.values())

            self._send_json({
                "primitives": primitives,
                "count": len(primitives),
                "catalog_tracks": list(PRIMITIVE_CATALOG.keys()),
                "catalog_count": catalog_count,
            })

        except ImportError as e:
            self._send_json({"error": f"Skill Engine not available: {e}"}, 503)
        except Exception as e:
            logger.error(f"All primitives error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_primitive_summary(self):
        """Get per-primitive accuracy summary across all skills."""
        try:
            from guild.skills import get_engine

            engine = get_engine()
            summary = engine.get_primitive_summary()

            # Get states for additional info
            states = engine.all_states()
            states_data = {
                skill_id: {
                    "level": state.level,
                    "xp_total": state.xp_total,
                    "total_evals": state.total_evals,
                    "last_eval_accuracy": state.last_eval_accuracy,
                    "primitive_accuracy": state.primitive_accuracy,
                }
                for skill_id, state in states.items()
            }

            self._send_json({
                "primitive_accuracy": summary,
                "skill_states": states_data,
            })

        except ImportError as e:
            self._send_json({"error": f"Skill Engine not available: {e}"}, 503)
        except Exception as e:
            logger.error(f"Primitive summary error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_skill_primitives(self, skill_id: str):
        """Get primitives for a specific skill."""
        try:
            from guild.skills import get_engine

            engine = get_engine()
            skill = engine.get(skill_id)

            primitives = [
                {
                    "name": prim.name,
                    "track": prim.track,
                    "version": prim.version,
                    "id": str(prim),
                }
                for prim in skill.primitives
            ]

            self._send_json({
                "skill_id": skill_id,
                "primitives": primitives,
                "count": len(primitives),
            })

        except KeyError:
            self._send_json({"error": f"Skill not found: {skill_id}"}, 404)
        except ImportError as e:
            self._send_json({"error": f"Skill Engine not available: {e}"}, 503)
        except Exception as e:
            logger.error(f"Skill primitives error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _serve_skill_state(self, skill_id: str):
        """Get state for a specific skill including per-primitive accuracy."""
        try:
            from guild.skills import get_engine

            engine = get_engine()
            state = engine.get_state(skill_id)

            self._send_json({
                "skill_id": skill_id,
                "level": state.level,
                "xp_total": state.xp_total,
                "accuracy": state.accuracy,
                "total_evals": state.total_evals,
                "total_samples_seen": state.total_samples_seen,
                "last_eval_accuracy": state.last_eval_accuracy,
                "last_eval_timestamp": state.last_eval_timestamp,
                "primitive_accuracy": state.primitive_accuracy,
                "primitive_history": state.primitive_history,
            })

        except ImportError as e:
            self._send_json({"error": f"Skill Engine not available: {e}"}, 503)
        except Exception as e:
            logger.error(f"Skill state error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _handle_oracle_load(self):
        """Load a checkpoint on an inference host. Campaign-aware for base models."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode())

            step = body.get("step")
            host_id = body.get("host", "3090")
            hero_id = body.get("hero_id")  # Optional: for campaign-aware base model
            force_sync = body.get("sync", False)  # Force sync even if might exist

            if step is None:
                self._send_json({"success": False, "error": "Missing 'step'"}, 400)
                return

            if host_id not in self.INFERENCE_HOSTS:
                self._send_json({"success": False, "error": f"Unknown host: {host_id}"}, 400)
                return

            host_config = self.INFERENCE_HOSTS[host_id]
            models_dir = host_config.get("models_dir")
            if not models_dir:
                self._send_json({"success": False, "error": f"Host {host_id} missing models_dir in hosts.json"}, 500)
                return

            # Special handling for base model (step 0)
            if int(step) == 0:
                # Get hero's base model (campaign-aware)
                base_model_name = "Qwen3-0.6B"
                local_path = BASE_DIR / "models" / "Qwen3-0.6B"
                display_name = "Qwen3-0.6B (Base)"

                # Try to get hero-specific base model
                if hero_id:
                    try:
                        from guild.heroes import get_hero
                        hero = get_hero(hero_id)
                        if hero and hero.model:
                            hero_model_path = hero.model.hf_name
                            if not hero_model_path.startswith("/"):
                                hero_model_path = BASE_DIR / hero_model_path
                            else:
                                hero_model_path = Path(hero_model_path)
                            if hero_model_path.exists():
                                local_path = hero_model_path
                                base_model_name = f"{hero.name}-base"
                                display_name = f"{hero.name}'s Base ({hero.model.size_b}B)"
                                logger.info(f"Using hero {hero.name}'s base model: {local_path}")
                    except Exception as e:
                        logger.warning(f"Could not get hero model, using default: {e}")

                checkpoint_name = base_model_name
                remote_path = f"{models_dir}/{base_model_name}"

                # Check if base model exists on remote
                if not self._check_remote_checkpoint(host_config, checkpoint_name):
                    # Sync base model
                    logger.info(f"Syncing base model {checkpoint_name} to {host_id}...")
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
                    "hero_id": hero_id,
                    "checkpoint_name": display_name,
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
            models_dir = host_config.get("models_dir")
            if not models_dir:
                self._send_json({"success": False, "error": f"Host {host_id} missing models_dir in hosts.json"}, 500)
                return
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
        models_dir = host_config.get("models_dir")
        if not models_dir:
            self._send_json({"success": False, "error": f"Host {host_id} missing models_dir in hosts.json"}, 500)
            return
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

    # Battle Log - Tavern opened event
    try:
        from core.battle_log import log_system
        log_system(
            f"Tavern opened for business on port {port}",
            severity="success",
            source="tavern.server",
            details={"port": port, "api_host": api_host, "api_port": api_port},
        )
    except Exception:
        pass  # Don't let battle log errors affect startup

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
