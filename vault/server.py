"""
VaultKeeper API Server - REST API for remote asset queries.

The VaultKeeper Server allows remote devices (3090, Synology, etc.) to:
    - Query where assets are located
    - Request asset retrieval
    - Search the catalog
    - Get system statistics

Runs on port 8767 by default (next to 8765 inference, 8766 GPU scheduler).

Usage:
    # Start server
    python3 vault/server.py --port 8767

    # Query from remote
    curl http://trainer.local:8767/api/locate/checkpoint_175000
    curl http://trainer.local:8767/api/stats

RPG Flavor:
    The VaultKeeper Server is the Oracle's Window - a magical portal
    through which distant allies can consult the Great Ledger.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vault.keeper import VaultKeeper, get_vault_keeper
from vault.assets import AssetQuery, AssetType, AssetStatus
from vault.zones import (
    ZoneRegistry,
    ZoneClient,
    ZoneTransfer,
    ZoneStatus,
    get_zone_registry,
    push_to_zone,
    pull_from_zone,
)
from guild.job_types import JobErrorCode, JobStatus, JobType
from jobs.registry import (
    validate_job_type,
    validate_payload,
    get_job_config,
    check_queue_limits,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vault_server")


class VaultKeeperHandler(BaseHTTPRequestHandler):
    """HTTP request handler for VaultKeeper API."""

    # Reference to keeper (set by server)
    keeper: Optional[VaultKeeper] = None
    zone_registry: Optional[ZoneRegistry] = None
    job_store = None  # Set by run_server

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        """Send an error response."""
        self._send_json({"error": message, "status": status}, status)

    def _parse_body(self) -> Optional[Dict[str, Any]]:
        """Parse JSON body from request."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Route handling
        if path == "/health":
            self._handle_health()
        elif path == "/api/stats":
            self._handle_stats()
        elif path.startswith("/api/locate/"):
            asset_id = path.replace("/api/locate/", "")
            self._handle_locate(asset_id)
        elif path.startswith("/api/asset/"):
            asset_id = path.replace("/api/asset/", "")
            self._handle_get_asset(asset_id)
        elif path == "/api/search":
            self._handle_search(query)
        elif path == "/api/list":
            self._handle_list(query)
        elif path == "/api/checkpoints":
            self._handle_list_checkpoints()
        elif path == "/api/strongholds":
            self._handle_strongholds()
        elif path == "/api/export":
            self._handle_export()
        # Zone federation endpoints
        elif path == "/api/zones":
            self._handle_list_zones()
        elif path == "/api/zones/refresh":
            self._handle_refresh_zones()
        elif path.startswith("/api/zones/") and "/assets" in path:
            zone_id = path.replace("/api/zones/", "").replace("/assets", "")
            self._handle_zone_assets(zone_id, query)
        elif path.startswith("/api/zones/"):
            zone_id = path.replace("/api/zones/", "")
            self._handle_get_zone(zone_id)
        # Ledger API - Checkpoint history and stats
        elif path == "/api/ledger":
            self._handle_ledger_list(query)
        elif path == "/api/ledger/summary":
            self._handle_ledger_summary()
        elif path == "/api/ledger/best":
            self._handle_ledger_best(query)
        elif path.startswith("/api/ledger/"):
            step_str = path.replace("/api/ledger/", "")
            self._handle_ledger_get(step_str)
        # Training API - Status and control
        elif path == "/api/training/status":
            self._handle_training_status()
        elif path == "/api/training/queue":
            self._handle_training_queue()
        # Evaluation API - Skill and passive evaluations
        elif path == "/api/evals":
            self._handle_evals_list(query)
        elif path == "/api/evals/summary":
            self._handle_evals_summary()
        elif path == "/api/evals/queue":
            self._handle_evals_queue()
        elif path.startswith("/api/evals/checkpoint/"):
            step_str = path.replace("/api/evals/checkpoint/", "")
            self._handle_evals_by_checkpoint(step_str)
        elif path.startswith("/api/evals/skill/"):
            skill = path.replace("/api/evals/skill/", "")
            self._handle_evals_by_skill(skill, query)
        elif path.startswith("/api/evals/best"):
            self._handle_evals_best(query)
        # Passives API
        elif path == "/api/passives":
            self._handle_passives_list(query)
        elif path == "/api/passives/summary":
            self._handle_passives_summary()
        elif path == "/api/passives/queue":
            self._handle_passives_queue()
        elif path.startswith("/api/passives/checkpoint/"):
            step_str = path.replace("/api/passives/checkpoint/", "")
            self._handle_passives_by_checkpoint(step_str)
        # Battle Log API - MMO-style event stream
        elif path == "/api/battle_log":
            self._handle_battle_log(query)
        elif path == "/api/battle_log/channels":
            self._handle_battle_log_channels()
        # Jobs API - Distributed job execution
        elif path == "/api/jobs":
            self._handle_jobs_list(query)
        elif path == "/api/jobs/stats":
            self._handle_jobs_stats()
        elif path == "/api/jobs/workers":
            self._handle_workers_list()
        elif path == "/api/jobs/cluster":
            self._handle_cluster_status()
        elif path == "/api/jobs/health":
            self._handle_jobs_health()
        elif path == "/api/jobs/events":
            # Recent events across all jobs
            self._handle_recent_events(query)
        elif path.endswith("/events") and path.startswith("/api/jobs/"):
            # Events for specific job: /api/jobs/{id}/events
            job_id = path.replace("/api/jobs/", "").replace("/events", "")
            self._handle_job_events(job_id)
        elif path.startswith("/api/jobs/"):
            job_id = path.replace("/api/jobs/", "")
            self._handle_jobs_get(job_id)
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._parse_body()

        if path == "/api/register":
            self._handle_register(body)
        elif path == "/api/fetch":
            self._handle_fetch(body)
        elif path == "/api/push":
            self._handle_push(body)
        elif path == "/api/verify":
            self._handle_verify(body)
        elif path == "/api/scan":
            self._handle_scan(body)
        # Zone transfer endpoint
        elif path == "/api/transfer":
            self._handle_transfer(body)
        # Training control
        elif path == "/api/training/control":
            self._handle_training_control(body)
        # Jobs API - Distributed job execution
        elif path == "/api/jobs":
            self._handle_jobs_submit(body)
        elif path == "/api/jobs/claim":
            self._handle_jobs_claim(body)
        # Worker registration endpoints
        elif path == "/api/jobs/workers/register":
            self._handle_worker_register(body)
        elif path == "/api/jobs/workers/heartbeat":
            self._handle_worker_heartbeat(body)
        elif path.endswith("/complete"):
            job_id = path.replace("/api/jobs/", "").replace("/complete", "")
            self._handle_jobs_complete(job_id, body)
        elif path.endswith("/failed"):
            job_id = path.replace("/api/jobs/", "").replace("/failed", "")
            self._handle_jobs_failed(job_id, body)
        elif path.endswith("/running"):
            job_id = path.replace("/api/jobs/", "").replace("/running", "")
            self._handle_jobs_running(job_id, body)
        elif path.endswith("/cancel"):
            job_id = path.replace("/api/jobs/", "").replace("/cancel", "")
            self._handle_jobs_cancel(job_id)
        elif path.endswith("/release"):
            job_id = path.replace("/api/jobs/", "").replace("/release", "")
            self._handle_jobs_release(job_id)
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def _handle_health(self):
        """Health check endpoint."""
        self._send_json({
            "status": "healthy",
            "service": "vault_keeper",
            "timestamp": datetime.now().isoformat(),
        })

    def _handle_stats(self):
        """Get catalog statistics."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return
        stats = self.keeper.get_stats()
        self._send_json(stats)

    def _handle_locate(self, asset_id: str):
        """Locate an asset."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        result = self.keeper.locate(asset_id)
        self._send_json(result.to_dict())

    def _handle_get_asset(self, asset_id: str):
        """Get full asset details."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        asset = self.keeper.get(asset_id)
        if asset:
            self._send_json(asset.to_dict())
        else:
            self._send_error(f"Asset not found: {asset_id}", 404)

    def _handle_search(self, query: Dict[str, list]):
        """Search for assets."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        # Build query from params
        asset_query = AssetQuery()

        if "type" in query:
            try:
                asset_query.asset_type = AssetType(query["type"][0])
            except ValueError:
                pass

        if "name" in query:
            asset_query.name_pattern = query["name"][0]

        if "stronghold" in query:
            asset_query.stronghold = query["stronghold"][0]

        if "status" in query:
            try:
                asset_query.status = AssetStatus(query["status"][0])
            except ValueError:
                pass

        assets = self.keeper.search(asset_query)
        self._send_json({
            "count": len(assets),
            "assets": [a.to_dict() for a in assets],
        })

    def _handle_list(self, query: Dict[str, list]):
        """List assets by type or stronghold."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if "type" in query:
            try:
                asset_type = AssetType(query["type"][0])
                assets = self.keeper.list_by_type(asset_type)
            except ValueError:
                self._send_error(f"Invalid asset type: {query['type'][0]}", 400)
                return
        elif "stronghold" in query:
            assets = self.keeper.list_in_stronghold(query["stronghold"][0])
        else:
            # List all
            assets = self.keeper.search(AssetQuery())

        self._send_json({
            "count": len(assets),
            "assets": [a.to_dict() for a in assets],
        })

    def _handle_list_checkpoints(self):
        """List all checkpoints sorted by step."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        checkpoints = self.keeper.list_checkpoints()
        self._send_json({
            "count": len(checkpoints),
            "checkpoints": [c.to_dict() for c in checkpoints],
        })

    def _handle_strongholds(self):
        """List all strongholds."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        strongholds = self.keeper.storage.list_strongholds()
        self._send_json({
            "strongholds": [s.to_dict() for s in strongholds],
        })

    def _handle_export(self):
        """Export catalog."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        data = self.keeper.export_catalog()
        self._send_json(data)

    def _handle_register(self, body: Optional[Dict[str, Any]]):
        """Register an asset from path."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if not body or "path" not in body:
            self._send_error("Missing 'path' in request body", 400)
            return

        path = body["path"]
        stronghold = body.get("stronghold", "local_vault")
        is_primary = body.get("is_primary", True)

        try:
            asset = self.keeper.register_from_path(path, stronghold, is_primary)
            self._send_json({
                "success": True,
                "asset": asset.to_dict(),
            })
        except Exception as e:
            self._send_error(f"Registration failed: {str(e)}", 500)

    def _handle_fetch(self, body: Optional[Dict[str, Any]]):
        """Fetch an asset to local path."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if not body or "asset_id" not in body or "destination" not in body:
            self._send_error("Missing 'asset_id' or 'destination' in request body", 400)
            return

        asset_id = body["asset_id"]
        destination = body["destination"]
        from_stronghold = body.get("from_stronghold")

        result = self.keeper.fetch(asset_id, destination, from_stronghold)
        self._send_json(result.to_dict())

    def _handle_push(self, body: Optional[Dict[str, Any]]):
        """Push an asset to a stronghold."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if not body or "asset_id" not in body or "to_stronghold" not in body:
            self._send_error("Missing 'asset_id' or 'to_stronghold' in request body", 400)
            return

        asset_id = body["asset_id"]
        to_stronghold = body["to_stronghold"]
        destination_path = body.get("destination_path")

        result = self.keeper.push(asset_id, to_stronghold, destination_path)
        self._send_json(result.to_dict())

    def _handle_verify(self, body: Optional[Dict[str, Any]]):
        """Verify an asset's locations."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if not body or "asset_id" not in body:
            self._send_error("Missing 'asset_id' in request body", 400)
            return

        asset_id = body["asset_id"]
        stronghold = body.get("stronghold")

        results = self.keeper.verify(asset_id, stronghold)
        self._send_json({
            "asset_id": asset_id,
            "verification_results": results,
        })

    def _handle_scan(self, body: Optional[Dict[str, Any]]):
        """Scan and register assets from a directory."""
        if not self.keeper:
            self._send_error("Keeper not initialized", 500)
            return

        if not body or "path" not in body:
            self._send_error("Missing 'path' in request body", 400)
            return

        path = Path(body["path"])
        stronghold = body.get("stronghold", "local_vault")
        recursive = body.get("recursive", True)

        if not path.exists():
            self._send_error(f"Path not found: {path}", 404)
            return

        # Scan directory
        registered = []
        if path.is_file():
            try:
                asset = self.keeper.register_from_path(str(path), stronghold)
                registered.append(asset.asset_id)
            except Exception as e:
                logger.warning(f"Failed to register {path}: {e}")
        else:
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    try:
                        asset = self.keeper.register_from_path(str(file_path), stronghold)
                        registered.append(asset.asset_id)
                    except Exception as e:
                        logger.warning(f"Failed to register {file_path}: {e}")

        self._send_json({
            "scanned_path": str(path),
            "registered_count": len(registered),
            "asset_ids": registered,
        })

    # =========================================================================
    # ZONE FEDERATION HANDLERS
    # =========================================================================

    def _handle_list_zones(self):
        """List all zones with status."""
        if not self.zone_registry:
            self._send_error("Zone registry not initialized", 500)
            return

        summary = self.zone_registry.get_summary()
        self._send_json(summary)

    def _handle_refresh_zones(self):
        """Refresh status of all zones."""
        if not self.zone_registry:
            self._send_error("Zone registry not initialized", 500)
            return

        status = self.zone_registry.check_all_status()
        summary = self.zone_registry.get_summary()

        self._send_json({
            "refreshed": True,
            "status": {k: v.value for k, v in status.items()},
            "summary": summary,
        })

    def _handle_get_zone(self, zone_id: str):
        """Get specific zone details."""
        if not self.zone_registry:
            self._send_error("Zone registry not initialized", 500)
            return

        zone = self.zone_registry.get(zone_id)
        if not zone:
            self._send_error(f"Zone not found: {zone_id}", 404)
            return

        # Get live status from Branch Officer
        client = ZoneClient(zone)
        status_data = client.get_status()

        if status_data:
            zone_data = zone.to_dict()
            zone_data["live_status"] = status_data
            self._send_json(zone_data)
        else:
            zone_data = zone.to_dict()
            zone_data["live_status"] = None
            zone_data["offline"] = True
            self._send_json(zone_data)

    def _handle_zone_assets(self, zone_id: str, query: Dict[str, list]):
        """List assets in a specific zone."""
        if not self.zone_registry:
            self._send_error("Zone registry not initialized", 500)
            return

        zone = self.zone_registry.get(zone_id)
        if not zone:
            self._send_error(f"Zone not found: {zone_id}", 404)
            return

        # Query Branch Officer for assets
        client = ZoneClient(zone)
        asset_type = query.get("type", [None])[0]
        limit = int(query.get("limit", [100])[0])

        assets = client.list_assets(asset_type=asset_type, limit=limit)

        self._send_json({
            "zone_id": zone_id,
            "count": len(assets),
            "assets": assets,
        })

    def _handle_transfer(self, body: Optional[Dict[str, Any]]):
        """Transfer an asset between zones."""
        if not self.zone_registry:
            self._send_error("Zone registry not initialized", 500)
            return

        if not body:
            self._send_error("Missing request body", 400)
            return

        # Required fields
        asset_path = body.get("asset_path") or body.get("source_path")
        if not asset_path:
            self._send_error("Missing 'asset_path' or 'source_path'", 400)
            return

        target_zone = body.get("target_zone") or body.get("to_zone")
        if not target_zone:
            self._send_error("Missing 'target_zone' or 'to_zone'", 400)
            return

        # Optional fields
        source_zone = body.get("source_zone") or body.get("from_zone") or "4090"
        dest_path = body.get("dest_path") or body.get("destination")
        asset_id = body.get("asset_id")

        # Get zones
        source = self.zone_registry.get(source_zone)
        target = self.zone_registry.get(target_zone)

        if not source:
            self._send_error(f"Unknown source zone: {source_zone}", 400)
            return
        if not target:
            self._send_error(f"Unknown target zone: {target_zone}", 400)
            return

        # Execute transfer
        transfer = ZoneTransfer(source, target)
        result = transfer.push(asset_path, dest_path, asset_id)

        self._send_json(result)

    # =========================================================================
    # LEDGER API - Checkpoint history and stats
    # =========================================================================

    def _handle_ledger_list(self, query: Dict[str, list]):
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
            self._send_json({"checkpoints": [], "count": 0, "error": "Ledger module not available"})
        except Exception as e:
            logger.error(f"Ledger list error: {e}")
            self._send_error(str(e), 500)

    def _handle_ledger_summary(self):
        """Get ledger summary statistics."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            summary = ledger.get_summary()
            self._send_json(summary)

        except ImportError:
            self._send_error("Ledger module not available", 500)
        except Exception as e:
            logger.error(f"Ledger summary error: {e}")
            self._send_error(str(e), 500)

    def _handle_ledger_best(self, query: Dict[str, list]):
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
                self._send_error("No checkpoints with that metric", 404)

        except ImportError:
            self._send_error("Ledger module not available", 500)
        except Exception as e:
            logger.error(f"Ledger best error: {e}")
            self._send_error(str(e), 500)

    def _handle_ledger_get(self, step_str: str):
        """Get specific checkpoint info by step."""
        try:
            from core.checkpoint_ledger import get_ledger

            step = int(step_str)
            ledger = get_ledger()
            record = ledger.get(step)

            if record:
                self._send_json(record.to_dict())
            else:
                self._send_error(f"Checkpoint {step} not found in ledger", 404)

        except ValueError:
            self._send_error(f"Invalid step: {step_str}", 400)
        except ImportError:
            self._send_error("Ledger module not available", 500)
        except Exception as e:
            logger.error(f"Ledger get error: {e}")
            self._send_error(str(e), 500)

    # =========================================================================
    # TRAINING API - Status and control
    # =========================================================================

    # Base directory (set by run_server)
    base_dir: Path = None

    def _handle_training_status(self):
        """Get current training status."""
        try:
            from core.paths import get_base_dir
            base = self.base_dir or get_base_dir()
            status_file = base / "status" / "training_status.json"

            if not status_file.exists():
                self._send_json({
                    "status": "unknown",
                    "message": "Training status file not found",
                })
                return

            with open(status_file) as f:
                status = json.load(f)

            self._send_json(status)

        except Exception as e:
            logger.error(f"Training status error: {e}")
            self._send_error(str(e), 500)

    def _handle_training_queue(self):
        """Get training queue status."""
        try:
            from core.paths import get_base_dir
            base = self.base_dir or get_base_dir()
            queue_dirs = ["high", "normal", "low", "processing", "failed"]

            queue_status = {}
            for priority in queue_dirs:
                queue_path = base / "queue" / priority
                if queue_path.exists():
                    files = list(queue_path.glob("*.jsonl"))
                    queue_status[priority] = {
                        "count": len(files),
                        "files": [f.name for f in files[:10]],  # First 10
                    }
                else:
                    queue_status[priority] = {"count": 0, "files": []}

            # Calculate totals
            pending = sum(queue_status[p]["count"] for p in ["high", "normal", "low"])
            processing = queue_status["processing"]["count"]

            self._send_json({
                "pending": pending,
                "processing": processing,
                "failed": queue_status["failed"]["count"],
                "queues": queue_status,
            })

        except Exception as e:
            logger.error(f"Training queue error: {e}")
            self._send_error(str(e), 500)

    def _handle_training_control(self, body: Optional[Dict[str, Any]]):
        """Control training (pause/resume/stop)."""
        if not body or "action" not in body:
            self._send_error("Missing 'action' in request body", 400)
            return

        action = body["action"]
        if action not in ["pause", "resume", "stop"]:
            self._send_error(f"Invalid action: {action}. Use pause/resume/stop", 400)
            return

        try:
            from core.paths import get_base_dir
            base = self.base_dir or get_base_dir()
            control_dir = base / "control"
            control_dir.mkdir(parents=True, exist_ok=True)

            if action == "pause":
                (control_dir / ".pause").touch()
                (control_dir / ".stop").unlink(missing_ok=True)
                message = "Pause signal sent"
            elif action == "resume":
                (control_dir / ".pause").unlink(missing_ok=True)
                message = "Resume signal sent"
            elif action == "stop":
                (control_dir / ".stop").touch()
                (control_dir / ".pause").unlink(missing_ok=True)
                message = "Stop signal sent"

            self._send_json({
                "success": True,
                "action": action,
                "message": message,
            })

        except Exception as e:
            logger.error(f"Training control error: {e}")
            self._send_error(str(e), 500)

    # =========================================================================
    # EVALUATION API - Skill evaluation results
    # =========================================================================

    def _handle_evals_list(self, query: Dict[str, list]):
        """List all skill evaluations."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            limit = int(query.get("limit", [100])[0])
            records = ledger.list_all(limit=limit)

            self._send_json({
                "count": len(records),
                "evaluations": [r.to_dict() for r in records],
            })

        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals list error: {e}")
            self._send_error(str(e), 500)

    def _handle_evals_summary(self):
        """Get evaluation summary statistics."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            summary = ledger.summary()
            self._send_json(summary)

        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals summary error: {e}")
            self._send_error(str(e), 500)

    def _handle_evals_queue(self):
        """Get pending evaluation queue."""
        try:
            from core.evaluation_ledger import get_pending_evaluations

            pending = get_pending_evaluations()
            self._send_json({
                "count": len(pending),
                "queue": pending,
            })

        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals queue error: {e}")
            self._send_error(str(e), 500)

    def _handle_evals_by_checkpoint(self, step_str: str):
        """Get all evaluations for a checkpoint."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            step = int(step_str)
            ledger = get_eval_ledger()
            records = ledger.get_by_checkpoint(step)

            self._send_json({
                "checkpoint_step": step,
                "count": len(records),
                "evaluations": [r.to_dict() for r in records],
            })

        except ValueError:
            self._send_error(f"Invalid step: {step_str}", 400)
        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals by checkpoint error: {e}")
            self._send_error(str(e), 500)

    def _handle_evals_by_skill(self, skill: str, query: Dict[str, list]):
        """Get evaluations for a skill."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            ledger = get_eval_ledger()
            level = query.get("level", [None])[0]
            level = int(level) if level else None

            records = ledger.get_by_skill(skill, level)

            self._send_json({
                "skill": skill,
                "level": level,
                "count": len(records),
                "evaluations": [r.to_dict() for r in records],
            })

        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals by skill error: {e}")
            self._send_error(str(e), 500)

    def _handle_evals_best(self, query: Dict[str, list]):
        """Get best checkpoint for a skill/level."""
        try:
            from core.evaluation_ledger import get_eval_ledger

            skill = query.get("skill", [None])[0]
            level = query.get("level", [None])[0]

            if not skill or not level:
                self._send_error("Missing 'skill' and 'level' query params", 400)
                return

            ledger = get_eval_ledger()
            best = ledger.get_best(skill, int(level))

            if best:
                self._send_json(best.to_dict())
            else:
                self._send_error(f"No evaluations for {skill} L{level}", 404)

        except ImportError:
            self._send_error("Evaluation ledger module not available", 500)
        except Exception as e:
            logger.error(f"Evals best error: {e}")
            self._send_error(str(e), 500)

    # =========================================================================
    # PASSIVES API - Transfer learning evaluations
    # =========================================================================

    def _handle_passives_list(self, query: Dict[str, list]):
        """List all passive evaluations."""
        try:
            from core.passives import get_passives_ledger

            ledger = get_passives_ledger()
            mode = query.get("mode", [None])[0]  # "lite" or "full"

            # Get all results
            results = []
            for key, result in ledger._cache.items():
                if mode is None or result.mode == mode:
                    results.append(result.to_dict())

            # Sort by timestamp desc
            results.sort(key=lambda x: x["timestamp"], reverse=True)

            self._send_json({
                "count": len(results),
                "results": results[:100],  # Limit to 100
            })

        except ImportError:
            self._send_error("Passives module not available", 500)
        except Exception as e:
            logger.error(f"Passives list error: {e}")
            self._send_error(str(e), 500)

    def _handle_passives_summary(self):
        """Get passives summary statistics."""
        try:
            from core.passives import get_passives_ledger

            ledger = get_passives_ledger()
            summary = ledger.summary()
            self._send_json(summary)

        except ImportError:
            self._send_error("Passives module not available", 500)
        except Exception as e:
            logger.error(f"Passives summary error: {e}")
            self._send_error(str(e), 500)

    def _handle_passives_queue(self):
        """Get pending passive queue."""
        try:
            from core.passives import get_pending_passives

            pending = get_pending_passives()
            self._send_json({
                "count": len(pending),
                "queue": pending,
            })

        except ImportError:
            self._send_error("Passives module not available", 500)
        except Exception as e:
            logger.error(f"Passives queue error: {e}")
            self._send_error(str(e), 500)

    def _handle_passives_by_checkpoint(self, step_str: str):
        """Get passive results for a checkpoint."""
        try:
            from core.passives import get_passives_ledger

            step = int(step_str)
            ledger = get_passives_ledger()
            summary = ledger.get_checkpoint_summary(step)

            self._send_json(summary)

        except ValueError:
            self._send_error(f"Invalid step: {step_str}", 400)
        except ImportError:
            self._send_error("Passives module not available", 500)
        except Exception as e:
            logger.error(f"Passives by checkpoint error: {e}")
            self._send_error(str(e), 500)

    # =========================================================================
    # BATTLE LOG API - MMO-style event stream
    # =========================================================================

    def _handle_battle_log(self, query: Dict[str, list]):
        """Get events from the battle log."""
        try:
            from core.battle_log import get_battle_logger, CHANNEL_ICONS, SEVERITY_COLORS

            blog = get_battle_logger()

            # Parse query params
            channels_str = query.get("channels", [None])[0]
            channels = channels_str.split(",") if channels_str else None
            since = query.get("since", [None])[0]
            limit = int(query.get("limit", [50])[0])
            limit = min(limit, 200)  # Cap at 200
            hero_id = query.get("hero", [None])[0]
            campaign_id = query.get("campaign", [None])[0]
            severity = query.get("severity", [None])[0]

            events = blog.get_events(
                channels=channels,
                since=since,
                limit=limit,
                hero_id=hero_id,
                campaign_id=campaign_id,
                severity=severity,
            )

            # Build response
            response = {
                "events": [e.to_dict() for e in events],
                "count": len(events),
            }

            # Include next_since for polling
            if events:
                response["next_since"] = events[0].timestamp

            # Include channel counts
            response["channel_counts"] = blog.get_channel_counts(hours=24)

            self._send_json(response)

        except ImportError:
            self._send_json({
                "events": [],
                "count": 0,
                "error": "Battle log module not available",
            })
        except Exception as e:
            logger.error(f"Battle log error: {e}")
            self._send_error(str(e), 500)

    def _handle_battle_log_channels(self):
        """Get available channels and their metadata."""
        from core.battle_log import CHANNEL_ICONS, SEVERITY_COLORS

        channels = {
            "system": {
                "name": "System",
                "icon": CHANNEL_ICONS.get("system", "⚙️"),
                "description": "Server events, errors, config changes",
            },
            "jobs": {
                "name": "Jobs",
                "icon": CHANNEL_ICONS.get("jobs", "⚔️"),
                "description": "Job lifecycle - claims, completions, failures",
            },
            "training": {
                "name": "Training",
                "icon": CHANNEL_ICONS.get("training", "📈"),
                "description": "Checkpoints, LR changes, milestones",
            },
            "eval": {
                "name": "Eval",
                "icon": CHANNEL_ICONS.get("eval", "📊"),
                "description": "Evaluation results, regressions",
            },
            "vault": {
                "name": "Vault",
                "icon": CHANNEL_ICONS.get("vault", "🗃️"),
                "description": "Archive and sync operations",
            },
            "guild": {
                "name": "Guild",
                "icon": CHANNEL_ICONS.get("guild", "🏰"),
                "description": "Titles, lore, hero progression",
            },
            "debug": {
                "name": "Debug",
                "icon": CHANNEL_ICONS.get("debug", "🔧"),
                "description": "Developer-only events",
            },
        }

        self._send_json({
            "channels": channels,
            "severities": SEVERITY_COLORS,
        })

    # =========================================================================
    # JOBS API - Distributed job execution
    # =========================================================================

    def _get_job_store(self):
        """Get the job store, initializing if needed."""
        if self.job_store is None:
            from jobs.store import get_store
            self.job_store = get_store()
        return self.job_store

    def _handle_jobs_list(self, query: Dict[str, list]):
        """List jobs with optional filters."""
        try:
            store = self._get_job_store()

            # Parse query params
            status = query.get("status", [None])[0]
            job_type = query.get("type", [None])[0]
            device_id = query.get("device", [None])[0]
            limit = int(query.get("limit", [100])[0])
            offset = int(query.get("offset", [0])[0])

            # Convert to enums if provided
            from guild.job_types import JobStatus, JobType
            status_enum = JobStatus(status) if status else None
            type_enum = JobType(job_type) if job_type else None

            jobs = store.list_jobs(
                status=status_enum,
                job_type=type_enum,
                device_id=device_id,
                limit=limit,
                offset=offset,
            )

            self._send_json({
                "count": len(jobs),
                "jobs": [j.to_dict() for j in jobs],
            })

        except Exception as e:
            logger.error(f"Jobs list error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_stats(self):
        """Get job statistics."""
        try:
            store = self._get_job_store()
            stats = store.get_stats()
            self._send_json(stats)

        except Exception as e:
            logger.error(f"Jobs stats error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_get(self, job_id: str):
        """Get a specific job."""
        try:
            store = self._get_job_store()
            job = store.get(job_id)

            if job:
                self._send_json(job.to_dict())
            else:
                self._send_error(f"Job not found: {job_id}", 404)

        except Exception as e:
            logger.error(f"Jobs get error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_submit(self, body: Optional[Dict[str, Any]]):
        """Submit a new job with validation and backpressure."""
        if not body:
            self._send_error("Missing request body", 400)
            return

        try:
            from guild.job_types import Job, JobSpec

            # Parse job spec
            if "spec" in body:
                spec_data = body["spec"]
            else:
                spec_data = body

            # Extract job type for validation
            job_type = spec_data.get("job_type")
            if not job_type:
                self._send_json({
                    "accepted": False,
                    "reason": "payload_invalid",
                    "message": "Missing job_type field",
                }, 400)
                return

            # Validate job type exists
            try:
                config = validate_job_type(job_type)
            except ValueError as e:
                self._send_json({
                    "accepted": False,
                    "reason": "payload_invalid",
                    "message": str(e),
                }, 400)
                return

            # Validate payload
            payload = spec_data.get("payload", {})
            try:
                warnings = validate_payload(job_type, payload)
            except ValueError as e:
                self._send_json({
                    "accepted": False,
                    "reason": "payload_invalid",
                    "message": str(e),
                }, 400)
                return

            # Check backpressure - use efficient count_jobs() helper
            store = self._get_job_store()
            pending = store.count_jobs(JobStatus.PENDING, job_type)
            running = store.count_jobs(JobStatus.RUNNING, job_type)

            can_accept, reason, limit_warning = check_queue_limits(job_type, pending, running)

            if not can_accept:
                self._send_json({
                    "accepted": False,
                    "reason": reason,
                    "message": limit_warning,
                    "pending": pending,
                    "running": running,
                    "max_pending": config.max_pending,
                    "max_running": config.max_running,
                    "retry_after_sec": 30,
                }, 429)
                return

            # Create and submit job
            spec = JobSpec.from_dict(spec_data)
            job = Job.create(spec)
            store.submit(job)

            # Build response
            response = {
                "accepted": True,
                "job_id": job.job_id,
                "status": job.status.value,
                "queue_position": pending + 1,
            }

            # Add warnings if any
            all_warnings = []
            if warnings:
                all_warnings.extend(warnings)
            if limit_warning:
                all_warnings.append(limit_warning)
            if all_warnings:
                response["warnings"] = all_warnings

            self._send_json(response, 201)

        except Exception as e:
            logger.error(f"Jobs submit error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_claim(self, body: Optional[Dict[str, Any]]):
        """
        Claim the next available job (worker pull).

        Supports two modes:
        1. worker_id mode (preferred): Server looks up roles from workers table
           - Uses smart routing based on resource_class, capabilities
           - Enforces max_concurrent_jobs limit
        2. Legacy mode: Client provides device_id + roles directly
           - Falls back to basic role-based routing

        Using worker_id mode enables heterogeneous cluster routing.
        """
        if not body:
            self._send_error("Missing request body", 400)
            return

        worker_id = body.get("worker_id")
        device_id = body.get("device_id")
        roles = body.get("roles", [])
        lease_duration = body.get("lease_duration", 300)
        use_smart_routing = body.get("smart_routing", True)  # Default to smart routing

        store = self._get_job_store()

        # Prefer worker_id mode - look up full worker info
        if worker_id:
            worker = store.get_worker(worker_id)
            if not worker:
                self._send_json({
                    "claimed": False,
                    "error": f"Worker not registered: {worker_id}",
                    "should_register": True,
                }, 404)
                return

            # Check load limits (max_concurrent_jobs)
            max_jobs = worker.get("max_concurrent_jobs", 1)
            active_jobs = worker.get("active_jobs", 0)
            if active_jobs >= max_jobs:
                self._send_json({
                    "claimed": False,
                    "message": "Worker at capacity",
                    "active_jobs": active_jobs,
                    "max_concurrent_jobs": max_jobs,
                })
                return

            # Use smart routing if worker has HC fields
            if use_smart_routing and worker.get("resource_class"):
                try:
                    job = store.claim_next_smart(worker, lease_duration)
                except Exception as e:
                    logger.warning(f"Smart routing failed, falling back: {e}")
                    job = store.claim_next(
                        worker.get("device_id"),
                        worker.get("roles", []),
                        lease_duration
                    )
            else:
                # Fall back to basic routing
                job = store.claim_next(
                    worker.get("device_id"),
                    worker.get("roles", []),
                    lease_duration
                )

            if job:
                self._send_json({
                    "claimed": True,
                    "job": job.to_dict(),
                    "routing": "smart" if worker.get("resource_class") else "basic",
                })
            else:
                self._send_json({
                    "claimed": False,
                    "message": "No jobs available for your capabilities",
                    "routing": "smart" if worker.get("resource_class") else "basic",
                })
            return

        # Legacy mode: device_id + roles
        if not device_id:
            self._send_error("Missing 'device_id' or 'worker_id' in request body", 400)
            return

        if not roles:
            self._send_error("Missing 'roles' in request body (or use 'worker_id')", 400)
            return

        try:
            job = store.claim_next(device_id, roles, lease_duration)

            if job:
                self._send_json({
                    "claimed": True,
                    "job": job.to_dict(),
                    "routing": "legacy",
                })
            else:
                self._send_json({
                    "claimed": False,
                    "message": "No jobs available for your roles",
                    "routing": "legacy",
                })

        except Exception as e:
            logger.error(f"Jobs claim error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_running(self, job_id: str, body: Optional[Dict[str, Any]]):
        """Mark a job as running."""
        device_id = body.get("device_id") if body else None

        if not device_id:
            self._send_error("Missing 'device_id' in request body", 400)
            return

        try:
            store = self._get_job_store()
            success = store.mark_running(job_id, device_id)

            if success:
                self._send_json({"success": True, "status": "running"})
            else:
                self._send_error("Cannot mark job as running", 400)

        except Exception as e:
            logger.error(f"Jobs running error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_complete(self, job_id: str, body: Optional[Dict[str, Any]]):
        """Mark a job as completed."""
        result = body.get("result", {}) if body else {}

        try:
            store = self._get_job_store()
            success = store.mark_complete(job_id, result)

            if success:
                self._send_json({"success": True, "status": "completed"})
            else:
                self._send_error("Cannot mark job as completed", 400)

        except Exception as e:
            logger.error(f"Jobs complete error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_failed(self, job_id: str, body: Optional[Dict[str, Any]]):
        """Mark a job as failed with structured error code."""
        error = body.get("error", "Unknown error") if body else "Unknown error"

        # Parse error code (defaults to UNKNOWN)
        error_code_str = body.get("error_code", "unknown") if body else "unknown"
        try:
            error_code = JobErrorCode(error_code_str)
        except ValueError:
            error_code = JobErrorCode.UNKNOWN

        try:
            store = self._get_job_store()
            success = store.mark_failed(job_id, error, error_code)

            if success:
                # Check if job was returned to queue for retry
                job = store.get(job_id)
                status = job.status.value if job else "unknown"
                self._send_json({
                    "success": True,
                    "status": status,
                    "error_code": error_code.value,
                    "retryable": error_code.is_retryable,
                    "message": "Returned to queue for retry" if status == "pending" else "Failed permanently",
                })
            else:
                self._send_error("Cannot mark job as failed", 400)

        except Exception as e:
            logger.error(f"Jobs failed error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_cancel(self, job_id: str):
        """Cancel a job."""
        try:
            store = self._get_job_store()
            success = store.cancel(job_id)

            if success:
                self._send_json({"success": True, "status": "cancelled"})
            else:
                self._send_error("Cannot cancel job", 400)

        except Exception as e:
            logger.error(f"Jobs cancel error: {e}")
            self._send_error(str(e), 500)

    def _handle_jobs_release(self, job_id: str):
        """Release a claimed job back to the queue."""
        try:
            store = self._get_job_store()
            success = store.release(job_id)

            if success:
                self._send_json({"success": True, "status": "pending"})
            else:
                self._send_error("Cannot release job", 400)

        except Exception as e:
            logger.error(f"Jobs release error: {e}")
            self._send_error(str(e), 500)

    # =========================================================================
    # WORKER HANDLERS
    # =========================================================================

    def _handle_worker_register(self, body: Optional[Dict[str, Any]]):
        """Register a worker with role validation against devices.json."""
        if not body:
            self._send_error("Missing request body", 400)
            return

        device_id = body.get("device_id")
        worker_kind = body.get("worker_kind", "claiming")
        roles = body.get("roles", [])
        version = body.get("version")
        hostname = body.get("hostname")

        if not device_id:
            self._send_json({
                "registered": False,
                "error": "Missing device_id",
            }, 400)
            return

        if not roles:
            self._send_json({
                "registered": False,
                "error": "Missing roles",
            }, 400)
            return

        # Validate device_id and roles against devices.json, extract HC fields
        resource_class = None
        priority_class = None
        max_concurrent_jobs = 1
        capabilities = []
        gpus = []

        try:
            from core.paths import get_base_dir
            devices_path = get_base_dir() / "config" / "devices.json"
            if devices_path.exists():
                with open(devices_path) as f:
                    devices_config = json.load(f)

                devices = devices_config.get("devices", {})
                if device_id not in devices:
                    self._send_json({
                        "registered": False,
                        "error": f"Unknown device: {device_id}",
                        "hint": "Device must be registered in config/devices.json",
                    }, 403)
                    return

                device_cfg = devices[device_id]
                if not device_cfg.get("enabled", True):
                    self._send_json({
                        "registered": False,
                        "error": f"Device {device_id} is disabled",
                    }, 403)
                    return

                # Validate roles are subset of device's allowed roles
                allowed_roles = set(device_cfg.get("roles", []))
                requested_roles = set(roles)
                invalid_roles = requested_roles - allowed_roles

                if invalid_roles:
                    self._send_json({
                        "registered": False,
                        "error": f"Roles not allowed for device {device_id}: {sorted(invalid_roles)}",
                        "allowed_roles": sorted(allowed_roles),
                    }, 403)
                    return

                # Use the validated roles (intersection with allowed)
                roles = list(requested_roles & allowed_roles)

                # Extract HC fields from device config
                resource_class = device_cfg.get("resource_class")
                priority_class = device_cfg.get("priority_class")
                max_concurrent_jobs = device_cfg.get("max_concurrent_jobs", 1)
                capabilities = device_cfg.get("capabilities", [])
                gpus = device_cfg.get("gpus", [])

        except Exception as e:
            # Log but don't block registration if devices.json can't be read
            logger.warning(f"Could not validate device roles: {e}")

        try:
            # Get client IP from request
            client_ip = self.client_address[0] if self.client_address else None

            # Construct worker ID
            worker_id = f"{device_id}.{worker_kind}"

            store = self._get_job_store()
            result = store.register_worker(
                worker_id=worker_id,
                device_id=device_id,
                worker_kind=worker_kind,
                roles=roles,
                version=version,
                hostname=hostname,
                client_ip=client_ip,
                # HC fields from devices.json
                resource_class=resource_class,
                priority_class=priority_class,
                max_concurrent_jobs=max_concurrent_jobs,
                capabilities=capabilities,
                gpus=gpus,
            )

            self._send_json(result)

        except Exception as e:
            logger.error(f"Worker register error: {e}")
            self._send_error(str(e), 500)

    def _handle_worker_heartbeat(self, body: Optional[Dict[str, Any]]):
        """Update worker heartbeat."""
        if not body:
            self._send_error("Missing request body", 400)
            return

        worker_id = body.get("worker_id")
        active_jobs = body.get("active_jobs", 0)
        status = body.get("status", "online")

        if not worker_id:
            self._send_json({
                "acknowledged": False,
                "error": "Missing worker_id",
            }, 400)
            return

        try:
            store = self._get_job_store()
            success = store.heartbeat_worker(worker_id, active_jobs, status)

            if success:
                self._send_json({
                    "acknowledged": True,
                    "server_time": datetime.now().isoformat(),
                })
            else:
                # Worker not found - tell them to register
                self._send_json({
                    "acknowledged": False,
                    "error": "Worker not registered",
                    "should_register": True,
                }, 404)

        except Exception as e:
            logger.error(f"Worker heartbeat error: {e}")
            self._send_error(str(e), 500)

    def _handle_workers_list(self):
        """List all workers."""
        try:
            store = self._get_job_store()
            workers = store.list_workers()
            stats = store.get_worker_stats()

            # Get allowed job types for each worker
            from jobs.registry import get_allowed_job_types
            for w in workers:
                w["allowed_job_types"] = get_allowed_job_types(w["roles"])

            self._send_json({
                "workers": workers,
                "summary": stats,
            })

        except Exception as e:
            logger.error(f"Workers list error: {e}")
            self._send_error(str(e), 500)

    def _handle_cluster_status(self):
        """
        Get heterogeneous cluster status.

        Returns cluster mode, queue depths, and worker summary with resource classes.
        """
        try:
            store = self._get_job_store()

            # Get queue stats
            queue_stats = store.get_queue_stats()

            # Compute cluster mode
            try:
                from jobs.routing import compute_cluster_mode
                cluster_mode = compute_cluster_mode(queue_stats)
            except ImportError:
                cluster_mode = "unknown"

            # Get workers
            workers = store.list_workers()
            online_workers = [w for w in workers if w.get("status") == "online"]

            # Group by resource class
            by_resource_class = {}
            for w in online_workers:
                rc = w.get("resource_class") or "unknown"
                if rc not in by_resource_class:
                    by_resource_class[rc] = {"count": 0, "active_jobs": 0, "max_jobs": 0}
                by_resource_class[rc]["count"] += 1
                by_resource_class[rc]["active_jobs"] += w.get("active_jobs", 0)
                by_resource_class[rc]["max_jobs"] += w.get("max_concurrent_jobs", 1)

            # Group by priority class
            by_priority_class = {}
            for w in online_workers:
                pc = w.get("priority_class") or "unknown"
                if pc not in by_priority_class:
                    by_priority_class[pc] = {"count": 0, "active_jobs": 0}
                by_priority_class[pc]["count"] += 1
                by_priority_class[pc]["active_jobs"] += w.get("active_jobs", 0)

            # Calculate total capacity
            total_capacity = sum(w.get("max_concurrent_jobs", 1) for w in online_workers)
            total_active = sum(w.get("active_jobs", 0) for w in online_workers)

            # Calculate queue depths
            total_pending = sum(
                stats.get("pending", 0)
                for stats in queue_stats.values()
            )

            self._send_json({
                "mode": cluster_mode,
                "queue_depths": queue_stats,
                "workers_summary": {
                    "total": len(workers),
                    "online": len(online_workers),
                    "by_resource_class": by_resource_class,
                    "by_priority_class": by_priority_class,
                    "total_capacity": total_capacity,
                    "total_active": total_active,
                    "utilization_pct": round(100 * total_active / total_capacity, 1) if total_capacity > 0 else 0,
                },
                "health": {
                    "total_pending": total_pending,
                    "cluster_mode": cluster_mode,
                    "mode_reason": self._get_mode_reason(cluster_mode, queue_stats),
                },
            })

        except Exception as e:
            logger.error(f"Cluster status error: {e}")
            self._send_error(str(e), 500)

    def _get_mode_reason(self, mode: str, queue_stats: dict) -> str:
        """Get human-readable reason for current cluster mode."""
        if mode == "catch_up":
            # Find which job types have high backlogs
            high_backlogs = [
                jt for jt, stats in queue_stats.items()
                if stats.get("pending", 0) > 10
            ]
            if high_backlogs:
                return f"High backlog in: {', '.join(high_backlogs)}"
            return "Critical job backlog detected"
        elif mode == "idle":
            return "No critical or high-priority work pending"
        else:
            return "Balanced operation"

    def _handle_jobs_health(self):
        """Job system health check."""
        try:
            store = self._get_job_store()
            stats = store.get_stats()
            worker_stats = store.get_worker_stats()

            # Build health checks
            checks = {}
            alerts = []

            # Database check
            checks["database"] = {"ok": True, "latency_ms": 1}

            # Workers check
            online_workers = worker_stats.get("online", 0)
            total_workers = worker_stats.get("total", 0)
            checks["workers"] = {
                "ok": online_workers > 0,
                "online": online_workers,
                "offline": worker_stats.get("offline", 0),
                "total": total_workers,
            }
            if online_workers == 0 and total_workers > 0:
                alerts.append({
                    "level": "warning",
                    "message": "No workers online",
                })

            # Queue check
            queue_depth = stats.get("queue_depth", 0)
            checks["queue"] = {
                "ok": queue_depth < 100,
                "depth": queue_depth,
            }
            if queue_depth > 100:
                alerts.append({
                    "level": "warning",
                    "message": f"Queue depth high: {queue_depth}",
                })

            # Error rate check
            errors_24h = stats.get("errors_24h", {})
            error_rate = errors_24h.get("error_rate", 0)
            checks["errors"] = {
                "ok": error_rate < 0.15,
                "error_rate_24h": error_rate,
                "failed_24h": errors_24h.get("total", 0),
            }
            if error_rate > 0.15:
                alerts.append({
                    "level": "warning",
                    "message": f"High error rate: {error_rate*100:.1f}%",
                })

            # Overall status
            all_ok = all(c.get("ok", True) for c in checks.values())
            status = "healthy" if all_ok else ("degraded" if alerts else "unhealthy")

            self._send_json({
                "status": status,
                "checks": checks,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"Jobs health error: {e}")
            self._send_json({
                "status": "unhealthy",
                "checks": {"database": {"ok": False, "error": str(e)}},
                "alerts": [{"level": "error", "message": str(e)}],
            }, 500)

    def _handle_job_events(self, job_id: str):
        """Get events for a specific job."""
        try:
            store = self._get_job_store()

            # Check if job exists
            job = store.get(job_id)
            if not job:
                self._send_error(f"Job not found: {job_id}", 404)
                return

            events = store.get_events(job_id)

            self._send_json({
                "job_id": job_id,
                "events": [e.to_dict() for e in events],
                "count": len(events),
            })

        except Exception as e:
            logger.error(f"Error getting job events: {e}")
            self._send_error(str(e), 500)

    def _handle_recent_events(self, query: Dict[str, list]):
        """Get recent events across all jobs."""
        try:
            store = self._get_job_store()

            # parse_qs returns lists, get first value
            limit_list = query.get("limit", ["50"])
            limit = int(limit_list[0] if isinstance(limit_list, list) else limit_list)
            limit = min(limit, 200)  # Cap at 200

            events = store.get_recent_events(limit=limit)

            self._send_json({
                "events": [e.to_dict() for e in events],
                "count": len(events),
            })

        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            self._send_error(str(e), 500)


def run_server(port: int = 8767, base_dir: Optional[str] = None):
    """Run the VaultKeeper server."""
    if base_dir is None:
        from core.paths import get_base_dir
        base_dir = str(get_base_dir())
    base_path = Path(base_dir)

    # Initialize keeper
    keeper = VaultKeeper(base_dir)
    VaultKeeperHandler.keeper = keeper

    # Initialize zone registry
    zone_registry = ZoneRegistry()
    VaultKeeperHandler.zone_registry = zone_registry

    # Set base directory for training API
    VaultKeeperHandler.base_dir = base_path

    # Initialize job store and start maintenance worker
    try:
        from jobs.store import get_store, StoreMaintenanceWorker
        job_store = get_store()
        VaultKeeperHandler.job_store = job_store

        # Start background maintenance (expire stale leases, cleanup old jobs)
        maintenance = StoreMaintenanceWorker(job_store)
        maintenance.start()
        logger.info(f"Jobs DB: {job_store.db_path}")
    except ImportError as e:
        logger.warning(f"Job store not available: {e}")

    # Start server - use ThreadingHTTPServer to handle concurrent requests without blocking
    server = ThreadingHTTPServer(("0.0.0.0", port), VaultKeeperHandler)
    logger.info(f"VaultKeeper server starting on port {port}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Catalog: {keeper.catalog_path}")
    logger.info(f"Zones: {[z.zone_id for z in zone_registry.list()]}")

    # Battle Log - server started event
    try:
        from core.battle_log import log_system
        zone_names = [z.zone_id for z in zone_registry.list()]
        log_system(
            f"VaultKeeper started on port {port}",
            severity="success",
            source="vault.server",
            details={
                "port": port,
                "base_dir": str(base_dir),
                "zones": zone_names,
                "job_store": job_store is not None,
            },
        )
    except Exception:
        pass  # Don't let battle log errors affect startup

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="VaultKeeper API Server")
    parser.add_argument("--port", type=int, default=8767, help="Port to listen on")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base training directory (default: auto-detect)",
    )
    args = parser.parse_args()

    run_server(port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
