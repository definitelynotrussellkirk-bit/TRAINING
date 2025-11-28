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

    # Start server - use ThreadingHTTPServer to handle concurrent requests without blocking
    server = ThreadingHTTPServer(("0.0.0.0", port), VaultKeeperHandler)
    logger.info(f"VaultKeeper server starting on port {port}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Catalog: {keeper.catalog_path}")
    logger.info(f"Zones: {[z.zone_id for z in zone_registry.list()]}")

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
