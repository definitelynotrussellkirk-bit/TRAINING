"""
Branch Officer - Lightweight asset daemon for remote zones.

The Branch Officer is a sentinel posted at each remote stronghold (3090, NAS).
It tracks local assets and responds to queries from the central Vault.

Responsibilities:
    - Track local assets (checkpoints, models, data)
    - Expose /status endpoint for health checks
    - Expose /assets endpoint for asset inventory
    - Serve files via /fetch endpoint
    - Accept pushed files via /receive endpoint

Usage:
    # On inference server
    python3 vault/branch_officer.py --zone 3090 --port 8768 --base-dir ~/llm/models

    # On NAS (if SSH accessible)
    python3 vault/branch_officer.py --zone nas --port 8768 --base-dir /volume1/data/llm_training

    # Query from 4090
    curl http://inference.local:8768/status
    curl http://inference.local:8768/assets

RPG Flavor:
    Branch Officers are loyal sentinels stationed at distant strongholds.
    They keep watch over local treasures and report back to the central Vault.
    When the Vault needs something, the Officer fetches it; when the Vault
    sends something, the Officer stores it safely.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Version
BRANCH_OFFICER_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("branch_officer")


# =============================================================================
# LOCAL ASSET CATALOG
# =============================================================================

class LocalCatalog:
    """
    SQLite-based catalog of local assets.

    Tracks all assets at this branch location.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT,
                    created_at TEXT,
                    modified_at TEXT,
                    scanned_at TEXT,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
                CREATE INDEX IF NOT EXISTS idx_assets_path ON assets(path);

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

    def _get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def register_asset(
        self,
        asset_id: str,
        asset_type: str,
        name: str,
        path: str,
        size_bytes: int = 0,
        checksum: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Register an asset in the local catalog."""
        now = datetime.now().isoformat()

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO assets
                    (asset_id, asset_type, name, path, size_bytes, checksum,
                     created_at, modified_at, scanned_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        asset_type,
                        name,
                        path,
                        size_bytes,
                        checksum,
                        now,
                        now,
                        now,
                        json.dumps(metadata or {}),
                    )
                )
                conn.commit()

    def remove_asset(self, asset_id: str):
        """Remove an asset from catalog."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM assets WHERE asset_id = ?", (asset_id,))
                conn.commit()

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        """Get asset by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?",
                (asset_id,)
            ).fetchone()

            if row:
                return self._row_to_dict(row)
            return None

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """List assets in catalog."""
        with self._get_conn() as conn:
            if asset_type:
                rows = conn.execute(
                    "SELECT * FROM assets WHERE asset_type = ? ORDER BY modified_at DESC LIMIT ?",
                    (asset_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM assets ORDER BY modified_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get catalog statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM assets").fetchone()["cnt"]

            by_type = {}
            for row in conn.execute(
                "SELECT asset_type, COUNT(*) as cnt, SUM(size_bytes) as size FROM assets GROUP BY asset_type"
            ):
                by_type[row["asset_type"]] = {
                    "count": row["cnt"],
                    "size_bytes": row["size"] or 0,
                }

            total_size = conn.execute(
                "SELECT SUM(size_bytes) as size FROM assets"
            ).fetchone()["size"] or 0

            return {
                "total_assets": total,
                "assets_by_type": by_type,
                "total_size_bytes": total_size,
                "total_size_gb": round(total_size / (1024**3), 2),
            }

    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dict."""
        return {
            "asset_id": row["asset_id"],
            "asset_type": row["asset_type"],
            "name": row["name"],
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "size_gb": round(row["size_bytes"] / (1024**3), 2),
            "checksum": row["checksum"],
            "created_at": row["created_at"],
            "modified_at": row["modified_at"],
            "scanned_at": row["scanned_at"],
            "metadata": json.loads(row["metadata"] or "{}"),
        }


# =============================================================================
# ASSET SCANNER
# =============================================================================

class AssetScanner:
    """
    Scans directories for assets to register.
    """

    # Asset type detection patterns
    PATTERNS = {
        "checkpoint": ["checkpoint-*", "checkpoint_*"],
        "model": ["*model*", "Qwen*", "qwen*", "llama*", "Llama*"],
        "base_model": ["Qwen3-*", "Llama-*", "Mistral-*"],
        "training_data": ["*.jsonl"],
        "config": ["config*.json", "*.yaml", "*.yml"],
    }

    def __init__(self, catalog: LocalCatalog, base_dir: Path):
        self.catalog = catalog
        self.base_dir = base_dir

    def scan(self, paths: Optional[List[str]] = None) -> Dict:
        """
        Scan for assets.

        Args:
            paths: Specific paths to scan (or scan base_dir)

        Returns:
            Scan results with counts
        """
        scan_paths = [Path(p) for p in paths] if paths else [self.base_dir]

        registered = 0
        skipped = 0
        errors = []

        for scan_path in scan_paths:
            if not scan_path.exists():
                errors.append(f"Path not found: {scan_path}")
                continue

            # Scan for checkpoints (directories)
            for pattern in self.PATTERNS["checkpoint"]:
                for item in scan_path.glob(f"**/{pattern}"):
                    if item.is_dir():
                        try:
                            self._register_checkpoint(item)
                            registered += 1
                        except Exception as e:
                            errors.append(f"{item}: {e}")

            # Scan for models (directories with model files)
            for item in scan_path.iterdir():
                if item.is_dir() and self._is_model_dir(item):
                    try:
                        self._register_model(item)
                        registered += 1
                    except Exception as e:
                        errors.append(f"{item}: {e}")

            # Scan for data files
            for item in scan_path.glob("**/*.jsonl"):
                try:
                    self._register_data(item)
                    registered += 1
                except Exception as e:
                    errors.append(f"{item}: {e}")

        return {
            "registered": registered,
            "skipped": skipped,
            "errors": errors,
            "scanned_paths": [str(p) for p in scan_paths],
        }

    def _is_model_dir(self, path: Path) -> bool:
        """Check if directory contains a model."""
        model_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        ]
        return any((path / f).exists() for f in model_files)

    def _register_checkpoint(self, path: Path):
        """Register a checkpoint directory."""
        name = path.name

        # Extract step number (handles canonical names like checkpoint-190000-20251127-1430)
        try:
            from core.checkpoint_ledger import extract_step
            step = extract_step(name)
            if step == 0:
                step = None
        except ImportError:
            # Fallback if running standalone without core module
            step_str = name.replace("checkpoint-", "").replace("checkpoint_", "").split("-")[0]
            try:
                step = int(step_str)
            except ValueError:
                step = None

        # Calculate size
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        asset_id = f"checkpoint_{step}" if step else f"checkpoint_{name}"

        self.catalog.register_asset(
            asset_id=asset_id,
            asset_type="checkpoint",
            name=name,
            path=str(path),
            size_bytes=size,
            metadata={"step_number": step} if step else {},
        )

    def _register_model(self, path: Path):
        """Register a model directory."""
        name = path.name
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        # Determine if base model
        is_base = any(x in name for x in ["Qwen3", "Llama", "Mistral", "base"])
        asset_type = "base_model" if is_base else "model"

        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        asset_id = f"model_{clean_name}"

        self.catalog.register_asset(
            asset_id=asset_id,
            asset_type=asset_type,
            name=name,
            path=str(path),
            size_bytes=size,
            metadata={"model_name": name},
        )

    def _register_data(self, path: Path):
        """Register a data file."""
        name = path.name
        size = path.stat().st_size

        # Hash for unique ID
        path_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
        asset_id = f"data_{path.stem}_{path_hash}"

        self.catalog.register_asset(
            asset_id=asset_id,
            asset_type="training_data",
            name=name,
            path=str(path),
            size_bytes=size,
        )


# =============================================================================
# HTTP HANDLER
# =============================================================================

class BranchOfficerHandler(BaseHTTPRequestHandler):
    """HTTP handler for Branch Officer API."""

    # Class-level references (set by server)
    zone_id: str = ""
    base_dir: Path = None
    catalog: LocalCatalog = None
    scanner: AssetScanner = None

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message, "status": status}, status)

    def _send_file(self, path: Path):
        """Send a file as binary response."""
        if not path.exists():
            self._send_error(f"File not found: {path}", 404)
            return

        size = path.stat().st_size

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def _parse_body(self) -> Optional[Dict]:
        """Parse JSON body."""
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/status" or path == "/health":
            self._handle_status()
        elif path == "/assets":
            self._handle_list_assets(query)
        elif path.startswith("/assets/"):
            asset_id = path.replace("/assets/", "")
            self._handle_get_asset(asset_id)
        elif path.startswith("/fetch/"):
            asset_id = path.replace("/fetch/", "")
            self._handle_fetch(asset_id)
        elif path == "/stats":
            self._handle_stats()
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._parse_body()

        if path == "/scan":
            self._handle_scan(body)
        elif path == "/receive":
            self._handle_receive(body)
        elif path == "/register":
            self._handle_register(body)
        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def _handle_status(self):
        """Return zone status and health."""
        stats = self.catalog.get_stats()

        # Check disk space
        try:
            disk = shutil.disk_usage(self.base_dir)
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            disk_used_pct = (disk.used / disk.total) * 100
        except Exception:
            disk_free_gb = 0
            disk_total_gb = 0
            disk_used_pct = 0

        self._send_json({
            "status": "online",
            "zone_id": self.zone_id,
            "version": BRANCH_OFFICER_VERSION,
            "timestamp": datetime.now().isoformat(),
            "base_dir": str(self.base_dir),
            "catalog": stats,
            "disk": {
                "free_gb": round(disk_free_gb, 1),
                "total_gb": round(disk_total_gb, 1),
                "used_pct": round(disk_used_pct, 1),
            },
        })

    def _handle_list_assets(self, query: Dict):
        """List assets in catalog."""
        asset_type = query.get("type", [None])[0]
        limit = int(query.get("limit", [100])[0])

        assets = self.catalog.list_assets(asset_type=asset_type, limit=limit)

        self._send_json({
            "zone_id": self.zone_id,
            "count": len(assets),
            "assets": assets,
        })

    def _handle_get_asset(self, asset_id: str):
        """Get specific asset details."""
        asset = self.catalog.get_asset(asset_id)

        if asset:
            self._send_json({
                "zone_id": self.zone_id,
                "asset": asset,
            })
        else:
            self._send_error(f"Asset not found: {asset_id}", 404)

    def _handle_fetch(self, asset_id: str):
        """
        Fetch an asset file.

        For directories (checkpoints/models), this tars them on-the-fly.
        For files, serves directly.
        """
        asset = self.catalog.get_asset(asset_id)

        if not asset:
            self._send_error(f"Asset not found: {asset_id}", 404)
            return

        path = Path(asset["path"])

        if not path.exists():
            self._send_error(f"Asset path not found: {path}", 404)
            return

        if path.is_file():
            self._send_file(path)
        else:
            # For directories, we'll return info about how to rsync
            # (actual transfer should use rsync for efficiency)
            self._send_json({
                "asset_id": asset_id,
                "type": "directory",
                "path": str(path),
                "size_bytes": asset["size_bytes"],
                "transfer_method": "rsync",
                "rsync_source": f"{self.zone_id}:{path}",
                "message": "Use rsync for directory transfers",
            })

    def _handle_stats(self):
        """Get catalog statistics."""
        stats = self.catalog.get_stats()
        stats["zone_id"] = self.zone_id
        self._send_json(stats)

    def _handle_scan(self, body: Optional[Dict]):
        """Scan for new assets."""
        paths = body.get("paths") if body else None

        result = self.scanner.scan(paths)
        result["zone_id"] = self.zone_id

        self._send_json(result)

    def _handle_receive(self, body: Optional[Dict]):
        """
        Receive a pushed asset.

        Body should contain:
            - asset_id: ID of asset being sent
            - asset_type: Type of asset
            - destination: Where to store it (relative to base_dir)
            - transfer_method: "http" or "rsync"

        For rsync, returns the target path for the sender to rsync to.
        For http, expects multipart file upload (future).
        """
        if not body:
            self._send_error("Missing request body", 400)
            return

        asset_id = body.get("asset_id")
        asset_type = body.get("asset_type", "checkpoint")
        destination = body.get("destination")

        if not asset_id:
            self._send_error("Missing asset_id", 400)
            return

        # Build destination path
        if destination:
            dest_path = self.base_dir / destination
        else:
            # Default organization by type
            dest_path = self.base_dir / asset_type / asset_id

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        self._send_json({
            "zone_id": self.zone_id,
            "asset_id": asset_id,
            "destination": str(dest_path),
            "rsync_target": f"{self.zone_id}:{dest_path}",
            "ready": True,
            "message": f"Ready to receive {asset_id} at {dest_path}",
        })

    def _handle_register(self, body: Optional[Dict]):
        """Manually register an asset."""
        if not body:
            self._send_error("Missing request body", 400)
            return

        required = ["asset_id", "asset_type", "name", "path"]
        for field in required:
            if field not in body:
                self._send_error(f"Missing required field: {field}", 400)
                return

        # Verify path exists
        path = Path(body["path"])
        if not path.exists():
            self._send_error(f"Path not found: {path}", 404)
            return

        # Calculate size
        if path.is_file():
            size = path.stat().st_size
        else:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        self.catalog.register_asset(
            asset_id=body["asset_id"],
            asset_type=body["asset_type"],
            name=body["name"],
            path=str(path),
            size_bytes=size,
            checksum=body.get("checksum"),
            metadata=body.get("metadata"),
        )

        self._send_json({
            "success": True,
            "asset_id": body["asset_id"],
            "registered": True,
        })


# =============================================================================
# SERVER
# =============================================================================

def run_server(
    zone_id: str,
    port: int,
    base_dir: str,
    db_path: Optional[str] = None,
    auto_scan: bool = True,
):
    """
    Run the Branch Officer server.

    Args:
        zone_id: Unique zone identifier (e.g., "3090", "nas")
        port: Port to listen on
        base_dir: Base directory for assets
        db_path: Path to catalog database (default: base_dir/.branch_catalog.db)
        auto_scan: Whether to scan on startup
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        sys.exit(1)

    # Initialize catalog
    if db_path:
        catalog_path = Path(db_path)
    else:
        catalog_path = base_dir / ".branch_catalog.db"

    catalog = LocalCatalog(catalog_path)
    scanner = AssetScanner(catalog, base_dir)

    # Set handler class attributes
    BranchOfficerHandler.zone_id = zone_id
    BranchOfficerHandler.base_dir = base_dir
    BranchOfficerHandler.catalog = catalog
    BranchOfficerHandler.scanner = scanner

    # Auto-scan on startup
    if auto_scan:
        logger.info(f"Scanning {base_dir} for assets...")
        result = scanner.scan()
        logger.info(f"Found {result['registered']} assets")

    # Start server - use ThreadingHTTPServer to handle concurrent requests without blocking
    server = ThreadingHTTPServer(("0.0.0.0", port), BranchOfficerHandler)

    logger.info(f"Branch Officer '{zone_id}' starting on port {port}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Catalog: {catalog_path}")
    logger.info(f"Endpoints:")
    logger.info(f"  GET  /status  - Zone health and status")
    logger.info(f"  GET  /assets  - List local assets")
    logger.info(f"  GET  /assets/<id>  - Get asset details")
    logger.info(f"  GET  /fetch/<id>   - Fetch asset file")
    logger.info(f"  POST /scan    - Scan for new assets")
    logger.info(f"  POST /receive - Prepare to receive asset")
    logger.info(f"  POST /register - Register asset manually")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Branch Officer - Lightweight asset daemon for remote zones"
    )
    parser.add_argument(
        "--zone",
        type=str,
        required=True,
        help="Zone identifier (e.g., '3090', 'nas', 'backup')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8768,
        help="Port to listen on (default: 8768)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory for assets",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to catalog database (default: base_dir/.branch_catalog.db)",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Don't scan for assets on startup",
    )

    args = parser.parse_args()

    run_server(
        zone_id=args.zone,
        port=args.port,
        base_dir=args.base_dir,
        db_path=args.db_path,
        auto_scan=not args.no_scan,
    )


if __name__ == "__main__":
    main()
