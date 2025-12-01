#!/usr/bin/env python3
"""
Fleet Controller - Central management for all nodes in the fleet.

The controller runs on the control plane (4090) and:
1. Polls all node agents for health status
2. Triggers retention when thresholds are exceeded
3. Provides aggregated fleet status API
4. Stores health history in SQLite

Usage:
    # As a library (integrated into Tavern)
    from fleet.controller import FleetController
    controller = FleetController()
    status = controller.get_fleet_status()

    # Standalone daemon
    python3 -m fleet.controller --daemon

    # One-shot status check
    python3 -m fleet.controller --status

API (when integrated with Tavern):
    GET  /api/fleet/status          - Full fleet status
    GET  /api/fleet/node/{host_id}  - Specific node status
    POST /api/fleet/retention/{host_id} - Trigger retention
    POST /api/fleet/retention/all   - Trigger retention on all nodes
"""

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from fleet.types import FleetStatus, NodeHealth, NodeStatus, RetentionResult

logger = logging.getLogger("fleet.controller")

# Agent port (must match agent.py)
AGENT_PORT = 8769

# How long before a node is considered offline
OFFLINE_THRESHOLD_SECONDS = 180


class FleetController:
    """
    Central controller for the fleet of nodes.

    Polls agents, aggregates health, triggers maintenance.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the controller.

        Args:
            config_path: Path to hosts.json (auto-detect if None)
            db_path: Path to SQLite database (default: data/fleet_state.db)
        """
        self.config_path = config_path or self._find_config()
        self.db_path = db_path or self._default_db_path()

        self.hosts: Dict[str, Dict] = {}
        self._health_cache: Dict[str, NodeHealth] = {}
        self._last_poll: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._load_hosts()
        self._init_db()

    def _find_config(self) -> Path:
        """Find hosts.json config file."""
        import os
        base = os.environ.get("TRAINING_BASE_DIR")
        if base:
            return Path(base) / "config" / "hosts.json"

        candidates = [
            Path(__file__).parent.parent / "config" / "hosts.json",
            Path.home() / "Desktop" / "TRAINING" / "config" / "hosts.json",
        ]
        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError("Cannot find hosts.json")

    def _default_db_path(self) -> Path:
        """Get default database path."""
        base = Path(__file__).parent.parent / "data"
        base.mkdir(exist_ok=True)
        return base / "fleet_state.db"

    def _load_hosts(self):
        """Load hosts from config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = json.load(f)

        self.hosts = config.get("hosts", {})
        logger.info(f"Loaded {len(self.hosts)} hosts from config")

    def _init_db(self):
        """Initialize SQLite database for health history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cpu_pct REAL,
                    memory_used_pct REAL,
                    storage_used_pct REAL,
                    checkpoint_count INTEGER,
                    alerts TEXT,
                    raw_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_host_time
                ON health_snapshots(host_id, timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    dry_run INTEGER,
                    deleted_count INTEGER,
                    freed_bytes INTEGER,
                    errors TEXT
                )
            """)
            conn.commit()

    def _get_agent_url(self, host_id: str) -> Optional[str]:
        """Get agent URL for a host."""
        host = self.hosts.get(host_id)
        if not host:
            return None

        # Check if agent service is defined
        services = host.get("services", {})
        if "agent" in services:
            agent_config = services["agent"]
            port = agent_config.get("port", AGENT_PORT)
        else:
            port = AGENT_PORT

        hostname = host.get("host", "localhost")
        return f"http://{hostname}:{port}"

    def _poll_agent(self, host_id: str, timeout: int = 10) -> Optional[NodeHealth]:
        """Poll a single agent for health."""
        url = self._get_agent_url(host_id)
        if not url:
            return None

        try:
            req = Request(f"{url}/api/status", method="GET")
            with urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read())

            # Parse into NodeHealth
            health = self._parse_health(host_id, data)

            with self._lock:
                self._health_cache[host_id] = health
                self._last_poll[host_id] = time.time()

            return health

        except URLError as e:
            logger.warning(f"Failed to poll {host_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error polling {host_id}: {e}")
            return None

    def _parse_health(self, host_id: str, data: Dict) -> NodeHealth:
        """Parse health data from agent response."""
        from fleet.types import (
            GPUHealth,
            ProcessInfo,
            RetentionPolicy,
            StorageHealth,
            StorageZone,
        )

        # Parse storage
        storage = []
        for s in data.get("storage", []):
            try:
                storage.append(StorageHealth(
                    path=s["path"],
                    zone=StorageZone(s["zone"]),
                    total_bytes=int(s["total_gb"] * 1024**3),
                    used_bytes=int(s["used_gb"] * 1024**3),
                    free_bytes=int(s["free_gb"] * 1024**3),
                    checkpoint_count=s.get("checkpoint_count", 0),
                    oldest_checkpoint=s.get("oldest_checkpoint"),
                    newest_checkpoint=s.get("newest_checkpoint"),
                ))
            except Exception:
                pass

        # Parse GPUs
        gpus = []
        for g in data.get("gpus", []):
            try:
                gpus.append(GPUHealth(
                    index=g["index"],
                    name=g["name"],
                    vram_total_mb=g["vram_total_mb"],
                    vram_used_mb=g["vram_used_mb"],
                    vram_free_mb=g["vram_free_mb"],
                    utilization_pct=g["utilization_pct"],
                    temperature_c=g["temperature_c"],
                    power_draw_w=g.get("power_draw_w", 0),
                ))
            except Exception:
                pass

        # Parse processes
        processes = []
        for p in data.get("processes", []):
            try:
                processes.append(ProcessInfo(
                    pid=p["pid"],
                    name=p["name"],
                    cpu_pct=p["cpu_pct"],
                    memory_mb=p["memory_mb"],
                ))
            except Exception:
                pass

        # Parse retention policy
        retention = None
        if data.get("retention_policy"):
            rp = data["retention_policy"]
            retention = RetentionPolicy(
                max_checkpoints=rp.get("max_checkpoints"),
                max_gb=rp.get("max_gb"),
                keep_strategy=rp.get("keep_strategy", "recently_used"),
                is_vault=rp.get("is_vault", False),
                cleanup_threshold_pct=rp.get("cleanup_threshold_pct", 90.0),
            )

        # Parse memory
        mem = data.get("memory", {})

        return NodeHealth(
            host_id=host_id,
            device_id=data.get("device_id", host_id),
            hostname=data.get("hostname", ""),
            status=NodeStatus(data.get("status", "unknown")),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            uptime_seconds=data.get("uptime_seconds", 0),
            cpu_pct=data.get("cpu_pct", 0),
            memory_total_mb=mem.get("total_mb", 0),
            memory_used_mb=mem.get("used_mb", 0),
            memory_free_mb=mem.get("free_mb", 0),
            storage=storage,
            gpus=gpus,
            processes=processes,
            retention_policy=retention,
            alerts=data.get("alerts", []),
            last_retention_run=data.get("last_retention_run"),
            checkpoints_deleted_today=data.get("checkpoints_deleted_today", 0),
            bytes_freed_today=data.get("bytes_freed_today", 0),
        )

    def poll_all(self, timeout: int = 10) -> Dict[str, Optional[NodeHealth]]:
        """Poll all nodes in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._poll_agent, host_id, timeout): host_id
                for host_id in self.hosts
            }
            for future in as_completed(futures, timeout=timeout + 5):
                host_id = futures[future]
                try:
                    results[host_id] = future.result()
                except Exception as e:
                    logger.error(f"Error polling {host_id}: {e}")
                    results[host_id] = None

        return results

    def get_node_health(
        self,
        host_id: str,
        max_age_seconds: int = 60,
    ) -> Optional[NodeHealth]:
        """
        Get health for a specific node.

        Uses cache if fresh enough, otherwise polls.
        """
        with self._lock:
            cached = self._health_cache.get(host_id)
            last_poll = self._last_poll.get(host_id, 0)

        if cached and (time.time() - last_poll) < max_age_seconds:
            return cached

        # Poll fresh
        return self._poll_agent(host_id)

    def get_fleet_status(self, max_age_seconds: int = 60) -> FleetStatus:
        """
        Get aggregated fleet status.

        Args:
            max_age_seconds: Max age for cached data before re-polling
        """
        # Check if we need to poll
        now = time.time()
        needs_poll = []
        for host_id in self.hosts:
            last = self._last_poll.get(host_id, 0)
            if (now - last) >= max_age_seconds:
                needs_poll.append(host_id)

        # Poll stale nodes
        if needs_poll:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self._poll_agent, h): h
                    for h in needs_poll
                }
                for future in as_completed(futures, timeout=15):
                    try:
                        future.result()
                    except Exception:
                        pass

        # Build status from cache
        nodes = {}
        for host_id, host_config in self.hosts.items():
            with self._lock:
                cached = self._health_cache.get(host_id)
                last_poll = self._last_poll.get(host_id, 0)

            if cached and (now - last_poll) < OFFLINE_THRESHOLD_SECONDS:
                nodes[host_id] = cached
            else:
                # Node is offline
                nodes[host_id] = NodeHealth.offline(
                    host_id=host_id,
                    device_id=host_config.get("device_id", host_id),
                    hostname=host_config.get("host", "unknown"),
                )

        # Aggregate stats
        status_counts = {
            NodeStatus.HEALTHY: 0,
            NodeStatus.WARNING: 0,
            NodeStatus.CRITICAL: 0,
            NodeStatus.OFFLINE: 0,
        }
        total_checkpoints = 0
        total_storage = 0
        total_storage_used = 0
        total_vram = 0
        total_vram_used = 0

        for health in nodes.values():
            status_counts[health.status] = status_counts.get(health.status, 0) + 1
            total_checkpoints += health.checkpoint_count

            for s in health.storage:
                total_storage += s.total_gb
                total_storage_used += s.used_gb

            for g in health.gpus:
                total_vram += g.vram_total_mb / 1024
                total_vram_used += g.vram_used_mb / 1024

        return FleetStatus(
            timestamp=datetime.now().isoformat(),
            total_nodes=len(nodes),
            healthy_nodes=status_counts.get(NodeStatus.HEALTHY, 0),
            warning_nodes=status_counts.get(NodeStatus.WARNING, 0),
            critical_nodes=status_counts.get(NodeStatus.CRITICAL, 0),
            offline_nodes=status_counts.get(NodeStatus.OFFLINE, 0),
            nodes=nodes,
            total_checkpoints=total_checkpoints,
            total_storage_gb=total_storage,
            total_storage_used_gb=total_storage_used,
            total_vram_gb=total_vram,
            total_vram_used_gb=total_vram_used,
        )

    def trigger_retention(
        self,
        host_id: str,
        dry_run: bool = False,
        timeout: int = 60,
    ) -> Optional[RetentionResult]:
        """
        Trigger retention on a specific node.

        Args:
            host_id: Host to run retention on
            dry_run: If True, only report what would be deleted
            timeout: Request timeout in seconds

        Returns:
            RetentionResult or None if failed
        """
        url = self._get_agent_url(host_id)
        if not url:
            logger.error(f"No agent URL for {host_id}")
            return None

        try:
            body = json.dumps({"dry_run": dry_run}).encode()
            req = Request(
                f"{url}/api/retention",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read())

            result = RetentionResult(
                host_id=data.get("host_id", host_id),
                device_id=data.get("device_id", host_id),
                timestamp=data.get("timestamp", datetime.now().isoformat()),
                dry_run=data.get("dry_run", dry_run),
                checkpoints_before=data.get("checkpoints_before", 0),
                checkpoints_after=data.get("checkpoints_after", 0),
                deleted_count=data.get("deleted_count", 0),
                deleted_steps=data.get("deleted_steps", []),
                freed_bytes=int(data.get("freed_gb", 0) * 1024**3),
                errors=data.get("errors", []),
            )

            # Store in database
            self._store_retention_run(result)

            return result

        except Exception as e:
            logger.error(f"Failed to trigger retention on {host_id}: {e}")
            return None

    def trigger_retention_all(
        self,
        dry_run: bool = False,
        only_critical: bool = True,
    ) -> Dict[str, Optional[RetentionResult]]:
        """
        Trigger retention on all (or critical) nodes.

        Args:
            dry_run: If True, only report what would be deleted
            only_critical: If True, only run on nodes that need it

        Returns:
            Dict mapping host_id to RetentionResult
        """
        results = {}

        # Get current status
        status = self.get_fleet_status()

        for host_id, health in status.nodes.items():
            # Skip offline nodes
            if health.status == NodeStatus.OFFLINE:
                continue

            # Skip if only_critical and node doesn't need retention
            if only_critical and not health.needs_retention:
                continue

            # Skip vault nodes
            if health.retention_policy and health.retention_policy.is_vault:
                continue

            logger.info(f"Triggering retention on {host_id}")
            results[host_id] = self.trigger_retention(host_id, dry_run=dry_run)

        return results

    def _store_retention_run(self, result: RetentionResult):
        """Store retention run in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO retention_runs
                    (host_id, timestamp, dry_run, deleted_count, freed_bytes, errors)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.host_id,
                        result.timestamp,
                        1 if result.dry_run else 0,
                        result.deleted_count,
                        result.freed_bytes,
                        json.dumps(result.errors),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store retention run: {e}")

    def store_health_snapshot(self, health: NodeHealth):
        """Store health snapshot in database."""
        try:
            # Calculate storage used percentage
            storage_used_pct = 0
            if health.storage:
                total = sum(s.total_bytes for s in health.storage)
                used = sum(s.used_bytes for s in health.storage)
                if total > 0:
                    storage_used_pct = (used / total) * 100

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO health_snapshots
                    (host_id, timestamp, status, cpu_pct, memory_used_pct,
                     storage_used_pct, checkpoint_count, alerts, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        health.host_id,
                        health.timestamp,
                        health.status.value,
                        health.cpu_pct,
                        health.memory_used_pct,
                        storage_used_pct,
                        health.checkpoint_count,
                        json.dumps(health.alerts),
                        json.dumps(health.to_dict()),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store health snapshot: {e}")

    def get_health_history(
        self,
        host_id: str,
        hours: int = 24,
    ) -> List[Dict]:
        """Get health history for a node."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM health_snapshots
                    WHERE host_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                    """,
                    (host_id, cutoff),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return []

    def check_and_maintain(self):
        """
        Check fleet status and trigger maintenance if needed.

        This is the main loop for daemon mode.
        """
        logger.info("Checking fleet status...")
        status = self.get_fleet_status(max_age_seconds=30)

        # Store snapshots
        for health in status.nodes.values():
            if health.status != NodeStatus.OFFLINE:
                self.store_health_snapshot(health)

        # Log summary
        logger.info(
            f"Fleet: {status.healthy_nodes} healthy, "
            f"{status.warning_nodes} warning, "
            f"{status.critical_nodes} critical, "
            f"{status.offline_nodes} offline"
        )

        # Trigger retention on nodes that need it
        for host_id, health in status.nodes.items():
            if health.needs_retention:
                logger.warning(f"{host_id} needs retention - triggering")
                self.trigger_retention(host_id, dry_run=False)

    def run_daemon(self, interval_seconds: int = 60):
        """Run the controller as a daemon."""
        logger.info(f"Starting fleet controller daemon (interval={interval_seconds}s)")

        while True:
            try:
                self.check_and_maintain()
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")

            time.sleep(interval_seconds)


# Singleton instance
_controller: Optional[FleetController] = None
_controller_lock = threading.Lock()


def get_controller() -> FleetController:
    """Get or create the fleet controller singleton."""
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = FleetController()
    return _controller


def get_fleet_status(max_age_seconds: int = 60) -> FleetStatus:
    """Get fleet status (convenience function)."""
    return get_controller().get_fleet_status(max_age_seconds)


def get_node_health(host_id: str) -> Optional[NodeHealth]:
    """Get node health (convenience function)."""
    return get_controller().get_node_health(host_id)


def trigger_retention(host_id: str, dry_run: bool = False) -> Optional[RetentionResult]:
    """Trigger retention on a node (convenience function)."""
    return get_controller().trigger_retention(host_id, dry_run)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fleet Controller")
    parser.add_argument("--status", action="store_true", help="Show fleet status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=60, help="Daemon poll interval")
    parser.add_argument("--retention", type=str, help="Trigger retention on host")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for retention")

    args = parser.parse_args()

    controller = FleetController()

    if args.status:
        status = controller.get_fleet_status()
        print(json.dumps(status.to_dict(), indent=2))

    elif args.retention:
        result = controller.trigger_retention(args.retention, dry_run=args.dry_run)
        if result:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Failed to trigger retention on {args.retention}")

    elif args.daemon:
        controller.run_daemon(interval_seconds=args.interval)

    else:
        # Default: show status
        status = controller.get_fleet_status()
        print(f"\nFleet Status ({status.timestamp})")
        print("=" * 50)
        print(f"Nodes: {status.total_nodes} total")
        print(f"  Healthy:  {status.healthy_nodes}")
        print(f"  Warning:  {status.warning_nodes}")
        print(f"  Critical: {status.critical_nodes}")
        print(f"  Offline:  {status.offline_nodes}")
        print(f"\nCheckpoints: {status.total_checkpoints}")
        print(f"Storage: {status.total_storage_used_gb:.1f} / {status.total_storage_gb:.1f} GB")

        print("\nNodes:")
        for host_id, health in status.nodes.items():
            icon = {
                NodeStatus.HEALTHY: "✓",
                NodeStatus.WARNING: "!",
                NodeStatus.CRITICAL: "✗",
                NodeStatus.OFFLINE: "○",
            }.get(health.status, "?")
            print(f"  [{icon}] {host_id}: {health.status.value}")
            if health.alerts:
                for alert in health.alerts[:2]:
                    print(f"      {alert}")
