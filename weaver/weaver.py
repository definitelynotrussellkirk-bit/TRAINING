#!/usr/bin/env python3
"""
The Weaver - Daemon orchestrator that keeps all threads alive

Now uses Service Registry as the single source of truth for service definitions.
Weaver adds continuous monitoring, restart throttling, and multi-level health checks.

Watches over (from configs/services.json):
- vault        - VaultKeeper (the memory)
- tavern       - Game UI (the face)
- training     - Training Daemon (the heart)
- realm_state  - RealmState service (the truth)
- data_flow    - Queue feeder (the fuel)
- eval_runner  - Eval processor (the judge) [optional]

Usage:
    python3 weaver/weaver.py              # Run once (check & fix)
    python3 weaver/weaver.py --daemon     # Run continuously
    python3 weaver/weaver.py --status     # Show tapestry status
    python3 weaver/weaver.py --restart SERVICE  # Restart a service
    python3 weaver/weaver.py --shutdown   # Force shutdown all
    python3 weaver/weaver.py --start      # Start all fresh
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir
from core.service_registry import (
    ServiceConfig,
    HealthCheckKind,
    get_service,
    get_all_services,
    get_service_status,
    get_dependency_order,
    is_service_running,
    start_service as registry_start_service,
    stop_service as registry_stop_service,
    start_realm,
    stop_realm,
    reload_services,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ThreadState:
    """Runtime state for a service being monitored."""
    service_id: str
    restart_count: int = 0
    last_restart: float = 0.0


class Weaver:
    """
    The Weaver - Orchestrates all daemon threads

    Uses Service Registry for service definitions, adds:
    1. Continuous monitoring loop
    2. Restart throttling (max_restarts per hour)
    3. Multi-level health checks (Level 2: data flow, Level 3: performance)
    4. Groundskeeper integration (hourly cleanup)
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.config = self._load_config()
        self.running = False
        self.check_interval = 30  # seconds
        self.pid_file = self.base_dir / ".pids" / "weaver.pid"

        # Runtime state per service (restart counts, etc.)
        self.thread_states: Dict[str, ThreadState] = {}

        # Services to watch (from service registry)
        self._watched_services = [
            "vault", "tavern", "training", "realm_state", "data_flow", "eval_runner"
        ]

    def _load_config(self) -> dict:
        """Load main config"""
        config_path = self.base_dir / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def _get_thread_state(self, service_id: str) -> ThreadState:
        """Get or create thread state for a service."""
        if service_id not in self.thread_states:
            self.thread_states[service_id] = ThreadState(service_id=service_id)
        return self.thread_states[service_id]

    # ========== Health Checks ==========

    def _check_service_health(self, service_id: str) -> bool:
        """Check if a service is healthy using Service Registry."""
        return is_service_running(service_id)

    def _check_data_flow_health(self) -> bool:
        """
        Level 2: Check if RealmStore is being updated.

        Not just "is daemon alive?" but "is data flowing?"
        """
        try:
            from core.realm_store import get_store
            from datetime import datetime, timezone

            store = get_store()
            training = store.get_training()

            if not training:
                logger.debug("No training data in RealmStore")
                return False

            updated_at = training.get("updated_at")
            if not updated_at:
                logger.debug("No updated_at timestamp in training data")
                return False

            try:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=timezone.utc)

                age_seconds = (now - last_update).total_seconds()
                is_fresh = age_seconds < 120  # 2 minutes

                if not is_fresh:
                    logger.warning(f"RealmStore data is stale ({age_seconds:.0f}s old)")

                return is_fresh

            except Exception as e:
                logger.debug(f"Could not parse updated_at timestamp: {e}")
                return False

        except Exception as e:
            logger.debug(f"Data flow check failed: {e}")
            return False

    def _check_performance_health(self) -> bool:
        """
        Level 3: Check if training performance is acceptable.

        Not just "is data flowing?" but "is it flowing at expected speed?"
        """
        try:
            from core.status_monitor import check_performance_health

            result = check_performance_health()

            if result.is_degraded:
                logger.warning(f"Performance degraded: {result.message}")

            return not result.is_degraded

        except Exception as e:
            logger.debug(f"Performance check failed: {e}")
            return True  # Don't block on check failures

    def _pid_alive(self, pid: int) -> bool:
        """Check if a PID is running"""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    # ========== Thread Management ==========

    def _start_thread(self, service_id: str) -> bool:
        """Start a service with restart throttling."""
        service = get_service(service_id)
        if not service:
            logger.error(f"Unknown service: {service_id}")
            return False

        state = self._get_thread_state(service_id)
        now = time.time()

        # Reset restart count after 1 hour
        if now - state.last_restart > 3600:
            state.restart_count = 0

        # Check restart limit
        max_restarts = service.startup.max_restarts
        if state.restart_count >= max_restarts:
            logger.error(f"Service {service.name} exceeded max restarts ({max_restarts}/hr)")
            return False

        logger.info(f"Starting service: {service.name}")
        logger.info(f"  Command: {' '.join(service.command)}")

        try:
            # Use service registry to start
            success = registry_start_service(service_id, ensure_deps=True)

            if success:
                state.restart_count += 1
                state.last_restart = now
                logger.info(f"  Service {service.name} started successfully")
            else:
                logger.warning(f"  Service {service.name} failed to start")

            return success

        except Exception as e:
            logger.error(f"  Failed to start {service.name}: {e}")
            return False

    def check_tapestry(self) -> Dict[str, dict]:
        """
        Check status of all watched services.

        Multi-level monitoring:
        - Level 1: Process alive (all services)
        - Level 2: Data flowing (training daemon)
        - Level 3: Performance OK (training daemon)
        """
        status = {}

        for service_id in self._watched_services:
            service = get_service(service_id)
            if not service:
                continue

            alive = self._check_service_health(service_id)
            state = self._get_thread_state(service_id)

            thread_status = {
                "name": service.name,
                "description": service.description,
                "alive": alive,
                "required": service.required,
                "restarts": state.restart_count,
            }

            # Level 2 & 3: Additional checks for training daemon
            if service_id == "training" and alive:
                # Check if monitoring is configured
                if service.monitoring.level2_check:
                    data_flow_ok = self._check_data_flow_health()
                    thread_status["data_flow_ok"] = data_flow_ok
                else:
                    data_flow_ok = True

                if service.monitoring.level3_check:
                    performance_ok = self._check_performance_health()
                    thread_status["performance_ok"] = performance_ok
                else:
                    performance_ok = True

                thread_status["fully_healthy"] = alive and data_flow_ok and performance_ok

            status[service_id] = thread_status

        return status

    def restart_thread(self, service_id: str) -> bool:
        """
        Force restart a specific service.

        Kills (if running) and restarts using service registry.
        """
        service = get_service(service_id)
        if not service:
            logger.error(f"Unknown service: {service_id}")
            available = [s for s in self._watched_services if get_service(s)]
            logger.info(f"Available services: {available}")
            return False

        logger.info(f"Restarting service: {service.name}")

        # Stop if running
        if is_service_running(service_id):
            logger.info(f"  Stopping {service.name}...")
            registry_stop_service(service_id)
            time.sleep(1)

        # Start the service
        logger.info(f"  Starting {service.name}...")
        return self._start_thread(service_id)

    def mend(self, dry_run: bool = False) -> Dict[str, str]:
        """Mend broken threads (restart dead daemons)"""
        results = {}

        for service_id in self._watched_services:
            service = get_service(service_id)
            if not service:
                continue

            if not service.required:
                results[service_id] = "skipped (not required)"
                continue

            alive = self._check_service_health(service_id)

            if alive:
                results[service_id] = "healthy"
            else:
                if dry_run:
                    results[service_id] = "would restart"
                else:
                    logger.warning(f"Service {service.name} is dead, restarting...")
                    success = self._start_thread(service_id)
                    results[service_id] = "restarted" if success else "restart failed"

        return results

    def weave(self):
        """Main daemon loop - continuously watch and mend"""
        logger.info("The Weaver awakens...")
        logger.info(f"Watching {len(self._watched_services)} services")
        logger.info(f"Check interval: {self.check_interval}s")

        # Write PID
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Groundskeeper integration
        last_groundskeeper_run = 0
        groundskeeper_interval = 3600  # Run cleanup every hour

        try:
            while self.running:
                # Check and mend services
                results = self.mend()

                # Log summary
                healthy = sum(1 for r in results.values() if r == "healthy")
                total = len(results)
                logger.info(f"Tapestry check: {healthy}/{total} services healthy")

                # Run Groundskeeper cleanup (hourly)
                now = time.time()
                if now - last_groundskeeper_run > groundskeeper_interval:
                    self._run_groundskeeper()
                    last_groundskeeper_run = now

                # Sleep
                time.sleep(self.check_interval)

        finally:
            logger.info("The Weaver sleeps...")
            if self.pid_file.exists():
                self.pid_file.unlink()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _run_groundskeeper(self):
        """Run Groundskeeper cleanup sweep."""
        try:
            from core.groundskeeper import Groundskeeper
            gk = Groundskeeper(base_dir=self.base_dir)
            results = gk.sweep(dry_run=False)

            total_items = sum(r.items_cleaned for r in results.values())
            total_mb = sum(r.bytes_freed for r in results.values()) / (1024 * 1024)

            if total_items > 0:
                logger.info(f"Groundskeeper: cleaned {total_items} items, freed {total_mb:.2f}MB")
            else:
                logger.debug("Groundskeeper: nothing to clean")

        except Exception as e:
            logger.warning(f"Groundskeeper sweep failed: {e}")

    def status(self) -> str:
        """Get formatted status string"""
        tapestry = self.check_tapestry()

        lines = [
            "",
            "=" * 60,
            "THE TAPESTRY - Thread Status",
            "=" * 60,
            ""
        ]

        for service_id, info in tapestry.items():
            icon = "✅" if info["alive"] else "❌"
            req = "(required)" if info["required"] else "(optional)"
            restarts = f"[restarts: {info['restarts']}]" if info["restarts"] > 0 else ""
            lines.append(f"{icon} {info['name']}: {info['description']} {req} {restarts}")

        lines.append("")

        healthy = sum(1 for i in tapestry.values() if i["alive"])
        total = len(tapestry)

        if healthy == total:
            lines.append("Tapestry Status: ALL THREADS INTACT")
        else:
            lines.append(f"Tapestry Status: {total - healthy} THREADS BROKEN")

        lines.append("=" * 60)

        return "\n".join(lines)


def is_weaver_running(base_dir: Path) -> tuple[bool, Optional[int]]:
    """Check if another Weaver is already running"""
    pid_file = base_dir / ".pids" / "weaver.pid"
    if not pid_file.exists():
        return False, None

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (ValueError, ProcessLookupError, OSError):
        pid_file.unlink(missing_ok=True)
        return False, None


def force_shutdown(base_dir: Path) -> bool:
    """
    Force shutdown all services cleanly using Service Registry.
    """
    print("\n" + "=" * 60)
    print("FORCE SHUTDOWN - Stopping all services")
    print("=" * 60 + "\n")

    # Use service registry to stop in correct order
    success = stop_realm()

    # Clean up shutdown marker
    control_dir = base_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    shutdown_marker = control_dir / ".clean_shutdown"
    shutdown_marker.write_text(datetime.now().isoformat())

    print("\n" + "=" * 60)
    if success:
        print("✅ SHUTDOWN COMPLETE")
    else:
        print("⚠  SHUTDOWN INCOMPLETE (some services may still be running)")
    print("   Restart with: python3 weaver/weaver.py --start")
    print("=" * 60 + "\n")

    return success


def start_fresh(base_dir: Path) -> bool:
    """
    Start all services from a clean state using Service Registry.
    """
    print("\n" + "=" * 60)
    print("STARTING FRESH - Bringing up all services")
    print("=" * 60 + "\n")

    control_dir = base_dir / "control"
    shutdown_marker = control_dir / ".clean_shutdown"

    # Check for clean shutdown
    if shutdown_marker.exists():
        shutdown_time = shutdown_marker.read_text().strip()
        print(f"✓ Clean shutdown detected at {shutdown_time}")
        shutdown_marker.unlink()
    else:
        print("⚠ No clean shutdown marker - previous session may have crashed")

    # Use service registry to start in correct order
    success = start_realm(include_optional=False)

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL SERVICES RUNNING")
        print("   Tavern:      http://localhost:8888")
        print("   VaultKeeper: http://localhost:8767")
        print("")
        print("   To run Weaver daemon: python3 weaver/weaver.py --daemon")
    else:
        print("❌ STARTUP FAILED - check logs for details")
    print("=" * 60 + "\n")

    return success


def main():
    import argparse

    parser = argparse.ArgumentParser(description="The Weaver - Daemon orchestrator")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (continuous)")
    parser.add_argument("--status", action="store_true", help="Show tapestry status")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't restart")
    parser.add_argument("--restart", metavar="SERVICE", help="Restart a specific service")
    parser.add_argument("--shutdown", action="store_true", help="Force shutdown all services cleanly")
    parser.add_argument("--start", action="store_true", help="Start all services fresh")
    parser.add_argument("--base-dir", default=None, help="Base directory")

    args = parser.parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else get_base_dir()

    # For daemon mode, check for existing Weaver
    if args.daemon:
        running, existing_pid = is_weaver_running(base_dir)
        if running:
            print(f"ERROR: Another Weaver is already running (PID {existing_pid})")
            print("Kill the existing Weaver first, or use --status to check status")
            sys.exit(1)

    weaver = Weaver(str(base_dir) if args.base_dir else None)

    if args.shutdown:
        force_shutdown(base_dir)
        sys.exit(0)
    elif args.start:
        success = start_fresh(base_dir)
        sys.exit(0 if success else 1)
    elif args.status:
        print(weaver.status())
    elif args.restart:
        service = args.restart.lower()
        print(f"Restarting {service}...")
        success = weaver.restart_thread(service)
        if success:
            print(f"  {service} restarted successfully")
            sys.exit(0)
        else:
            print(f"  Failed to restart {service}")
            sys.exit(1)
    elif args.daemon:
        weaver.weave()
    else:
        # Single check and mend
        print(weaver.status())
        print("\nMending broken threads...")
        results = weaver.mend(dry_run=args.dry_run)
        for name, result in results.items():
            print(f"  {name}: {result}")


if __name__ == "__main__":
    main()
