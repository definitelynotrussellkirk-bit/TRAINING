#!/usr/bin/env python3
"""
The Weaver - Daemon orchestrator that keeps all threads alive

Watches over:
- Training daemon (the heart)
- Tavern server (the face)
- VaultKeeper (the memory)
- Data generation (the fuel)

Usage:
    python3 weaver/weaver.py              # Run once (check & fix)
    python3 weaver/weaver.py --daemon     # Run continuously
    python3 weaver/weaver.py --status     # Show tapestry status
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
from typing import Dict, List, Optional, Callable
import requests

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Thread:
    """A thread in the tapestry (a daemon to watch)"""
    name: str
    description: str
    health_check: Callable[[], bool]
    start_cmd: List[str]
    pid_file: Optional[str] = None
    required: bool = True  # If False, won't auto-start
    restart_delay: int = 5  # Seconds to wait before restart
    max_restarts: int = 3   # Max restarts per hour
    restart_count: int = field(default=0, repr=False)
    last_restart: float = field(default=0.0, repr=False)


class Weaver:
    """
    The Weaver - Orchestrates all daemon threads

    Responsibilities:
    1. Monitor thread health
    2. Restart dead threads
    3. Maintain data flow (queue depth)
    4. Report tapestry status
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.config = self._load_config()
        self.threads: Dict[str, Thread] = {}
        self.running = False
        self.check_interval = 30  # seconds
        self.pid_file = self.base_dir / ".pids" / "weaver.pid"

        # Register all threads
        self._register_threads()

    def _load_config(self) -> dict:
        """Load main config"""
        config_path = self.base_dir / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def _register_threads(self):
        """Register all daemon threads to watch"""

        # Training Daemon - The Heart
        self.threads["training"] = Thread(
            name="Training Daemon",
            description="The heart - processes quests and trains DIO",
            health_check=self._check_training_daemon,
            start_cmd=[
                "python3", str(self.base_dir / "core" / "training_daemon.py"),
                "--base-dir", str(self.base_dir)
            ],
            pid_file=str(self.base_dir / ".daemon.pid"),
            required=True
        )

        # Tavern Server - The Face
        self.threads["tavern"] = Thread(
            name="Tavern Server",
            description="The face - game UI at port 8888",
            health_check=self._check_tavern,
            start_cmd=[
                "python3", str(self.base_dir / "tavern" / "server.py"),
                "--port", "8888"
            ],
            pid_file=str(self.base_dir / ".pids" / "tavern.pid"),
            required=True
        )

        # VaultKeeper - The Memory
        self.threads["vault"] = Thread(
            name="VaultKeeper",
            description="The memory - asset registry at port 8767",
            health_check=self._check_vaultkeeper,
            start_cmd=[
                "python3", str(self.base_dir / "vault" / "server.py"),
                "--port", "8767"
            ],
            pid_file=str(self.base_dir / ".pids" / "vault.pid"),
            required=True
        )

        # Data Flow - The Fuel (special: not a daemon, but a task)
        self.threads["data_flow"] = Thread(
            name="Data Flow",
            description="The fuel - keeps quest queue fed",
            health_check=self._check_data_flow,
            start_cmd=[
                "python3", str(self.base_dir / "data_manager" / "manager.py"),
                "generate", "--force"
            ],
            required=True,
            restart_delay=10,
            max_restarts=100  # Can generate many times
        )

    # ========== Health Checks ==========

    def _check_training_daemon(self) -> bool:
        """Check if training daemon is alive"""
        try:
            resp = requests.get("http://localhost:8888/api/daemon/status", timeout=5)
            data = resp.json()
            return data.get("daemon_running", False)
        except:
            # Check PID file as fallback
            pid_file = self.base_dir / ".daemon.pid"
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                return self._pid_alive(pid)
            return False

    def _check_tavern(self) -> bool:
        """Check if Tavern is responding"""
        try:
            resp = requests.get("http://localhost:8888/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def _check_vaultkeeper(self) -> bool:
        """Check if VaultKeeper is responding"""
        try:
            resp = requests.get("http://localhost:8767/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def _check_data_flow(self) -> bool:
        """Check if queue has enough data (queued + processing)"""
        try:
            # Check queue directly (more reliable than API)
            from core.training_queue import TrainingQueue
            queue = TrainingQueue(self.base_dir)
            status = queue.get_queue_status()
            available = status["total_queued"] + status.get("processing", 0)
            min_depth = self.config.get("auto_generate", {}).get("min_queue_depth", 2)
            return available >= min_depth
        except Exception as e:
            logger.debug(f"Could not check data flow: {e}")
            return True  # Assume OK if can't check

    def _pid_alive(self, pid: int) -> bool:
        """Check if a PID is running"""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    # ========== Thread Management ==========

    def _start_thread(self, thread: Thread) -> bool:
        """Start a thread (daemon)"""
        now = time.time()

        # Check restart limits (reset counter after 1 hour)
        if now - thread.last_restart > 3600:
            thread.restart_count = 0

        if thread.restart_count >= thread.max_restarts:
            logger.error(f"Thread {thread.name} exceeded max restarts ({thread.max_restarts}/hr)")
            return False

        logger.info(f"Starting thread: {thread.name}")
        logger.info(f"  Command: {' '.join(thread.start_cmd)}")

        try:
            # Start as background process
            log_file = self.base_dir / "logs" / f"{thread.name.lower().replace(' ', '_')}.log"
            with open(log_file, 'a') as log:
                process = subprocess.Popen(
                    thread.start_cmd,
                    stdout=log,
                    stderr=log,
                    cwd=str(self.base_dir),
                    start_new_session=True
                )

            thread.restart_count += 1
            thread.last_restart = now

            # Wait a moment and verify
            time.sleep(thread.restart_delay)

            if thread.health_check():
                logger.info(f"  Thread {thread.name} started successfully")
                return True
            else:
                logger.warning(f"  Thread {thread.name} started but health check failed")
                return False

        except Exception as e:
            logger.error(f"  Failed to start {thread.name}: {e}")
            return False

    def check_tapestry(self) -> Dict[str, dict]:
        """Check status of all threads"""
        status = {}

        for name, thread in self.threads.items():
            alive = thread.health_check()
            status[name] = {
                "name": thread.name,
                "description": thread.description,
                "alive": alive,
                "required": thread.required,
                "restarts": thread.restart_count
            }

        return status

    def restart_thread(self, thread_name: str) -> bool:
        """
        Force restart a specific thread.

        This is the canonical way to restart a service after code changes.
        The thread will be killed (if running) and then restarted.

        Args:
            thread_name: Name of thread to restart (training, tavern, vault, data_flow)

        Returns:
            True if restart succeeded, False otherwise
        """
        if thread_name not in self.threads:
            logger.error(f"Unknown thread: {thread_name}")
            logger.info(f"Available threads: {list(self.threads.keys())}")
            return False

        thread = self.threads[thread_name]
        logger.info(f"Restarting thread: {thread.name}")

        # Kill if running
        if thread.pid_file:
            pid_path = Path(thread.pid_file)
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                    if self._pid_alive(pid):
                        logger.info(f"  Killing PID {pid}...")
                        os.kill(pid, signal.SIGTERM)
                        # Wait for graceful shutdown
                        for _ in range(10):
                            time.sleep(0.5)
                            if not self._pid_alive(pid):
                                break
                        else:
                            # Force kill if still running
                            logger.warning(f"  Force killing PID {pid}...")
                            os.kill(pid, signal.SIGKILL)
                            time.sleep(0.5)
                except (ValueError, ProcessLookupError):
                    pass
                pid_path.unlink(missing_ok=True)

        # Wait a moment for cleanup
        time.sleep(1)

        # Start the thread
        logger.info(f"  Starting {thread.name}...")
        return self._start_thread(thread)

    def mend(self, dry_run: bool = False) -> Dict[str, str]:
        """Mend broken threads (restart dead daemons)"""
        results = {}

        for name, thread in self.threads.items():
            if not thread.required:
                results[name] = "skipped (not required)"
                continue

            alive = thread.health_check()

            if alive:
                results[name] = "healthy"
            else:
                if dry_run:
                    results[name] = "would restart"
                else:
                    logger.warning(f"Thread {thread.name} is dead, restarting...")
                    success = self._start_thread(thread)
                    results[name] = "restarted" if success else "restart failed"

        return results

    def weave(self):
        """Main daemon loop - continuously watch and mend"""
        logger.info("The Weaver awakens...")
        logger.info(f"Watching {len(self.threads)} threads")
        logger.info(f"Check interval: {self.check_interval}s")

        # Write PID
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            while self.running:
                # Check and mend
                results = self.mend()

                # Log summary
                healthy = sum(1 for r in results.values() if r == "healthy")
                total = len(results)
                logger.info(f"Tapestry check: {healthy}/{total} threads healthy")

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

        for name, info in tapestry.items():
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
        # Check if process is running
        os.kill(pid, 0)
        return True, pid
    except (ValueError, ProcessLookupError, OSError):
        # PID file exists but process is dead - clean it up
        pid_file.unlink(missing_ok=True)
        return False, None


def force_shutdown(base_dir: Path) -> bool:
    """
    Force shutdown all services cleanly.

    Order matters:
    1. Signal training to finish current batch (graceful)
    2. Stop weaver daemon (so it doesn't restart things)
    3. Stop tavern (UI)
    4. Stop vault (API)
    5. Stop training daemon
    6. Write shutdown marker

    Returns:
        True if all services stopped cleanly
    """
    print("\n" + "=" * 60)
    print("FORCE SHUTDOWN - Stopping all services")
    print("=" * 60 + "\n")

    pids_dir = base_dir / ".pids"

    # 1. Stop weaver daemon first (so it doesn't restart things)
    weaver_pid = pids_dir / "weaver.pid"
    if weaver_pid.exists():
        try:
            pid = int(weaver_pid.read_text().strip())
            print(f"[1/5] Stopping Weaver daemon (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except (ValueError, ProcessLookupError):
            pass
        weaver_pid.unlink(missing_ok=True)
    else:
        print("[1/5] Weaver daemon not running")

    # 2. Signal training to pause (graceful)
    control_dir = base_dir / "control"
    pause_file = control_dir / ".pause"
    print("[2/5] Signaling training to pause...")
    pause_file.touch()
    time.sleep(2)  # Give it time to finish current batch

    # 3. Stop tavern
    tavern_pid = pids_dir / "tavern.pid"
    if tavern_pid.exists():
        try:
            pid = int(tavern_pid.read_text().strip())
            print(f"[3/5] Stopping Tavern (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except (ValueError, ProcessLookupError):
            pass
        tavern_pid.unlink(missing_ok=True)
    else:
        print("[3/5] Tavern not running (checking port)...")
        subprocess.run(["fuser", "-k", "8888/tcp"], capture_output=True)

    # 4. Stop vault
    vault_pid = pids_dir / "vault.pid"
    if vault_pid.exists():
        try:
            pid = int(vault_pid.read_text().strip())
            print(f"[4/5] Stopping VaultKeeper (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except (ValueError, ProcessLookupError):
            pass
        vault_pid.unlink(missing_ok=True)
    else:
        print("[4/5] VaultKeeper not running (checking port)...")
        subprocess.run(["fuser", "-k", "8767/tcp"], capture_output=True)

    # 5. Stop training daemon
    daemon_pid = base_dir / ".daemon.pid"
    if daemon_pid.exists():
        try:
            pid = int(daemon_pid.read_text().strip())
            print(f"[5/5] Stopping Training Daemon (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful shutdown
            for i in range(10):
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                print("      Force killing training daemon...")
                os.kill(pid, signal.SIGKILL)
        except (ValueError, ProcessLookupError):
            pass
        daemon_pid.unlink(missing_ok=True)
    else:
        print("[5/5] Training daemon not running")

    # Remove pause file
    pause_file.unlink(missing_ok=True)

    # Write shutdown marker
    shutdown_marker = control_dir / ".clean_shutdown"
    shutdown_marker.write_text(datetime.now().isoformat())

    print("\n" + "=" * 60)
    print("✅ SHUTDOWN COMPLETE")
    print("   All services stopped. Safe to make code changes.")
    print("   Restart with: python3 weaver/weaver.py --start")
    print("=" * 60 + "\n")

    return True


def start_fresh(base_dir: Path) -> bool:
    """
    Start all services from a clean state.

    Order matters:
    1. Check for clean shutdown marker
    2. Start VaultKeeper (needed by others)
    3. Start Tavern (UI)
    4. Start Training Daemon
    5. Optionally start Weaver daemon

    Returns:
        True if all services started
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

    # Ensure dirs exist
    (base_dir / ".pids").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)

    weaver = Weaver(str(base_dir))

    # Start services in order
    services = ["vault", "tavern", "training"]

    for i, svc in enumerate(services, 1):
        print(f"[{i}/{len(services)}] Starting {weaver.threads[svc].name}...")
        success = weaver._start_thread(weaver.threads[svc])
        if not success:
            print(f"   ❌ Failed to start {svc}")
            return False
        print(f"   ✅ {svc} started")

    print("\n" + "=" * 60)
    print("✅ ALL SERVICES RUNNING")
    print("   Tavern:      http://localhost:8888")
    print("   VaultKeeper: http://localhost:8767")
    print("")
    print("   To run Weaver daemon: python3 weaver/weaver.py --daemon")
    print("=" * 60 + "\n")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="The Weaver - Daemon orchestrator")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (continuous)")
    parser.add_argument("--status", action="store_true", help="Show tapestry status")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't restart")
    parser.add_argument("--restart", metavar="SERVICE", help="Restart a specific service (tavern, vault, training, data_flow)")
    parser.add_argument("--shutdown", action="store_true", help="Force shutdown all services cleanly")
    parser.add_argument("--start", action="store_true", help="Start all services fresh")
    parser.add_argument("--base-dir", default=None, help="Base directory (auto-detected if not provided)")

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
        # Restart a specific service
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
