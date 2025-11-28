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

    def __init__(self, base_dir: str = "/path/to/training"):
        self.base_dir = Path(base_dir)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="The Weaver - Daemon orchestrator")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (continuous)")
    parser.add_argument("--status", action="store_true", help="Show tapestry status")
    parser.add_argument("--dry-run", action="store_true", help="Check but don't restart")
    parser.add_argument("--base-dir", default="/path/to/training", help="Base directory")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    # For daemon mode, check for existing Weaver
    if args.daemon:
        running, existing_pid = is_weaver_running(base_dir)
        if running:
            print(f"ERROR: Another Weaver is already running (PID {existing_pid})")
            print("Kill the existing Weaver first, or use --status to check status")
            sys.exit(1)

    weaver = Weaver(args.base_dir)

    if args.status:
        print(weaver.status())
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
