"""
ServiceWatcher - Generic HTTP service monitoring and auto-restart.

The ServiceWatcher monitors any HTTP service with a health endpoint and
automatically restarts it if it becomes unresponsive.

RPG Flavor:
    The ServiceWatcher is a vigilant scout who patrols the realm's
    various outposts (services). When an outpost falls silent, the
    scout sends reinforcements (restart) to bring it back online.

Usage:
    # Create watcher for Tavern
    watcher = ServiceWatcher(
        name="tavern",
        health_url="http://localhost:8888/health",
        start_cmd=["python3", "tavern/server.py", "--port", "8888"],
        pid_file=".pids/tavern.pid",
    )

    # Check health
    if not watcher.is_healthy():
        watcher.restart()

    # Run continuous watch (blocking)
    watcher.watch(interval=30, auto_restart=True)

    # Or run as daemon
    watcher.watch_daemon(interval=30)
"""

import logging
import os
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a watched service."""
    name: str
    health_url: str
    start_cmd: List[str]
    pid_file: Path
    log_file: Optional[Path] = None
    cwd: Optional[Path] = None
    health_timeout: float = 5.0
    startup_delay: float = 3.0
    max_restart_attempts: int = 3
    restart_cooldown: float = 30.0


@dataclass
class ServiceStatus:
    """Current status of a service."""
    name: str
    healthy: bool
    pid: Optional[int] = None
    pid_running: bool = False
    health_response: Optional[str] = None
    health_error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    uptime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "pid": self.pid,
            "pid_running": self.pid_running,
            "health_response": self.health_response,
            "health_error": self.health_error,
            "last_check": self.last_check.isoformat(),
            "uptime_seconds": self.uptime_seconds,
        }


class ServiceWatcher:
    """
    Watches an HTTP service and auto-restarts on failure.

    Features:
    - Health check via HTTP endpoint
    - PID file tracking
    - Graceful restart with cooldown
    - Restart attempt limiting
    - Logging
    """

    def __init__(
        self,
        name: str,
        health_url: str,
        start_cmd: List[str],
        pid_file: str | Path,
        log_file: Optional[str | Path] = None,
        cwd: Optional[str | Path] = None,
        health_timeout: float = 5.0,
        startup_delay: float = 3.0,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize a ServiceWatcher.

        Args:
            name: Human-readable service name
            health_url: URL to check for health (should return 200)
            start_cmd: Command to start the service
            pid_file: Path to PID file (relative to base_dir or absolute)
            log_file: Path to log file for service output
            cwd: Working directory for the service
            health_timeout: Timeout for health checks in seconds
            startup_delay: Seconds to wait after starting before health check
            base_dir: Base directory for relative paths
        """
        if base_dir is None:
            try:
                from core.paths import get_base_dir
                self.base_dir = get_base_dir()
            except ImportError:
                self.base_dir = Path(__file__).parent.parent  # Fallback
        else:
            self.base_dir = Path(base_dir)

        self.config = ServiceConfig(
            name=name,
            health_url=health_url,
            start_cmd=start_cmd,
            pid_file=self._resolve_path(pid_file),
            log_file=self._resolve_path(log_file) if log_file else None,
            cwd=self._resolve_path(cwd) if cwd else self.base_dir,
            health_timeout=health_timeout,
            startup_delay=startup_delay,
        )

        self._restart_attempts = 0
        self._last_restart: Optional[datetime] = None
        self._start_time: Optional[datetime] = None

    def _resolve_path(self, path: str | Path | None) -> Optional[Path]:
        """Resolve a path relative to base_dir if not absolute."""
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def get_pid(self) -> Optional[int]:
        """Get PID from file if it exists."""
        if not self.config.pid_file.exists():
            return None
        try:
            return int(self.config.pid_file.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def is_pid_running(self, pid: Optional[int] = None) -> bool:
        """Check if a process with the given PID is running."""
        pid = pid or self.get_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def check_health(self) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check service health via HTTP.

        Returns:
            Tuple of (healthy: bool, response: str | None, error: str | None)
        """
        try:
            req = urllib.request.Request(
                self.config.health_url,
                headers={"User-Agent": "ServiceWatcher/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self.config.health_timeout) as resp:
                body = resp.read().decode("utf-8")[:500]
                return (resp.status == 200, body, None)
        except urllib.error.HTTPError as e:
            return (False, None, f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            return (False, None, f"Connection failed: {e.reason}")
        except Exception as e:
            return (False, None, f"Error: {e}")

    def is_healthy(self) -> bool:
        """Quick health check."""
        healthy, _, _ = self.check_health()
        return healthy

    def get_status(self) -> ServiceStatus:
        """Get full service status."""
        pid = self.get_pid()
        pid_running = self.is_pid_running(pid)
        healthy, response, error = self.check_health()

        uptime = None
        if self._start_time and pid_running:
            uptime = (datetime.now() - self._start_time).total_seconds()

        return ServiceStatus(
            name=self.config.name,
            healthy=healthy,
            pid=pid,
            pid_running=pid_running,
            health_response=response,
            health_error=error,
            uptime_seconds=uptime,
        )

    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the service gracefully.

        Args:
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully
        """
        pid = self.get_pid()
        if pid is None:
            logger.info(f"[{self.config.name}] No PID file found")
            return True

        if not self.is_pid_running(pid):
            logger.info(f"[{self.config.name}] Process {pid} not running")
            # Clean up stale PID file
            self.config.pid_file.unlink(missing_ok=True)
            return True

        # Try graceful shutdown first (SIGTERM)
        logger.info(f"[{self.config.name}] Sending SIGTERM to {pid}")
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            return True

        # Wait for graceful shutdown
        for _ in range(int(timeout)):
            time.sleep(1)
            if not self.is_pid_running(pid):
                logger.info(f"[{self.config.name}] Stopped gracefully")
                return True

        # Force kill
        logger.warning(f"[{self.config.name}] Force killing {pid}")
        try:
            os.kill(pid, 9)  # SIGKILL
        except ProcessLookupError:
            pass

        time.sleep(1)
        return not self.is_pid_running(pid)

    def start(self) -> bool:
        """
        Start the service.

        Returns:
            True if started successfully
        """
        # Check if already running
        if self.is_pid_running():
            logger.warning(f"[{self.config.name}] Already running")
            return True

        # Clean up stale PID file
        self.config.pid_file.unlink(missing_ok=True)

        # Ensure directories exist
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{self.config.name}] Starting: {' '.join(self.config.start_cmd)}")

        try:
            # Open log file if configured
            log_handle = None
            if self.config.log_file:
                log_handle = open(self.config.log_file, "a")

            process = subprocess.Popen(
                self.config.start_cmd,
                stdout=log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=str(self.config.cwd) if self.config.cwd else None,
                start_new_session=True,
            )

            # Wait for startup
            time.sleep(self.config.startup_delay)

            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"[{self.config.name}] Failed to start (exit code {process.returncode})")
                return False

            # Record start time
            self._start_time = datetime.now()

            # Check health
            if self.is_healthy():
                logger.info(f"[{self.config.name}] Started successfully (PID {process.pid})")
                return True
            else:
                logger.warning(f"[{self.config.name}] Started but health check failed")
                return True  # Process is running, might just need more time

        except Exception as e:
            logger.error(f"[{self.config.name}] Failed to start: {e}")
            return False

    def restart(self) -> bool:
        """
        Restart the service.

        Returns:
            True if restarted successfully
        """
        # Check cooldown
        if self._last_restart:
            elapsed = (datetime.now() - self._last_restart).total_seconds()
            if elapsed < self.config.restart_cooldown:
                remaining = self.config.restart_cooldown - elapsed
                logger.warning(
                    f"[{self.config.name}] Restart cooldown ({remaining:.0f}s remaining)"
                )
                return False

        # Check max attempts
        if self._restart_attempts >= self.config.max_restart_attempts:
            logger.error(
                f"[{self.config.name}] Max restart attempts ({self.config.max_restart_attempts}) reached"
            )
            return False

        self._restart_attempts += 1
        self._last_restart = datetime.now()

        logger.info(
            f"[{self.config.name}] Restarting (attempt {self._restart_attempts}/{self.config.max_restart_attempts})"
        )

        if not self.stop():
            logger.error(f"[{self.config.name}] Failed to stop")
            return False

        time.sleep(1)

        if self.start():
            self._restart_attempts = 0  # Reset on success
            return True

        return False

    def reset_restart_counter(self) -> None:
        """Reset the restart attempt counter."""
        self._restart_attempts = 0
        self._last_restart = None

    def watch(
        self,
        interval: float = 30.0,
        auto_restart: bool = True,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Continuously watch the service.

        Args:
            interval: Seconds between health checks
            auto_restart: Automatically restart on failure
            max_iterations: Maximum checks (None for infinite)
        """
        logger.info(f"[{self.config.name}] Starting watch (interval={interval}s)")

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            status = self.get_status()

            if status.healthy:
                logger.debug(f"[{self.config.name}] Healthy (PID {status.pid})")
                self.reset_restart_counter()
            else:
                logger.warning(
                    f"[{self.config.name}] Unhealthy: {status.health_error}"
                )

                if auto_restart:
                    self.restart()

            iterations += 1
            time.sleep(interval)


# ==============================================================================
# PREDEFINED SERVICE CONFIGS
# ==============================================================================

def get_tavern_watcher(base_dir: Optional[Path] = None) -> ServiceWatcher:
    """Get a ServiceWatcher configured for the Tavern."""
    if base_dir is None:
        try:
            from core.paths import get_base_dir
            base = get_base_dir()
        except ImportError:
            base = Path(__file__).parent.parent
    else:
        base = Path(base_dir)
    return ServiceWatcher(
        name="tavern",
        health_url="http://localhost:8888/health",
        start_cmd=["python3", str(base / "tavern/server.py"), "--port", "8888", "--log-to-file"],
        pid_file=".pids/tavern.pid",
        log_file="logs/tavern.log",
        cwd=str(base),
        base_dir=base,
    )


def get_skill_watcher(
    skill_id: str,
    port: int,
    base_dir: Optional[Path] = None,
) -> ServiceWatcher:
    """Get a ServiceWatcher for a skill API."""
    # Skills live in a different directory (singleSKILL)
    # Default to ~/Desktop/singleSKILL or parent of TRAINING
    if base_dir is None:
        try:
            from core.paths import get_base_dir
            training_base = get_base_dir()
            # Assume singleSKILL is adjacent to TRAINING
            base = training_base.parent / "singleSKILL"
        except ImportError:
            base = Path.home() / "Desktop" / "singleSKILL"
    else:
        base = Path(base_dir)
    return ServiceWatcher(
        name=f"skill_{skill_id}",
        health_url=f"http://localhost:{port}/health",
        start_cmd=["python3", f"skill_{skill_id}/api_server.py", "--port", str(port)],
        pid_file=f".pids/skill_{skill_id}.pid",
        log_file=f"logs/skill_{skill_id}.log",
        cwd=str(base),
        base_dir=base,
    )


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="ServiceWatcher CLI")
    parser.add_argument("service", choices=["tavern", "sy", "bin"], help="Service to watch")
    parser.add_argument("--check", action="store_true", help="Just check health, don't watch")
    parser.add_argument("--restart", action="store_true", help="Restart the service")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--interval", type=float, default=30, help="Watch interval in seconds")

    args = parser.parse_args()

    # Get appropriate watcher
    if args.service == "tavern":
        watcher = get_tavern_watcher()
    elif args.service == "sy":
        watcher = get_skill_watcher("syllo_variant", 8080)
    elif args.service == "bin":
        watcher = get_skill_watcher("binary", 8090)

    if args.check:
        status = watcher.get_status()
        print(f"Service: {status.name}")
        print(f"Healthy: {status.healthy}")
        print(f"PID: {status.pid} (running: {status.pid_running})")
        if status.health_error:
            print(f"Error: {status.health_error}")

    elif args.stop:
        watcher.stop()

    elif args.restart:
        watcher.restart()

    else:
        # Watch mode
        watcher.watch(interval=args.interval, auto_restart=True)
