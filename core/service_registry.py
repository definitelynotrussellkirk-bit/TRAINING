#!/usr/bin/env python3
"""
Service Registry - Dependency graph and auto-start for services

Each service can ensure its dependencies are running before starting.
This implements the "if ONE system is online, relevant OTHER systems are online too" principle.

Service Dependency Graph:
    vault           → (none)          # VaultKeeper has no dependencies
    tavern          → vault           # Tavern needs VaultKeeper for API
    training        → vault, tavern   # Training needs both for status updates
    eval_runner     → vault           # Eval runner needs vault for job store
    groundskeeper   → (none)          # Groundskeeper is independent
    realm_state     → vault           # RealmState needs VaultKeeper
    weaver          → (none)          # Weaver orchestrates others

Usage:
    from core.service_registry import ensure_dependencies, start_service

    # In your daemon startup:
    if not ensure_dependencies("training"):
        print("Failed to start dependencies")
        sys.exit(1)

    # Or start a specific service:
    start_service("tavern")
"""

import errno
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthCheckKind(Enum):
    """Type of health check."""
    HTTP = "http"
    PID = "pid"


@dataclass
class HealthCheck:
    """
    Health check configuration.

    Supports two kinds:
    - HTTP: GET request to a URL, expect 200
    - PID: Check if PID file exists and process is alive
    """
    kind: HealthCheckKind
    target: str  # URL for HTTP, PID file path for PID
    timeout: float = 5.0


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    description: str
    dependencies: List[str]  # Services that must be running first
    health: HealthCheck  # Health check configuration
    start_cmd: List[str]
    pid_file: Optional[str] = None
    port: Optional[int] = None
    required: bool = True
    start_delay: float = 5.0  # Max seconds to wait for health after starting


# Service definitions
SERVICES: Dict[str, ServiceConfig] = {}


def _init_services():
    """Initialize service configurations."""
    global SERVICES
    base_dir = get_base_dir()

    SERVICES = {
        "vault": ServiceConfig(
            name="VaultKeeper",
            description="Asset registry and job store",
            dependencies=[],  # No dependencies
            health=HealthCheck(HealthCheckKind.HTTP, "http://localhost:8767/health"),
            start_cmd=["python3", str(base_dir / "vault" / "server.py"), "--port", "8767"],
            pid_file=str(base_dir / ".pids" / "vault.pid"),
            port=8767,
        ),
        "tavern": ServiceConfig(
            name="Tavern",
            description="Game UI",
            dependencies=["vault"],  # Needs VaultKeeper
            health=HealthCheck(HealthCheckKind.HTTP, "http://localhost:8888/health"),
            start_cmd=["python3", str(base_dir / "tavern" / "server.py"), "--port", "8888"],
            pid_file=str(base_dir / ".pids" / "tavern.pid"),
            port=8888,
        ),
        "training": ServiceConfig(
            name="Training Daemon",
            description="Training orchestrator",
            dependencies=["vault", "tavern"],  # Needs both
            health=HealthCheck(HealthCheckKind.PID, str(base_dir / ".daemon.pid")),
            start_cmd=["python3", str(base_dir / "core" / "training_daemon.py"), "--base-dir", str(base_dir)],
            pid_file=str(base_dir / ".daemon.pid"),
        ),
        "eval_runner": ServiceConfig(
            name="Eval Runner",
            description="Evaluation processor",
            dependencies=["vault"],  # Needs VaultKeeper for job store
            health=HealthCheck(HealthCheckKind.PID, str(base_dir / ".pids" / "eval_runner.pid")),
            start_cmd=[
                "python3", str(base_dir / "core" / "eval_runner.py"),
                "--daemon", "--interval", "60"
            ],
            pid_file=str(base_dir / ".pids" / "eval_runner.pid"),
            required=False,  # Optional
        ),
        "realm_state": ServiceConfig(
            name="RealmState",
            description="State service",
            dependencies=["vault"],
            health=HealthCheck(HealthCheckKind.HTTP, "http://localhost:8866/health"),
            start_cmd=["python3", "-m", "realm.server", "--host", "0.0.0.0", "--port", "8866"],
            pid_file=str(base_dir / ".pids" / "realm_state.pid"),
            port=8866,
        ),
        "groundskeeper": ServiceConfig(
            name="Groundskeeper",
            description="Cleanup daemon",
            dependencies=[],  # Independent
            health=HealthCheck(HealthCheckKind.PID, str(base_dir / ".pids" / "groundskeeper.pid")),
            start_cmd=["python3", str(base_dir / "core" / "groundskeeper.py"), "--daemon"],
            pid_file=str(base_dir / ".pids" / "groundskeeper.pid"),
            required=False,
        ),
        "weaver": ServiceConfig(
            name="Weaver",
            description="Daemon orchestrator",
            dependencies=[],  # Orchestrates others, doesn't depend on them
            health=HealthCheck(HealthCheckKind.PID, str(base_dir / ".pids" / "weaver.pid")),
            start_cmd=["python3", str(base_dir / "weaver" / "weaver.py"), "--daemon"],
            pid_file=str(base_dir / ".pids" / "weaver.pid"),
            required=False,
        ),
    }


def _check_health(service: ServiceConfig) -> bool:
    """
    Check if a service is healthy.

    HTTP checks: GET the URL, expect 200
    PID checks: Verify PID file exists and process is alive
    """
    hc = service.health

    if hc.kind == HealthCheckKind.HTTP:
        try:
            resp = requests.get(hc.target, timeout=hc.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    elif hc.kind == HealthCheckKind.PID:
        pid_file = Path(hc.target)
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except ValueError:
            # Invalid PID in file
            return False
        except OSError as e:
            if e.errno == errno.ESRCH:
                # ESRCH: No such process - definitely dead
                return False
            elif e.errno == errno.EPERM:
                # EPERM: Process exists but no permission to signal
                # Treat as alive (we just can't signal it)
                return True
            else:
                # Other OS error
                return False

    logger.warning(f"Unknown health check kind: {hc.kind}")
    return False


def _start_service(service: ServiceConfig, base_dir: Path) -> bool:
    """
    Start a service with multi-try health check.

    Instead of sleeping once and checking once, we poll for readiness
    within the start_delay window. This handles slow-booting services.
    """
    logger.info(f"Starting {service.name}...")

    # Ensure PID directory exists
    if service.pid_file:
        Path(service.pid_file).parent.mkdir(parents=True, exist_ok=True)

    # Ensure logs directory exists
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{service.name.lower().replace(' ', '_')}.log"

    try:
        with open(log_file, 'a') as log:
            process = subprocess.Popen(
                service.start_cmd,
                stdout=log,
                stderr=log,
                cwd=str(base_dir),
                start_new_session=True
            )

        # Multi-try readiness: poll until healthy or timeout
        deadline = time.time() + service.start_delay
        poll_interval = 0.5  # Check every 500ms

        while time.time() < deadline:
            if _check_health(service):
                logger.info(f"  ✓ {service.name} started successfully")
                return True
            time.sleep(poll_interval)

        # Final check after timeout
        if _check_health(service):
            logger.info(f"  ✓ {service.name} started successfully")
            return True

        logger.warning(f"  ✗ {service.name} started but health check failed within {service.start_delay}s")
        return False

    except Exception as e:
        logger.error(f"  ✗ Failed to start {service.name}: {e}")
        return False


def is_service_running(service_name: str) -> bool:
    """Check if a service is running."""
    if not SERVICES:
        _init_services()

    if service_name not in SERVICES:
        logger.warning(f"Unknown service: {service_name}")
        return False

    return _check_health(SERVICES[service_name])


def start_service(service_name: str, ensure_deps: bool = True) -> bool:
    """
    Start a service, optionally ensuring its dependencies first.

    Args:
        service_name: Name of service to start
        ensure_deps: If True, start dependencies first

    Returns:
        True if service started successfully
    """
    if not SERVICES:
        _init_services()

    if service_name not in SERVICES:
        logger.error(f"Unknown service: {service_name}")
        return False

    service = SERVICES[service_name]
    base_dir = get_base_dir()

    # Check if already running
    if _check_health(service):
        logger.info(f"{service.name} is already running")
        return True

    # Start dependencies first
    if ensure_deps:
        for dep_name in service.dependencies:
            if dep_name not in SERVICES:
                logger.error(f"Unknown dependency: {dep_name}")
                return False

            dep = SERVICES[dep_name]

            if not _check_health(dep):
                logger.info(f"{service.name} requires {dep.name}, starting it first...")
                if not _start_service(dep, base_dir):
                    logger.error(f"Failed to start dependency {dep.name}")
                    return False

    # Start the service
    return _start_service(service, base_dir)


def ensure_dependencies(service_name: str) -> bool:
    """
    Ensure all dependencies for a service are running.

    Call this at the start of your daemon to auto-start dependencies.

    Args:
        service_name: Service whose dependencies should be checked

    Returns:
        True if all dependencies are running (or were started)
    """
    if not SERVICES:
        _init_services()

    if service_name not in SERVICES:
        logger.warning(f"Unknown service: {service_name}")
        return True  # Don't block unknown services

    service = SERVICES[service_name]
    base_dir = get_base_dir()

    all_ok = True

    for dep_name in service.dependencies:
        if dep_name not in SERVICES:
            logger.warning(f"Unknown dependency: {dep_name}")
            continue

        dep = SERVICES[dep_name]

        if _check_health(dep):
            logger.debug(f"Dependency {dep.name} is healthy")
        else:
            logger.info(f"Dependency {dep.name} is not running, starting it...")
            if _start_service(dep, base_dir):
                logger.info(f"  ✓ Started {dep.name}")
            else:
                logger.error(f"  ✗ Failed to start {dep.name}")
                if dep.required:
                    all_ok = False

    return all_ok


def stop_service(service_name: str) -> bool:
    """Stop a service gracefully."""
    if not SERVICES:
        _init_services()

    if service_name not in SERVICES:
        logger.error(f"Unknown service: {service_name}")
        return False

    service = SERVICES[service_name]

    if not service.pid_file:
        logger.warning(f"{service.name} has no PID file, cannot stop")
        return False

    pid_path = Path(service.pid_file)
    if not pid_path.exists():
        logger.info(f"{service.name} is not running (no PID file)")
        return True

    try:
        pid = int(pid_path.read_text().strip())

        # Send SIGTERM
        logger.info(f"Stopping {service.name} (PID {pid})...")
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            # Force kill
            logger.warning(f"Force killing {service.name}...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        pid_path.unlink(missing_ok=True)
        logger.info(f"  ✓ {service.name} stopped")
        return True

    except (ValueError, OSError, ProcessLookupError) as e:
        logger.warning(f"Error stopping {service.name}: {e}")
        pid_path.unlink(missing_ok=True)
        return True


def get_service_status() -> Dict[str, dict]:
    """Get status of all services."""
    if not SERVICES:
        _init_services()

    status = {}

    for name, service in SERVICES.items():
        running = _check_health(service)
        status[name] = {
            "name": service.name,
            "description": service.description,
            "running": running,
            "required": service.required,
            "dependencies": service.dependencies,
            "port": service.port,
        }

    return status


def get_dependency_order() -> List[str]:
    """Get services in dependency order (dependencies first)."""
    if not SERVICES:
        _init_services()

    # Topological sort
    order = []
    visited = set()

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)

        service = SERVICES.get(name)
        if service:
            for dep in service.dependencies:
                visit(dep)

        order.append(name)

    for name in SERVICES:
        visit(name)

    return order


# ========== Realm-Level Operations ==========

def start_realm(include_optional: bool = False) -> bool:
    """
    Start all services in the Realm in dependency order.

    Args:
        include_optional: If True, also start optional services (groundskeeper, weaver, etc.)

    Returns:
        True if all required services started successfully
    """
    if not SERVICES:
        _init_services()

    order = get_dependency_order()
    base_dir = get_base_dir()
    all_ok = True
    started = []
    failed = []

    logger.info("=" * 50)
    logger.info("Starting Realm...")
    logger.info("=" * 50)

    for name in order:
        service = SERVICES[name]

        # Skip optional services unless requested
        if not service.required and not include_optional:
            logger.debug(f"Skipping optional service: {name}")
            continue

        # Check if already running
        if _check_health(service):
            logger.info(f"  ✓ {service.name} already running")
            started.append(name)
            continue

        # Start the service
        if _start_service(service, base_dir):
            started.append(name)
        else:
            failed.append(name)
            if service.required:
                logger.error(f"Required service {service.name} failed to start, aborting")
                all_ok = False
                break

    # Summary
    logger.info("=" * 50)
    if all_ok:
        logger.info(f"Realm started: {len(started)} services running")
    else:
        logger.error(f"Realm startup failed: {len(started)} started, {len(failed)} failed")
    logger.info("=" * 50)

    return all_ok


def stop_realm() -> bool:
    """
    Stop all services in the Realm in reverse dependency order.

    Stops dependents before their dependencies.

    Returns:
        True if all services stopped successfully
    """
    if not SERVICES:
        _init_services()

    # Reverse order so dependents stop first
    order = list(reversed(get_dependency_order()))
    all_ok = True
    stopped = []
    failed = []

    logger.info("=" * 50)
    logger.info("Stopping Realm...")
    logger.info("=" * 50)

    for name in order:
        service = SERVICES[name]

        # Check if running
        if not _check_health(service):
            logger.debug(f"  {service.name} is not running")
            continue

        # Stop the service
        if stop_service(name):
            stopped.append(name)
        else:
            failed.append(name)
            all_ok = False

    # Summary
    logger.info("=" * 50)
    if all_ok:
        logger.info(f"Realm stopped: {len(stopped)} services stopped")
    else:
        logger.warning(f"Realm stop incomplete: {len(stopped)} stopped, {len(failed)} failed")
    logger.info("=" * 50)

    return all_ok


def main():
    """CLI for service management."""
    import argparse

    parser = argparse.ArgumentParser(description="Service Registry")
    parser.add_argument("command", choices=["status", "start", "stop", "deps", "order", "realm-up", "realm-down"],
                       help="Command to run")
    parser.add_argument("service", nargs="?", help="Service name (for start/stop/deps)")
    parser.add_argument("--all", action="store_true", help="Include optional services (for realm-up)")

    args = parser.parse_args()

    if args.command == "status":
        status = get_service_status()
        print("\n" + "=" * 60)
        print("SERVICE STATUS")
        print("=" * 60)
        for name, info in status.items():
            icon = "✓" if info["running"] else "✗"
            req = "" if info["required"] else " (optional)"
            port = f" :{info['port']}" if info['port'] else ""
            deps = f" [deps: {', '.join(info['dependencies'])}]" if info['dependencies'] else ""
            print(f"  {icon} {info['name']}{port}{req}{deps}")
        print("=" * 60)

    elif args.command == "start":
        if not args.service:
            print("Error: service name required")
            sys.exit(1)
        success = start_service(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "stop":
        if not args.service:
            print("Error: service name required")
            sys.exit(1)
        success = stop_service(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "deps":
        if not args.service:
            print("Error: service name required")
            sys.exit(1)
        success = ensure_dependencies(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "order":
        order = get_dependency_order()
        print("Start order (dependencies first):")
        for i, name in enumerate(order, 1):
            print(f"  {i}. {name}")

    elif args.command == "realm-up":
        success = start_realm(include_optional=args.all)
        sys.exit(0 if success else 1)

    elif args.command == "realm-down":
        success = stop_realm()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
