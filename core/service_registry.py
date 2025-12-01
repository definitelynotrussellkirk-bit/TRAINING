#!/usr/bin/env python3
"""
Service Registry - Unified service management loaded from configs/services.json

This is the single source of truth for service definitions. Both the Service Registry
and Weaver consume these definitions.

Service Dependency Graph (from configs/services.json):
    vault           → (none)          # VaultKeeper has no dependencies
    tavern          → vault           # Tavern needs VaultKeeper for API
    training        → vault, tavern   # Training needs both for status updates
    realm_state     → vault           # RealmState needs VaultKeeper
    eval_runner     → vault           # Eval runner needs vault for job store
    groundskeeper   → (none)          # Groundskeeper is independent
    weaver          → (none)          # Weaver orchestrates others
    data_flow       → (none)          # Special task (not a daemon)

Usage:
    from core.service_registry import get_service, get_all_services, start_service

    # Get a specific service
    vault = get_service("vault")

    # Start a service with dependencies
    start_service("training")

    # Bring up/down entire realm
    start_realm()
    stop_realm()
"""

import errno
import json
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


# ========== Data Classes ==========

class HealthCheckKind(Enum):
    """Type of health check."""
    HTTP = "http"
    PID = "pid"
    QUEUE_DEPTH = "queue_depth"  # Special: checks queue has enough data


@dataclass
class HealthCheck:
    """Health check configuration."""
    kind: HealthCheckKind
    target: str  # URL for HTTP, PID file path for PID, min_depth for queue
    timeout: float = 5.0
    path: Optional[str] = None  # HTTP path (appended to target URL)
    min_depth: int = 2  # For queue_depth checks


@dataclass
class StartupConfig:
    """Startup/restart configuration."""
    delay: float = 5.0  # Max seconds to wait for health after starting
    max_restarts: int = 3  # Max restarts per hour
    restart_delay: int = 5  # Seconds to wait before restart


@dataclass
class MonitoringConfig:
    """Multi-level monitoring configuration."""
    level2_check: Optional[str] = None  # e.g., "data_flow_health"
    level3_check: Optional[str] = None  # e.g., "performance_health"


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    id: str
    name: str
    description: str
    command: List[str]
    dependencies: List[str]
    health: HealthCheck
    pid_file: Optional[str] = None
    port: Optional[int] = None
    required: bool = True
    is_task: bool = False  # If True, not a daemon but a one-shot task
    startup: StartupConfig = field(default_factory=StartupConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Runtime state (not from config)
    restart_count: int = field(default=0, repr=False)
    last_restart: float = field(default=0.0, repr=False)


# ========== Service Loading ==========

_SERVICES: Dict[str, ServiceConfig] = {}
_SERVICES_LOADED: bool = False


def _load_services() -> Dict[str, ServiceConfig]:
    """Load services from configs/services.json."""
    global _SERVICES, _SERVICES_LOADED

    if _SERVICES_LOADED:
        return _SERVICES

    base_dir = get_base_dir()
    config_file = base_dir / "configs" / "services.json"

    if not config_file.exists():
        logger.warning(f"Services config not found: {config_file}")
        return {}

    try:
        data = json.loads(config_file.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in services config: {e}")
        return {}

    services = {}

    for svc_id, svc_data in data.items():
        # Skip schema comments
        if svc_id.startswith("$"):
            continue

        try:
            # Parse health check
            hc_data = svc_data.get("health_check", {})
            hc_type = hc_data.get("type", "pid")

            if hc_type == "http":
                # Build URL from port and path
                port = svc_data.get("port", 8080)
                path = hc_data.get("path", "/health")
                health = HealthCheck(
                    kind=HealthCheckKind.HTTP,
                    target=f"http://localhost:{port}{path}",
                    timeout=hc_data.get("timeout", 5.0),
                    path=path,
                )
            elif hc_type == "queue_depth":
                health = HealthCheck(
                    kind=HealthCheckKind.QUEUE_DEPTH,
                    target="queue",
                    timeout=hc_data.get("timeout", 5.0),
                    min_depth=hc_data.get("min_depth", 2),
                )
            else:
                # PID check - target is the PID file path
                pid_file = svc_data.get("pid_file", f".pids/{svc_id}.pid")
                health = HealthCheck(
                    kind=HealthCheckKind.PID,
                    target=str(base_dir / pid_file),
                    timeout=hc_data.get("timeout", 5.0),
                )

            # Parse startup config
            startup_data = svc_data.get("startup", {})
            startup = StartupConfig(
                delay=startup_data.get("delay", 5.0),
                max_restarts=startup_data.get("max_restarts", 3),
                restart_delay=startup_data.get("restart_delay", 5),
            )

            # Parse monitoring config
            mon_data = svc_data.get("monitoring", {})
            monitoring = MonitoringConfig(
                level2_check=mon_data.get("level2_check"),
                level3_check=mon_data.get("level3_check"),
            )

            # Resolve PID file path
            pid_file_str = svc_data.get("pid_file")
            if pid_file_str:
                pid_file_path = str(base_dir / pid_file_str)
            else:
                pid_file_path = None

            # Build command with absolute paths where needed
            command = svc_data.get("command", [])

            # Create service config
            services[svc_id] = ServiceConfig(
                id=svc_id,
                name=svc_data.get("name", svc_id),
                description=svc_data.get("description", ""),
                command=command,
                dependencies=svc_data.get("depends_on", []),
                health=health,
                pid_file=pid_file_path,
                port=svc_data.get("port"),
                required=svc_data.get("required", True),
                is_task=svc_data.get("is_task", False),
                startup=startup,
                monitoring=monitoring,
            )

        except Exception as e:
            logger.error(f"Failed to parse service '{svc_id}': {e}")
            continue

    _SERVICES = services
    _SERVICES_LOADED = True
    logger.debug(f"Loaded {len(services)} services from config")

    return services


def get_service(service_id: str) -> Optional[ServiceConfig]:
    """Get a service by ID."""
    services = _load_services()
    return services.get(service_id)


def get_all_services() -> Dict[str, ServiceConfig]:
    """Get all services."""
    return _load_services()


def reload_services():
    """Force reload of services from config file."""
    global _SERVICES, _SERVICES_LOADED
    _SERVICES = {}
    _SERVICES_LOADED = False
    return _load_services()


# ========== Health Checks ==========

def _check_health(service: ServiceConfig, base_dir: Path) -> bool:
    """
    Check if a service is healthy.

    HTTP checks: GET the URL, expect 200
    PID checks: Verify PID file exists and process is alive
    Queue checks: Verify queue has enough data
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
            return False
        except OSError as e:
            if e.errno == errno.ESRCH:
                return False
            elif e.errno == errno.EPERM:
                return True  # Process exists but no permission
            return False

    elif hc.kind == HealthCheckKind.QUEUE_DEPTH:
        try:
            from core.training_queue import TrainingQueue
            queue = TrainingQueue(base_dir)
            status = queue.get_queue_status()
            available = status["total_queued"] + status.get("processing", 0)
            return available >= hc.min_depth
        except Exception:
            return True  # Assume OK if can't check

    logger.warning(f"Unknown health check kind: {hc.kind}")
    return False


def is_service_running(service_id: str) -> bool:
    """Check if a service is running."""
    service = get_service(service_id)
    if not service:
        logger.warning(f"Unknown service: {service_id}")
        return False

    return _check_health(service, get_base_dir())


# ========== Service Start/Stop ==========

def _start_service(service: ServiceConfig, base_dir: Path) -> bool:
    """
    Start a service with multi-try health check.

    Polls for readiness within the start_delay window.
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
                service.command,
                stdout=log,
                stderr=log,
                cwd=str(base_dir),
                start_new_session=True
            )

        # Multi-try readiness: poll until healthy or timeout
        deadline = time.time() + service.startup.delay
        poll_interval = 0.5

        while time.time() < deadline:
            if _check_health(service, base_dir):
                logger.info(f"  ✓ {service.name} started successfully")
                return True
            time.sleep(poll_interval)

        # Final check
        if _check_health(service, base_dir):
            logger.info(f"  ✓ {service.name} started successfully")
            return True

        logger.warning(f"  ✗ {service.name} started but health check failed within {service.startup.delay}s")
        return False

    except Exception as e:
        logger.error(f"  ✗ Failed to start {service.name}: {e}")
        return False


def start_service(service_id: str, ensure_deps: bool = True) -> bool:
    """
    Start a service, optionally ensuring its dependencies first.

    Args:
        service_id: ID of service to start
        ensure_deps: If True, start dependencies first

    Returns:
        True if service started successfully
    """
    service = get_service(service_id)
    if not service:
        logger.error(f"Unknown service: {service_id}")
        return False

    base_dir = get_base_dir()

    # Check if already running
    if _check_health(service, base_dir):
        logger.info(f"{service.name} is already running")
        return True

    # Start dependencies first
    if ensure_deps:
        for dep_id in service.dependencies:
            dep = get_service(dep_id)
            if not dep:
                logger.error(f"Unknown dependency: {dep_id}")
                return False

            if not _check_health(dep, base_dir):
                logger.info(f"{service.name} requires {dep.name}, starting it first...")
                if not _start_service(dep, base_dir):
                    logger.error(f"Failed to start dependency {dep.name}")
                    return False

    # Start the service
    return _start_service(service, base_dir)


def stop_service(service_id: str) -> bool:
    """Stop a service gracefully."""
    service = get_service(service_id)
    if not service:
        logger.error(f"Unknown service: {service_id}")
        return False

    if not service.pid_file:
        logger.warning(f"{service.name} has no PID file, cannot stop")
        return False

    pid_path = Path(service.pid_file)
    if not pid_path.exists():
        logger.info(f"{service.name} is not running (no PID file)")
        return True

    try:
        pid = int(pid_path.read_text().strip())

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


def ensure_dependencies(service_id: str) -> bool:
    """
    Ensure all dependencies for a service are running.

    Args:
        service_id: Service whose dependencies should be checked

    Returns:
        True if all dependencies are running
    """
    service = get_service(service_id)
    if not service:
        logger.warning(f"Unknown service: {service_id}")
        return True

    base_dir = get_base_dir()
    all_ok = True

    for dep_id in service.dependencies:
        dep = get_service(dep_id)
        if not dep:
            logger.warning(f"Unknown dependency: {dep_id}")
            continue

        if _check_health(dep, base_dir):
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


# ========== Status and Info ==========

def get_service_status() -> Dict[str, dict]:
    """Get status of all services."""
    services = _load_services()
    base_dir = get_base_dir()
    status = {}

    for svc_id, service in services.items():
        running = _check_health(service, base_dir)
        status[svc_id] = {
            "name": service.name,
            "description": service.description,
            "running": running,
            "required": service.required,
            "is_task": service.is_task,
            "dependencies": service.dependencies,
            "port": service.port,
        }

    return status


def get_dependency_order() -> List[str]:
    """Get services in dependency order (dependencies first)."""
    services = _load_services()

    order = []
    visited = set()

    def visit(svc_id: str):
        if svc_id in visited:
            return
        visited.add(svc_id)

        service = services.get(svc_id)
        if service:
            for dep in service.dependencies:
                visit(dep)

        order.append(svc_id)

    for svc_id in services:
        visit(svc_id)

    return order


# ========== Realm-Level Operations ==========

def start_realm(include_optional: bool = False, exclude_tasks: bool = True) -> bool:
    """
    Start all services in the Realm in dependency order.

    Args:
        include_optional: If True, also start optional services
        exclude_tasks: If True, skip task-type services (data_flow)

    Returns:
        True if all required services started successfully
    """
    services = _load_services()
    order = get_dependency_order()
    base_dir = get_base_dir()
    all_ok = True
    started = []
    failed = []

    logger.info("=" * 50)
    logger.info("Starting Realm...")
    logger.info("=" * 50)

    for svc_id in order:
        service = services.get(svc_id)
        if not service:
            continue

        # Skip optional services unless requested
        if not service.required and not include_optional:
            logger.debug(f"Skipping optional service: {svc_id}")
            continue

        # Skip tasks unless they're included
        if service.is_task and exclude_tasks:
            logger.debug(f"Skipping task: {svc_id}")
            continue

        # Check if already running
        if _check_health(service, base_dir):
            logger.info(f"  ✓ {service.name} already running")
            started.append(svc_id)
            continue

        # Start the service
        if _start_service(service, base_dir):
            started.append(svc_id)
        else:
            failed.append(svc_id)
            if service.required:
                logger.error(f"Required service {service.name} failed to start, aborting")
                all_ok = False
                break

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

    Returns:
        True if all services stopped successfully
    """
    services = _load_services()
    order = list(reversed(get_dependency_order()))
    base_dir = get_base_dir()
    all_ok = True
    stopped = []
    failed = []

    logger.info("=" * 50)
    logger.info("Stopping Realm...")
    logger.info("=" * 50)

    for svc_id in order:
        service = services.get(svc_id)
        if not service:
            continue

        # Skip tasks
        if service.is_task:
            continue

        # Check if running
        if not _check_health(service, base_dir):
            logger.debug(f"  {service.name} is not running")
            continue

        # Stop the service
        if stop_service(svc_id):
            stopped.append(svc_id)
        else:
            failed.append(svc_id)
            all_ok = False

    logger.info("=" * 50)
    if all_ok:
        logger.info(f"Realm stopped: {len(stopped)} services stopped")
    else:
        logger.warning(f"Realm stop incomplete: {len(stopped)} stopped, {len(failed)} failed")
    logger.info("=" * 50)

    return all_ok


# ========== CLI ==========

def main():
    """CLI for service management."""
    import argparse

    parser = argparse.ArgumentParser(description="Service Registry")
    parser.add_argument("command", choices=[
        "status", "start", "stop", "deps", "order",
        "realm-up", "realm-down", "list", "reload"
    ], help="Command to run")
    parser.add_argument("service", nargs="?", help="Service ID (for start/stop/deps)")
    parser.add_argument("--all", action="store_true", help="Include optional services (for realm-up)")
    parser.add_argument("--tasks", action="store_true", help="Include task services (for realm-up)")

    args = parser.parse_args()

    if args.command == "status":
        status = get_service_status()
        print("\n" + "=" * 60)
        print("SERVICE STATUS")
        print("=" * 60)
        for svc_id, info in status.items():
            icon = "✓" if info["running"] else "✗"
            req = "" if info["required"] else " (optional)"
            task = " [task]" if info["is_task"] else ""
            port = f" :{info['port']}" if info['port'] else ""
            deps = f" [deps: {', '.join(info['dependencies'])}]" if info['dependencies'] else ""
            print(f"  {icon} {info['name']}{port}{req}{task}{deps}")
        print("=" * 60)

    elif args.command == "list":
        services = get_all_services()
        print("\nConfigured Services:")
        for svc_id, svc in services.items():
            print(f"  {svc_id}: {svc.name}")
            print(f"    Command: {' '.join(svc.command)}")
            print(f"    Port: {svc.port or 'N/A'}")
            print(f"    Required: {svc.required}")
            print(f"    Dependencies: {svc.dependencies or 'none'}")
            print()

    elif args.command == "start":
        if not args.service:
            print("Error: service ID required")
            sys.exit(1)
        success = start_service(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "stop":
        if not args.service:
            print("Error: service ID required")
            sys.exit(1)
        success = stop_service(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "deps":
        if not args.service:
            print("Error: service ID required")
            sys.exit(1)
        success = ensure_dependencies(args.service)
        sys.exit(0 if success else 1)

    elif args.command == "order":
        order = get_dependency_order()
        print("Start order (dependencies first):")
        for i, svc_id in enumerate(order, 1):
            svc = get_service(svc_id)
            if svc:
                print(f"  {i}. {svc_id} ({svc.name})")

    elif args.command == "realm-up":
        success = start_realm(include_optional=args.all, exclude_tasks=not args.tasks)
        sys.exit(0 if success else 1)

    elif args.command == "realm-down":
        success = stop_realm()
        sys.exit(0 if success else 1)

    elif args.command == "reload":
        services = reload_services()
        print(f"Reloaded {len(services)} services from config")


if __name__ == "__main__":
    main()
