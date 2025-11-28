"""
Zone Warden - Zone-level service monitor and health aggregator.

The Warden is the chief overseer of a zone. It monitors all services running
in its territory and reports aggregate health status. Other zones query the
Warden to know if a zone is operational.

Responsibilities:
    - Monitor health of all services in the zone
    - Aggregate status into zone-level health (online/degraded/offline)
    - Expose /health, /services, /service/{id} endpoints
    - Cache health checks to serve fast responses
    - Optionally restart crashed services (phase 2)

Usage:
    # On 4090 (Training Server)
    python3 vault/zone_warden.py --zone 4090

    # On 3090 (Inference Server)
    python3 vault/zone_warden.py --zone 3090

    # On NAS (Storage)
    python3 vault/zone_warden.py --zone nas

    # Query from anywhere
    curl http://192.168.x.x:8760/health
    curl http://192.168.x.x:8760/services

RPG Flavor:
    The Warden keeps vigilant watch over their assigned territory.
    They know the state of every garrison (service) under their command.
    When the Realm needs to know if a zone is operational, they ask the Warden.
    A good Warden never sleeps - they patrol continuously, ready to report.
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Version
ZONE_WARDEN_VERSION = "1.0.0"

# Defaults
DEFAULT_PORT = 8760
DEFAULT_CHECK_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 5  # seconds for HTTP checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("zone_warden")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ServiceStatus:
    """Status of a single service."""
    service_id: str
    name: str
    status: str  # online, offline, unknown
    service_type: str = "http"  # http, process, tcp
    port: Optional[int] = None
    health_endpoint: Optional[str] = None
    critical: bool = False
    response_ms: Optional[int] = None
    error: Optional[str] = None
    last_check: Optional[str] = None
    pid: Optional[int] = None  # for process type
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneHealth:
    """Aggregate health of a zone."""
    zone_id: str
    zone_name: str
    zone_role: str
    status: str  # online, degraded, offline
    host: str
    warden_version: str = ZONE_WARDEN_VERSION
    services: Dict[str, ServiceStatus] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)
    last_patrol: Optional[str] = None
    uptime_seconds: int = 0


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def find_config_file() -> Path:
    """Find hosts.json config file."""
    # Try relative to this file
    here = Path(__file__).parent.parent
    candidates = [
        here / "config" / "hosts.json",
        Path("/path/to/training/config/hosts.json"),
        Path.cwd() / "config" / "hosts.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find hosts.json in {candidates}")


def load_zone_config(zone_id: str) -> Dict[str, Any]:
    """Load configuration for a specific zone."""
    config_path = find_config_file()
    with open(config_path) as f:
        config = json.load(f)

    if zone_id not in config.get("hosts", {}):
        raise ValueError(f"Unknown zone: {zone_id}. Known zones: {list(config['hosts'].keys())}")

    zone = config["hosts"][zone_id]
    zone["zone_id"] = zone_id
    zone["warden_port"] = config.get("warden_port", DEFAULT_PORT)

    return zone


# =============================================================================
# SERVICE CHECKERS
# =============================================================================

class ServiceChecker:
    """Checks health of services."""

    def __init__(self, base_dir: Path, timeout: int = DEFAULT_TIMEOUT):
        self.base_dir = base_dir
        self.timeout = timeout

    def check_service(self, service_id: str, config: Dict[str, Any]) -> ServiceStatus:
        """Check a single service based on its type."""
        service_type = config.get("type", "http")
        name = config.get("name", service_id)
        critical = config.get("critical", False)

        if service_type == "process":
            return self._check_process(service_id, name, config, critical)
        elif service_type == "tcp":
            return self._check_tcp(service_id, name, config, critical)
        else:
            return self._check_http(service_id, name, config, critical)

    def _check_http(self, service_id: str, name: str, config: Dict[str, Any],
                    critical: bool) -> ServiceStatus:
        """Check HTTP service via health endpoint."""
        port = config.get("port")
        health = config.get("health", "/health")

        if not port:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="unknown",
                service_type="http",
                critical=critical,
                error="No port configured",
                last_check=datetime.now().isoformat()
            )

        url = f"http://localhost:{port}{health}"
        start = time.time()

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed_ms = int((time.time() - start) * 1000)
                status = "online" if resp.status == 200 else "degraded"

                # Try to parse response for extra info
                extra = {}
                try:
                    body = json.loads(resp.read().decode())
                    if isinstance(body, dict):
                        extra = {k: v for k, v in body.items()
                                if k in ("version", "status", "uptime", "step", "training")}
                except:
                    pass

                return ServiceStatus(
                    service_id=service_id,
                    name=name,
                    status=status,
                    service_type="http",
                    port=port,
                    health_endpoint=health,
                    critical=critical,
                    response_ms=elapsed_ms,
                    last_check=datetime.now().isoformat(),
                    extra=extra
                )

        except urllib.error.URLError as e:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="http",
                port=port,
                health_endpoint=health,
                critical=critical,
                error=str(e.reason),
                last_check=datetime.now().isoformat()
            )
        except Exception as e:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="http",
                port=port,
                health_endpoint=health,
                critical=critical,
                error=str(e),
                last_check=datetime.now().isoformat()
            )

    def _check_process(self, service_id: str, name: str, config: Dict[str, Any],
                       critical: bool) -> ServiceStatus:
        """Check if a process is running via PID file."""
        pid_file = config.get("pid_file")

        if not pid_file:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="unknown",
                service_type="process",
                critical=critical,
                error="No pid_file configured",
                last_check=datetime.now().isoformat()
            )

        pid_path = self.base_dir / pid_file

        if not pid_path.exists():
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="process",
                critical=critical,
                error="PID file not found",
                last_check=datetime.now().isoformat()
            )

        try:
            pid = int(pid_path.read_text().strip())

            # Check if process is actually running
            # On Linux, /proc/{pid} exists if process is running
            if os.path.exists(f"/proc/{pid}"):
                return ServiceStatus(
                    service_id=service_id,
                    name=name,
                    status="online",
                    service_type="process",
                    critical=critical,
                    pid=pid,
                    last_check=datetime.now().isoformat()
                )
            else:
                return ServiceStatus(
                    service_id=service_id,
                    name=name,
                    status="offline",
                    service_type="process",
                    critical=critical,
                    error=f"PID {pid} not running",
                    last_check=datetime.now().isoformat()
                )

        except ValueError:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="process",
                critical=critical,
                error="Invalid PID file",
                last_check=datetime.now().isoformat()
            )

    def _check_tcp(self, service_id: str, name: str, config: Dict[str, Any],
                   critical: bool) -> ServiceStatus:
        """Check if a TCP port is open."""
        port = config.get("port")

        if not port:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="unknown",
                service_type="tcp",
                critical=critical,
                error="No port configured",
                last_check=datetime.now().isoformat()
            )

        start = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            elapsed_ms = int((time.time() - start) * 1000)

            if result == 0:
                return ServiceStatus(
                    service_id=service_id,
                    name=name,
                    status="online",
                    service_type="tcp",
                    port=port,
                    critical=critical,
                    response_ms=elapsed_ms,
                    last_check=datetime.now().isoformat()
                )
            else:
                return ServiceStatus(
                    service_id=service_id,
                    name=name,
                    status="offline",
                    service_type="tcp",
                    port=port,
                    critical=critical,
                    error=f"Connection refused (code {result})",
                    last_check=datetime.now().isoformat()
                )

        except socket.timeout:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="tcp",
                port=port,
                critical=critical,
                error="Connection timeout",
                last_check=datetime.now().isoformat()
            )
        except Exception as e:
            return ServiceStatus(
                service_id=service_id,
                name=name,
                status="offline",
                service_type="tcp",
                port=port,
                critical=critical,
                error=str(e),
                last_check=datetime.now().isoformat()
            )


# =============================================================================
# ZONE WARDEN
# =============================================================================

class ZoneWarden:
    """
    The Warden monitors all services in a zone and aggregates health.
    """

    def __init__(self, zone_config: Dict[str, Any], base_dir: Path,
                 check_interval: int = DEFAULT_CHECK_INTERVAL):
        self.zone_id = zone_config["zone_id"]
        self.zone_name = zone_config.get("name", self.zone_id)
        self.zone_role = zone_config.get("role", "unknown")
        self.zone_host = zone_config.get("host", "localhost")
        self.services_config = zone_config.get("services", {})
        self.check_interval = check_interval
        self.base_dir = base_dir

        self.checker = ServiceChecker(base_dir)
        self.start_time = datetime.now()

        # Current state (protected by lock)
        self._lock = threading.Lock()
        self._services: Dict[str, ServiceStatus] = {}
        self._last_patrol: Optional[datetime] = None

        # Background patrol thread
        self._stop_event = threading.Event()
        self._patrol_thread: Optional[threading.Thread] = None

    def start_patrol(self):
        """Start background health checks."""
        if self._patrol_thread and self._patrol_thread.is_alive():
            return

        self._stop_event.clear()
        self._patrol_thread = threading.Thread(target=self._patrol_loop, daemon=True)
        self._patrol_thread.start()
        logger.info(f"Warden patrol started (every {self.check_interval}s)")

        # Do initial patrol immediately
        self._do_patrol()

    def stop_patrol(self):
        """Stop background health checks."""
        self._stop_event.set()
        if self._patrol_thread:
            self._patrol_thread.join(timeout=5)
        logger.info("Warden patrol stopped")

    def _patrol_loop(self):
        """Background patrol loop."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.check_interval)
            if not self._stop_event.is_set():
                self._do_patrol()

    def _do_patrol(self):
        """Perform one patrol (check all services)."""
        new_statuses = {}

        for service_id, config in self.services_config.items():
            try:
                status = self.checker.check_service(service_id, config)
                new_statuses[service_id] = status
            except Exception as e:
                logger.error(f"Error checking {service_id}: {e}")
                new_statuses[service_id] = ServiceStatus(
                    service_id=service_id,
                    name=config.get("name", service_id),
                    status="unknown",
                    critical=config.get("critical", False),
                    error=str(e),
                    last_check=datetime.now().isoformat()
                )

        with self._lock:
            self._services = new_statuses
            self._last_patrol = datetime.now()

        # Log summary
        online = sum(1 for s in new_statuses.values() if s.status == "online")
        total = len(new_statuses)
        logger.debug(f"Patrol complete: {online}/{total} services online")

    def get_zone_health(self) -> ZoneHealth:
        """Get current zone health."""
        with self._lock:
            services = dict(self._services)
            last_patrol = self._last_patrol

        # Calculate summary
        total = len(services)
        online = sum(1 for s in services.values() if s.status == "online")
        offline = sum(1 for s in services.values() if s.status == "offline")
        degraded = sum(1 for s in services.values() if s.status == "degraded")
        unknown = sum(1 for s in services.values() if s.status == "unknown")

        # Calculate zone status
        critical_down = any(
            s.status in ("offline", "unknown") and s.critical
            for s in services.values()
        )

        if total == 0:
            zone_status = "online"  # No services to check
        elif critical_down:
            zone_status = "offline"
        elif offline > 0 or degraded > 0:
            zone_status = "degraded"
        else:
            zone_status = "online"

        uptime = int((datetime.now() - self.start_time).total_seconds())

        return ZoneHealth(
            zone_id=self.zone_id,
            zone_name=self.zone_name,
            zone_role=self.zone_role,
            status=zone_status,
            host=self.zone_host,
            warden_version=ZONE_WARDEN_VERSION,
            services=services,
            summary={
                "total": total,
                "online": online,
                "offline": offline,
                "degraded": degraded,
                "unknown": unknown,
            },
            last_patrol=last_patrol.isoformat() if last_patrol else None,
            uptime_seconds=uptime
        )

    def get_service(self, service_id: str) -> Optional[ServiceStatus]:
        """Get status of a specific service."""
        with self._lock:
            return self._services.get(service_id)


# =============================================================================
# HTTP SERVER
# =============================================================================

class WardenHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Warden API."""

    warden: ZoneWarden = None  # Set by server

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json({"error": message}, status)

    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split("?")[0]

        if path == "/" or path == "/health":
            self._handle_health()
        elif path == "/services":
            self._handle_services()
        elif path.startswith("/service/"):
            service_id = path[9:]  # Remove "/service/"
            self._handle_service_detail(service_id)
        elif path == "/version":
            self._send_json({
                "warden_version": ZONE_WARDEN_VERSION,
                "zone_id": self.warden.zone_id,
            })
        else:
            self._send_error("Not found", 404)

    def _handle_health(self):
        """Return zone health summary."""
        health = self.warden.get_zone_health()

        # Simple health response for /health
        response = {
            "zone_id": health.zone_id,
            "zone_name": health.zone_name,
            "status": health.status,
            "host": health.host,
            "role": health.zone_role,
            "summary": health.summary,
            "warden_version": health.warden_version,
            "uptime_seconds": health.uptime_seconds,
            "last_patrol": health.last_patrol,
        }
        self._send_json(response)

    def _handle_services(self):
        """Return detailed service status."""
        health = self.warden.get_zone_health()

        # Convert ServiceStatus objects to dicts
        services_dict = {}
        for sid, status in health.services.items():
            services_dict[sid] = asdict(status)

        response = {
            "zone_id": health.zone_id,
            "zone_name": health.zone_name,
            "status": health.status,
            "services": services_dict,
            "summary": health.summary,
            "last_patrol": health.last_patrol,
        }
        self._send_json(response)

    def _handle_service_detail(self, service_id: str):
        """Return status of a specific service."""
        status = self.warden.get_service(service_id)

        if status is None:
            self._send_error(f"Unknown service: {service_id}", 404)
            return

        self._send_json(asdict(status))


class WardenServer:
    """HTTP server wrapper for the Warden."""

    def __init__(self, warden: ZoneWarden, port: int):
        self.warden = warden
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None

    def start(self):
        """Start the HTTP server."""
        WardenHandler.warden = self.warden

        # Use ThreadingHTTPServer to handle concurrent requests without blocking
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), WardenHandler)
        self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        logger.info(f"Warden server listening on port {self.port}")
        self.server.serve_forever()

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zone Warden - Service health aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 zone_warden.py --zone 4090
  python3 zone_warden.py --zone 3090 --port 8760
  python3 zone_warden.py --zone nas --interval 60

The Warden monitors all services defined for this zone in config/hosts.json
and exposes aggregate health at http://localhost:PORT/health
        """
    )
    parser.add_argument(
        "--zone", "-z",
        required=True,
        help="Zone ID (4090, 3090, nas)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
        help=f"Health check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})"
    )
    parser.add_argument(
        "--base-dir", "-d",
        type=str,
        default=None,
        help="Base directory for this zone (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load zone config
    try:
        zone_config = load_zone_config(args.zone)
    except Exception as e:
        logger.error(f"Failed to load zone config: {e}")
        sys.exit(1)

    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Auto-detect based on zone
        if args.zone == "4090":
            base_dir = Path("/path/to/training")
        elif args.zone == "3090":
            base_dir = Path("/home/user/llm")
        else:
            base_dir = Path.cwd()

    # Use configured port if not overridden
    port = args.port if args.port != DEFAULT_PORT else zone_config.get("warden_port", DEFAULT_PORT)

    logger.info(f"Starting Zone Warden v{ZONE_WARDEN_VERSION}")
    logger.info(f"  Zone: {args.zone} ({zone_config.get('name', 'Unknown')})")
    logger.info(f"  Port: {port}")
    logger.info(f"  Base: {base_dir}")
    logger.info(f"  Services: {len(zone_config.get('services', {}))}")

    # Create warden
    warden = ZoneWarden(zone_config, base_dir, check_interval=args.interval)

    # Create server
    server = WardenServer(warden, port)

    # Handle shutdown signals
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        warden.stop_patrol()
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start patrol and server
    warden.start_patrol()

    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        warden.stop_patrol()


if __name__ == "__main__":
    main()
