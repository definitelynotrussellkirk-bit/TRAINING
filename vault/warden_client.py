"""
Warden Client - Query zone wardens for health status.

Provides a simple interface to query Zone Wardens across the Realm.
Used by the Tavern and other services to check zone health.

Usage:
    from vault.warden_client import WardenClient, get_all_zone_health

    # Query a specific warden
    client = WardenClient("trainer.local", 8760)
    health = client.get_health()
    print(f"Zone {health['zone_id']}: {health['status']}")

    # Query all zones
    all_health = get_all_zone_health()
    for zone_id, health in all_health.items():
        print(f"{zone_id}: {health['status']}")
"""

import json
import logging
import socket
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8760
DEFAULT_TIMEOUT = 5


# =============================================================================
# CONFIGURATION
# =============================================================================

def find_config_file() -> Path:
    """Find hosts.json config file."""
    from core.paths import get_base_dir

    base_dir = get_base_dir()
    here = Path(__file__).parent.parent
    candidates = [
        here / "config" / "hosts.json",
        base_dir / "config" / "hosts.json",
        Path.cwd() / "config" / "hosts.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find hosts.json in {candidates}")


def load_hosts_config() -> Dict[str, Any]:
    """Load the full hosts configuration."""
    config_path = find_config_file()
    with open(config_path) as f:
        return json.load(f)


def get_zone_endpoints() -> Dict[str, Dict[str, Any]]:
    """Get warden endpoints for all zones."""
    config = load_hosts_config()
    warden_port = config.get("warden_port", DEFAULT_PORT)

    endpoints = {}
    for zone_id, zone_config in config.get("hosts", {}).items():
        endpoints[zone_id] = {
            "host": zone_config.get("host", "localhost"),
            "port": warden_port,
            "name": zone_config.get("name", zone_id),
            "role": zone_config.get("role", "unknown"),
        }
    return endpoints


# =============================================================================
# WARDEN CLIENT
# =============================================================================

@dataclass
class ZoneHealthResult:
    """Result of querying a zone's warden."""
    zone_id: str
    reachable: bool
    status: str  # online, degraded, offline, unreachable
    zone_name: Optional[str] = None
    zone_role: Optional[str] = None
    host: Optional[str] = None
    services: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    response_ms: Optional[int] = None
    warden_version: Optional[str] = None
    last_patrol: Optional[str] = None


class WardenClient:
    """Client for querying a single Zone Warden."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout: int = DEFAULT_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def _get(self, path: str) -> Dict[str, Any]:
        """Make GET request to warden."""
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def get_health(self) -> Dict[str, Any]:
        """Get zone health summary."""
        return self._get("/health")

    def get_services(self) -> Dict[str, Any]:
        """Get detailed service status."""
        return self._get("/services")

    def get_service(self, service_id: str) -> Dict[str, Any]:
        """Get status of a specific service."""
        return self._get(f"/service/{service_id}")

    def is_available(self) -> bool:
        """Check if warden is reachable."""
        try:
            self.get_health()
            return True
        except:
            return False


def query_zone_warden(zone_id: str, host: str, port: int = DEFAULT_PORT,
                      timeout: int = DEFAULT_TIMEOUT) -> ZoneHealthResult:
    """
    Query a zone's warden and return structured result.

    Args:
        zone_id: The zone identifier (4090, 3090, nas)
        host: The warden's host address
        port: The warden's port (default: 8760)
        timeout: Request timeout in seconds

    Returns:
        ZoneHealthResult with status information
    """
    import time
    start = time.time()

    try:
        client = WardenClient(host, port, timeout)
        data = client.get_services()  # Get full details
        elapsed_ms = int((time.time() - start) * 1000)

        return ZoneHealthResult(
            zone_id=zone_id,
            reachable=True,
            status=data.get("status", "unknown"),
            zone_name=data.get("zone_name"),
            zone_role=data.get("zone_role"),
            host=host,
            services=data.get("services", {}),
            summary=data.get("summary", {}),
            response_ms=elapsed_ms,
            warden_version=data.get("warden_version"),
            last_patrol=data.get("last_patrol"),
        )

    except urllib.error.URLError as e:
        return ZoneHealthResult(
            zone_id=zone_id,
            reachable=False,
            status="unreachable",
            host=host,
            error=f"Connection failed: {e.reason}",
        )
    except socket.timeout:
        return ZoneHealthResult(
            zone_id=zone_id,
            reachable=False,
            status="unreachable",
            host=host,
            error="Connection timeout",
        )
    except Exception as e:
        return ZoneHealthResult(
            zone_id=zone_id,
            reachable=False,
            status="unreachable",
            host=host,
            error=str(e),
        )


def get_all_zone_health(timeout: int = DEFAULT_TIMEOUT,
                        parallel: bool = True) -> Dict[str, ZoneHealthResult]:
    """
    Query all zone wardens and return health status.

    Args:
        timeout: Request timeout per zone
        parallel: If True, query zones in parallel

    Returns:
        Dict mapping zone_id to ZoneHealthResult
    """
    endpoints = get_zone_endpoints()
    results = {}

    if parallel and len(endpoints) > 1:
        # Query all zones in parallel
        with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = {
                executor.submit(
                    query_zone_warden,
                    zone_id,
                    info["host"],
                    info["port"],
                    timeout
                ): zone_id
                for zone_id, info in endpoints.items()
            }

            for future in as_completed(futures):
                zone_id = futures[future]
                try:
                    results[zone_id] = future.result()
                except Exception as e:
                    results[zone_id] = ZoneHealthResult(
                        zone_id=zone_id,
                        reachable=False,
                        status="unreachable",
                        error=str(e)
                    )
    else:
        # Query sequentially
        for zone_id, info in endpoints.items():
            results[zone_id] = query_zone_warden(
                zone_id, info["host"], info["port"], timeout
            )

    return results


def get_zone_summary() -> List[Dict[str, Any]]:
    """
    Get a summary of all zones suitable for UI display.

    Returns list of zone info dicts with status.
    """
    endpoints = get_zone_endpoints()
    health_results = get_all_zone_health()

    zones = []
    for zone_id, endpoint in endpoints.items():
        health = health_results.get(zone_id)

        zone_info = {
            "zone_id": zone_id,
            "name": endpoint["name"],
            "role": endpoint["role"],
            "host": endpoint["host"],
            "port": endpoint["port"],
            "status": health.status if health else "unknown",
            "reachable": health.reachable if health else False,
        }

        if health and health.reachable:
            zone_info.update({
                "services": health.summary or {},
                "response_ms": health.response_ms,
                "warden_version": health.warden_version,
                "last_patrol": health.last_patrol,
            })
        elif health:
            zone_info["error"] = health.error

        zones.append(zone_info)

    return zones


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_zone_online(zone_id: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Check if a zone is online (quick check)."""
    endpoints = get_zone_endpoints()
    if zone_id not in endpoints:
        return False

    info = endpoints[zone_id]
    result = query_zone_warden(zone_id, info["host"], info["port"], timeout)
    return result.status == "online"


def get_zone_services(zone_id: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Get detailed service status for a zone."""
    endpoints = get_zone_endpoints()
    if zone_id not in endpoints:
        return None

    info = endpoints[zone_id]
    try:
        client = WardenClient(info["host"], info["port"], timeout)
        return client.get_services()
    except:
        return None


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for testing warden client."""
    import argparse

    parser = argparse.ArgumentParser(description="Query Zone Wardens")
    parser.add_argument("--zone", "-z", help="Query specific zone")
    parser.add_argument("--all", "-a", action="store_true", help="Query all zones")
    parser.add_argument("--services", "-s", action="store_true", help="Show service details")
    parser.add_argument("--timeout", "-t", type=int, default=5, help="Timeout in seconds")

    args = parser.parse_args()

    if args.zone:
        endpoints = get_zone_endpoints()
        if args.zone not in endpoints:
            print(f"Unknown zone: {args.zone}")
            print(f"Available: {list(endpoints.keys())}")
            return

        info = endpoints[args.zone]
        print(f"Querying {args.zone} at {info['host']}:{info['port']}...")

        if args.services:
            result = get_zone_services(args.zone, args.timeout)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("Failed to get services")
        else:
            result = query_zone_warden(args.zone, info["host"], info["port"], args.timeout)
            print(f"Zone: {result.zone_id}")
            print(f"Status: {result.status}")
            print(f"Reachable: {result.reachable}")
            if result.error:
                print(f"Error: {result.error}")
            if result.summary:
                print(f"Services: {result.summary}")

    elif args.all:
        print("Querying all zones...")
        zones = get_zone_summary()
        for zone in zones:
            status_icon = {
                "online": "✅",
                "degraded": "⚠️",
                "offline": "❌",
                "unreachable": "🔌"
            }.get(zone["status"], "❓")

            print(f"{status_icon} {zone['zone_id']:6} ({zone['name']:20}) - {zone['status']}")
            if "error" in zone:
                print(f"   Error: {zone['error']}")
            elif "services" in zone:
                s = zone["services"]
                print(f"   Services: {s.get('online', 0)}/{s.get('total', 0)} online")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
