"""
Cluster State - Host Registry for the Realm

This module manages the cluster of hosts that can participate in training,
inference, and other distributed tasks. It's the single source of truth for:

- What hosts exist
- What roles each host can perform (trainer, oracle, forge, monitor)
- Health status and heartbeats
- Resource availability (GPU, CPU, RAM, disk)
- Which realm each host belongs to

Usage:
    from core.cluster_state import get_cluster_state, register_host, heartbeat

    # Register this host on startup
    register_host(
        host_id="arena-4090",
        name="Training Arena",
        roles=["trainer"],
        realm_id="main",
    )

    # Send periodic heartbeats
    heartbeat("arena-4090")

    # Query cluster
    cluster = get_cluster_state()
    online_hosts = [h for h in cluster.hosts.values() if h.status.status == "online"]

The cluster state is persisted to status/cluster_state.json.
"""

import json
import logging
import os
import socket
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

logger = logging.getLogger(__name__)

# Type aliases
Role = Literal["trainer", "oracle", "forge", "monitor", "mixed"]
HostStatusType = Literal["online", "degraded", "offline", "unknown"]

# Default heartbeat timeout (seconds)
HEARTBEAT_TIMEOUT = 120.0
HEARTBEAT_STALE = 60.0


@dataclass
class HostResources:
    """Resource snapshot for a host."""
    gpu_count: int = 0
    gpu_total_vram_gb: float = 0.0
    gpu_free_vram_gb: float = 0.0
    gpu_utilization: int = 0  # percent
    cpu_cores: int = 0
    cpu_load: float = 0.0  # percent
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    disk_free_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostResources":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HostStatus:
    """Status information for a host."""
    status: HostStatusType = "unknown"
    last_heartbeat: Optional[str] = None  # ISO timestamp
    last_error: Optional[str] = None
    uptime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostStatus":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def seconds_since_heartbeat(self) -> Optional[float]:
        """Seconds since last heartbeat."""
        if not self.last_heartbeat:
            return None
        try:
            ts = datetime.fromisoformat(self.last_heartbeat)
            return (datetime.now() - ts).total_seconds()
        except Exception:
            return None


@dataclass
class HostInfo:
    """Complete information about a host in the cluster."""
    host_id: str
    name: str
    roles: List[Role]
    realm_id: Optional[str] = None
    resources: HostResources = field(default_factory=HostResources)
    status: HostStatus = field(default_factory=HostStatus)
    tags: List[str] = field(default_factory=list)
    # Job tracking
    running_jobs: int = 0
    queued_jobs: int = 0
    # Metadata
    registered_at: Optional[str] = None
    ip_address: Optional[str] = None
    hostname: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "name": self.name,
            "roles": self.roles,
            "realm_id": self.realm_id,
            "resources": self.resources.to_dict(),
            "status": self.status.to_dict(),
            "tags": self.tags,
            "running_jobs": self.running_jobs,
            "queued_jobs": self.queued_jobs,
            "registered_at": self.registered_at,
            "ip_address": self.ip_address,
            "hostname": self.hostname,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostInfo":
        resources = HostResources.from_dict(data.get("resources", {}))
        status = HostStatus.from_dict(data.get("status", {}))
        return cls(
            host_id=data["host_id"],
            name=data.get("name", data["host_id"]),
            roles=data.get("roles", []),
            realm_id=data.get("realm_id"),
            resources=resources,
            status=status,
            tags=data.get("tags", []),
            running_jobs=data.get("running_jobs", 0),
            queued_jobs=data.get("queued_jobs", 0),
            registered_at=data.get("registered_at"),
            ip_address=data.get("ip_address"),
            hostname=data.get("hostname"),
        )


@dataclass
class ClusterState:
    """Complete cluster state."""
    hosts: Dict[str, HostInfo] = field(default_factory=dict)
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hosts": {k: v.to_dict() for k, v in self.hosts.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterState":
        hosts = {}
        for host_id, host_data in data.get("hosts", {}).items():
            try:
                hosts[host_id] = HostInfo.from_dict(host_data)
            except Exception as e:
                logger.warning(f"Failed to load host {host_id}: {e}")
        return cls(hosts=hosts, updated_at=data.get("updated_at"))


# =============================================================================
# SINGLETON CLUSTER STATE MANAGER
# =============================================================================

class ClusterStateManager:
    """
    Manages the cluster state with thread-safe operations and persistence.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self._state = ClusterState()
        self._lock = threading.RLock()
        self._state_file = state_file
        self._loaded = False

    def _get_state_file(self) -> Path:
        """Get the state file path."""
        if self._state_file:
            return self._state_file
        try:
            from core.paths import get_base_dir
            return get_base_dir() / "status" / "cluster_state.json"
        except Exception:
            return Path("status/cluster_state.json")

    def _ensure_loaded(self):
        """Load state from disk if not already loaded."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        """Load state from disk."""
        state_file = self._get_state_file()
        try:
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                self._state = ClusterState.from_dict(data)
                logger.debug(f"Loaded cluster state with {len(self._state.hosts)} hosts")
        except Exception as e:
            logger.warning(f"Failed to load cluster state: {e}")
            self._state = ClusterState()

    def _save(self):
        """Save state to disk."""
        self._state.updated_at = datetime.now().isoformat()
        state_file = self._get_state_file()
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
            tmp.replace(state_file)
        except Exception as e:
            logger.error(f"Failed to save cluster state: {e}")

    def reload(self):
        """Force reload from disk."""
        with self._lock:
            self._loaded = False
            self._ensure_loaded()

    # =========================================================================
    # HOST MANAGEMENT
    # =========================================================================

    def register_host(
        self,
        host_id: str,
        name: Optional[str] = None,
        roles: Optional[List[Role]] = None,
        realm_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
    ) -> HostInfo:
        """
        Register a new host or update an existing one.

        This should be called on host startup.
        """
        with self._lock:
            self._ensure_loaded()

            now = datetime.now().isoformat()

            # Get or create host
            if host_id in self._state.hosts:
                host = self._state.hosts[host_id]
                # Update fields if provided
                if name:
                    host.name = name
                if roles is not None:
                    host.roles = roles
                if realm_id is not None:
                    host.realm_id = realm_id
                if tags is not None:
                    host.tags = tags
                if ip_address:
                    host.ip_address = ip_address
            else:
                # Create new host
                host = HostInfo(
                    host_id=host_id,
                    name=name or host_id,
                    roles=roles or [],
                    realm_id=realm_id,
                    tags=tags or [],
                    registered_at=now,
                    ip_address=ip_address,
                    hostname=socket.gethostname(),
                )
                self._state.hosts[host_id] = host

            # Update status
            host.status.status = "online"
            host.status.last_heartbeat = now

            self._save()
            logger.info(f"Registered host: {host_id} ({name or host_id})")
            return host

    def heartbeat(self, host_id: str, extra: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a heartbeat from a host.

        Returns True if the host exists, False otherwise.
        """
        with self._lock:
            self._ensure_loaded()

            if host_id not in self._state.hosts:
                logger.warning(f"Heartbeat from unknown host: {host_id}")
                return False

            host = self._state.hosts[host_id]
            host.status.last_heartbeat = datetime.now().isoformat()
            host.status.status = "online"
            host.status.last_error = None

            # Update extra fields if provided
            if extra:
                if "running_jobs" in extra:
                    host.running_jobs = extra["running_jobs"]
                if "queued_jobs" in extra:
                    host.queued_jobs = extra["queued_jobs"]

            self._save()
            return True

    def update_host_resources(
        self,
        host_id: str,
        resources: Dict[str, Any],
    ) -> bool:
        """
        Update resource metrics for a host.
        """
        with self._lock:
            self._ensure_loaded()

            if host_id not in self._state.hosts:
                logger.warning(f"Resource update for unknown host: {host_id}")
                return False

            host = self._state.hosts[host_id]
            host.resources = HostResources.from_dict(resources)
            self._save()
            return True

    def set_host_status(
        self,
        host_id: str,
        status: HostStatusType,
        error: Optional[str] = None,
    ) -> bool:
        """
        Set the status of a host.
        """
        with self._lock:
            self._ensure_loaded()

            if host_id not in self._state.hosts:
                logger.warning(f"Status update for unknown host: {host_id}")
                return False

            host = self._state.hosts[host_id]
            host.status.status = status
            host.status.last_error = error
            self._save()
            return True

    def remove_host(self, host_id: str) -> bool:
        """
        Remove a host from the cluster.
        """
        with self._lock:
            self._ensure_loaded()

            if host_id not in self._state.hosts:
                return False

            del self._state.hosts[host_id]
            self._save()
            logger.info(f"Removed host: {host_id}")
            return True

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_cluster_state(self) -> ClusterState:
        """Get a copy of the cluster state."""
        with self._lock:
            self._ensure_loaded()
            return self._state

    def get_host(self, host_id: str) -> Optional[HostInfo]:
        """Get a specific host."""
        with self._lock:
            self._ensure_loaded()
            return self._state.hosts.get(host_id)

    def get_hosts_by_role(self, role: Role) -> List[HostInfo]:
        """Get all hosts with a specific role."""
        with self._lock:
            self._ensure_loaded()
            return [h for h in self._state.hosts.values() if role in h.roles]

    def get_hosts_by_realm(self, realm_id: str) -> List[HostInfo]:
        """Get all hosts in a specific realm."""
        with self._lock:
            self._ensure_loaded()
            return [h for h in self._state.hosts.values() if h.realm_id == realm_id]

    def get_online_hosts(self) -> List[HostInfo]:
        """Get all online hosts."""
        with self._lock:
            self._ensure_loaded()
            return [h for h in self._state.hosts.values() if h.status.status == "online"]

    def get_hosts_for_job(self, role: Role) -> List[HostInfo]:
        """Get hosts eligible for a job type (online + has role)."""
        with self._lock:
            self._ensure_loaded()
            return [
                h for h in self._state.hosts.values()
                if role in h.roles and h.status.status == "online"
            ]

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def check_heartbeats(self) -> List[str]:
        """
        Check for stale heartbeats and mark hosts as offline.

        Returns list of host_ids that were marked offline.
        """
        marked_offline = []
        with self._lock:
            self._ensure_loaded()
            now = datetime.now()

            for host_id, host in self._state.hosts.items():
                if host.status.status == "offline":
                    continue

                age = host.status.seconds_since_heartbeat
                if age is None:
                    continue

                if age > HEARTBEAT_TIMEOUT:
                    host.status.status = "offline"
                    host.status.last_error = f"No heartbeat for {age:.0f}s"
                    marked_offline.append(host_id)
                    logger.warning(f"Host {host_id} marked offline (no heartbeat for {age:.0f}s)")
                elif age > HEARTBEAT_STALE:
                    host.status.status = "degraded"

            if marked_offline:
                self._save()

        return marked_offline

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the cluster."""
        with self._lock:
            self._ensure_loaded()

            hosts = list(self._state.hosts.values())
            online = [h for h in hosts if h.status.status == "online"]
            degraded = [h for h in hosts if h.status.status == "degraded"]
            offline = [h for h in hosts if h.status.status == "offline"]

            roles_count = {}
            for host in online:
                for role in host.roles:
                    roles_count[role] = roles_count.get(role, 0) + 1

            total_running_jobs = sum(h.running_jobs for h in hosts)
            total_queued_jobs = sum(h.queued_jobs for h in hosts)

            return {
                "total_hosts": len(hosts),
                "online": len(online),
                "degraded": len(degraded),
                "offline": len(offline),
                "roles": roles_count,
                "total_running_jobs": total_running_jobs,
                "total_queued_jobs": total_queued_jobs,
                "updated_at": self._state.updated_at,
            }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_manager: Optional[ClusterStateManager] = None
_manager_lock = threading.Lock()


def _get_manager() -> ClusterStateManager:
    """Get the singleton cluster state manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ClusterStateManager()
    return _manager


# =============================================================================
# PUBLIC API
# =============================================================================

def get_cluster_state() -> ClusterState:
    """Get the current cluster state."""
    return _get_manager().get_cluster_state()


def register_host(
    host_id: str,
    name: Optional[str] = None,
    roles: Optional[List[Role]] = None,
    realm_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    ip_address: Optional[str] = None,
) -> HostInfo:
    """Register a host in the cluster."""
    return _get_manager().register_host(
        host_id=host_id,
        name=name,
        roles=roles,
        realm_id=realm_id,
        tags=tags,
        ip_address=ip_address,
    )


def heartbeat(host_id: str, extra: Optional[Dict[str, Any]] = None) -> bool:
    """Send a heartbeat from a host."""
    return _get_manager().heartbeat(host_id, extra)


def update_host_resources(host_id: str, resources: Dict[str, Any]) -> bool:
    """Update resource metrics for a host."""
    return _get_manager().update_host_resources(host_id, resources)


def set_host_status(host_id: str, status: HostStatusType, error: Optional[str] = None) -> bool:
    """Set the status of a host."""
    return _get_manager().set_host_status(host_id, status, error)


def remove_host(host_id: str) -> bool:
    """Remove a host from the cluster."""
    return _get_manager().remove_host(host_id)


def get_host(host_id: str) -> Optional[HostInfo]:
    """Get a specific host."""
    return _get_manager().get_host(host_id)


def get_hosts_by_role(role: Role) -> List[HostInfo]:
    """Get all hosts with a specific role."""
    return _get_manager().get_hosts_by_role(role)


def get_hosts_by_realm(realm_id: str) -> List[HostInfo]:
    """Get all hosts in a specific realm."""
    return _get_manager().get_hosts_by_realm(realm_id)


def get_online_hosts() -> List[HostInfo]:
    """Get all online hosts."""
    return _get_manager().get_online_hosts()


def get_hosts_for_job(role: Role) -> List[HostInfo]:
    """Get hosts eligible for a job type."""
    return _get_manager().get_hosts_for_job(role)


def check_heartbeats() -> List[str]:
    """Check for stale heartbeats."""
    return _get_manager().check_heartbeats()


def get_cluster_summary() -> Dict[str, Any]:
    """Get cluster summary."""
    return _get_manager().get_summary()


def reload():
    """Force reload cluster state from disk."""
    _get_manager().reload()


# =============================================================================
# LOCAL HOST HELPERS
# =============================================================================

def get_local_host_id() -> str:
    """
    Get the host ID for this machine.

    Uses CLUSTER_HOST_ID env var if set, otherwise uses hostname.
    """
    return os.environ.get("CLUSTER_HOST_ID", socket.gethostname())


def probe_local_resources() -> Dict[str, Any]:
    """
    Probe local machine resources (GPU, CPU, RAM, disk).

    Returns a dict suitable for update_host_resources().
    """
    resources = {
        "gpu_count": 0,
        "gpu_total_vram_gb": 0.0,
        "gpu_free_vram_gb": 0.0,
        "gpu_utilization": 0,
        "cpu_cores": os.cpu_count() or 0,
        "cpu_load": 0.0,
        "ram_total_gb": 0.0,
        "ram_used_gb": 0.0,
        "disk_free_gb": 0.0,
    }

    # GPU stats
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            total_vram = 0.0
            free_vram = 0.0
            total_util = 0
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    resources["gpu_count"] += 1
                    total_vram += float(parts[0]) / 1024  # MB to GB
                    free_vram += float(parts[1]) / 1024
                    total_util += int(parts[2]) if parts[2].isdigit() else 0
            resources["gpu_total_vram_gb"] = round(total_vram, 2)
            resources["gpu_free_vram_gb"] = round(free_vram, 2)
            if resources["gpu_count"] > 0:
                resources["gpu_utilization"] = total_util // resources["gpu_count"]
    except Exception as e:
        logger.debug(f"Failed to probe GPU: {e}")

    # CPU load
    try:
        load = os.getloadavg()[0]  # 1-minute average
        cores = resources["cpu_cores"] or 1
        resources["cpu_load"] = round(load / cores * 100, 1)
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                resources["ram_total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 2)
            elif line.startswith("MemAvailable:"):
                available_gb = int(line.split()[1]) / 1024 / 1024
                resources["ram_used_gb"] = round(resources["ram_total_gb"] - available_gb, 2)
    except Exception:
        pass

    # Disk
    try:
        from core.paths import get_base_dir
        base = get_base_dir()
        stat = os.statvfs(base)
        resources["disk_free_gb"] = round(stat.f_bavail * stat.f_frsize / 1024 / 1024 / 1024, 2)
    except Exception:
        pass

    return resources


def register_local_host(
    roles: Optional[List[Role]] = None,
    realm_id: Optional[str] = None,
    name: Optional[str] = None,
) -> HostInfo:
    """
    Register this local host in the cluster.

    Convenience function that probes resources and registers.
    """
    host_id = get_local_host_id()

    # Try to get IP address
    ip_address = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    # Register
    host = register_host(
        host_id=host_id,
        name=name or host_id,
        roles=roles or [],
        realm_id=realm_id,
        ip_address=ip_address,
    )

    # Update resources
    resources = probe_local_resources()
    update_host_resources(host_id, resources)

    return host


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Cluster State Manager")
    parser.add_argument("command", choices=["show", "register", "heartbeat", "check", "summary"],
                        help="Command to run")
    parser.add_argument("--host-id", help="Host ID (default: local hostname)")
    parser.add_argument("--name", help="Host display name")
    parser.add_argument("--roles", help="Comma-separated roles (trainer,oracle,forge,monitor)")
    parser.add_argument("--realm", help="Realm ID")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "show":
        cluster = get_cluster_state()
        if args.json:
            print(json.dumps(cluster.to_dict(), indent=2))
        else:
            print(f"Cluster State ({len(cluster.hosts)} hosts)")
            print("=" * 60)
            for host_id, host in cluster.hosts.items():
                status_icon = {
                    "online": "🟢",
                    "degraded": "🟡",
                    "offline": "🔴",
                    "unknown": "⚪",
                }.get(host.status.status, "⚪")
                roles_str = ", ".join(host.roles) if host.roles else "none"
                print(f"\n{status_icon} {host.name} ({host_id})")
                print(f"   Roles: {roles_str}")
                print(f"   Realm: {host.realm_id or 'none'}")
                if host.resources.gpu_count > 0:
                    print(f"   GPU: {host.resources.gpu_count}x ({host.resources.gpu_free_vram_gb:.1f}/{host.resources.gpu_total_vram_gb:.1f} GB free)")
                print(f"   Jobs: {host.running_jobs} running, {host.queued_jobs} queued")
                if host.status.last_heartbeat:
                    age = host.status.seconds_since_heartbeat
                    print(f"   Last heartbeat: {age:.0f}s ago" if age else "   Last heartbeat: unknown")

    elif args.command == "register":
        host_id = args.host_id or get_local_host_id()
        roles = args.roles.split(",") if args.roles else None
        host = register_local_host(
            roles=roles,
            realm_id=args.realm,
            name=args.name,
        )
        print(f"Registered: {host.host_id} ({host.name})")

    elif args.command == "heartbeat":
        host_id = args.host_id or get_local_host_id()
        if heartbeat(host_id):
            print(f"Heartbeat sent for {host_id}")
        else:
            print(f"Host {host_id} not registered")

    elif args.command == "check":
        offline = check_heartbeats()
        if offline:
            print(f"Marked offline: {', '.join(offline)}")
        else:
            print("All hosts OK")

    elif args.command == "summary":
        summary = get_cluster_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Cluster Summary")
            print(f"  Total hosts: {summary['total_hosts']}")
            print(f"  Online: {summary['online']}")
            print(f"  Degraded: {summary['degraded']}")
            print(f"  Offline: {summary['offline']}")
            print(f"  Roles: {summary['roles']}")
            print(f"  Running jobs: {summary['total_running_jobs']}")
            print(f"  Queued jobs: {summary['total_queued_jobs']}")
