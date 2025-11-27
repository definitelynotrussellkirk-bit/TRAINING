"""
Host Registry - Service discovery for distributed training system.

This module provides location-independent access to services across hosts.
Instead of hardcoding IPs, components query the registry for service URLs.

Usage:
    from core.hosts import get_service_url, get_host, is_local

    # Get URL for a service
    ledger_url = get_service_url("ledger")  # "http://192.168.x.x:8767/api/ledger"
    inference_url = get_service_url("inference")  # "http://192.168.x.x:8765"

    # Check if we're on a specific host
    if is_local("trainer"):
        # Use local file access
        ledger = CheckpointLedger()
    else:
        # Use remote API
        ledger = RemoteLedgerClient(get_service_url("ledger"))

    # Get host config
    host = get_host("3090")
    print(host.name)  # "Inference Server"

Design:
    - config/hosts.json is the single source of truth
    - Each host has a role and list of services
    - Services have ports and optional paths
    - Auto-detection of "am I on this host?"
"""

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger("hosts")


class HostRole(Enum):
    """Role of a host in the system."""
    TRAINER = "trainer"      # Runs training, owns ledger
    INFERENCE = "inference"  # Runs inference server
    STORAGE = "storage"      # Archive/backup storage
    CONTROLLER = "controller"  # UI/orchestration (can be any)
    HYBRID = "hybrid"        # Multiple roles


class ServiceType(Enum):
    """Types of services that can run on hosts."""
    VAULT = "vault"          # VaultKeeper API
    LEDGER = "ledger"        # Checkpoint ledger API
    TRAINING = "training"    # Training status/control API
    INFERENCE = "inference"  # Inference/chat API
    BRANCH = "branch"        # Branch Officer API
    MONITOR = "monitor"      # Monitoring API


@dataclass
class ServiceConfig:
    """Configuration for a service on a host."""
    port: int
    path: str = ""
    protocol: str = "http"
    auth_required: bool = False

    def get_url(self, host: str) -> str:
        """Build full URL for this service."""
        base = f"{self.protocol}://{host}:{self.port}"
        if self.path:
            return f"{base}{self.path}"
        return base


@dataclass
class HostConfig:
    """Configuration for a host."""
    host_id: str
    name: str
    host: str  # IP or hostname
    role: HostRole
    services: Dict[str, ServiceConfig] = field(default_factory=dict)
    ssh_user: str = "user"
    models_dir: str = ""
    checkpoints_dir: str = ""
    capabilities: List[str] = field(default_factory=list)

    def get_service_url(self, service: str) -> Optional[str]:
        """Get URL for a service on this host."""
        if service in self.services:
            return self.services[service].get_url(self.host)
        return None

    def has_service(self, service: str) -> bool:
        """Check if host provides a service."""
        return service in self.services

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host_id": self.host_id,
            "name": self.name,
            "host": self.host,
            "role": self.role.value,
            "services": {
                k: {"port": v.port, "path": v.path, "protocol": v.protocol}
                for k, v in self.services.items()
            },
            "ssh_user": self.ssh_user,
            "models_dir": self.models_dir,
            "checkpoints_dir": self.checkpoints_dir,
            "capabilities": self.capabilities,
        }


class HostRegistry:
    """
    Central registry of all hosts and services.

    Loads from config/hosts.json and provides service discovery.
    """

    # Default configuration (used if hosts.json doesn't exist)
    DEFAULTS = {
        "4090": {
            "name": "Training Server",
            "host": "192.168.x.x",
            "role": "trainer",
            "services": {
                "vault": {"port": 8767, "path": "/api"},
                "ledger": {"port": 8767, "path": "/api/ledger"},
                "training": {"port": 8767, "path": "/api/training"},
                "branch": {"port": 8768},
                "monitor": {"port": 8081},
            },
            "models_dir": "/path/to/training/models",
            "checkpoints_dir": "/path/to/training/current_model",
            "capabilities": ["training", "vault", "ledger"],
        },
        "3090": {
            "name": "Inference Server",
            "host": "192.168.x.x",
            "role": "inference",
            "services": {
                "inference": {"port": 8765},
                "branch": {"port": 8768},
            },
            "models_dir": "/path/to/models",
            "capabilities": ["inference", "eval"],
        },
        "nas": {
            "name": "Synology NAS",
            "host": "192.168.x.x",
            "role": "storage",
            "services": {
                "branch": {"port": 8768},
            },
            "models_dir": "/volume1/data/llm_training/models",
            "capabilities": ["storage", "backup"],
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            config_path: Path to hosts.json (default: auto-detect)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Auto-detect base dir
            base_dir = Path(__file__).parent.parent
            self.config_path = base_dir / "config" / "hosts.json"

        self._hosts: Dict[str, HostConfig] = {}
        self._local_host_id: Optional[str] = None
        self._default_trainer: str = "4090"
        self._default_inference: str = "3090"
        self._lock = threading.Lock()

        self._load()
        self._detect_local_host()

    def _load(self):
        """Load hosts from config file."""
        config_data = {}

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config_data = json.load(f)
                logger.info(f"Loaded hosts from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load hosts.json: {e}, using defaults")

        # Merge with defaults
        hosts_data = config_data.get("hosts", self.DEFAULTS)

        # Also check for old format (inference_hosts, training_hosts)
        if "inference_hosts" in config_data:
            for host_id, data in config_data["inference_hosts"].items():
                if host_id not in hosts_data:
                    hosts_data[host_id] = {**data, "role": "inference"}

        for host_id, data in hosts_data.items():
            self._hosts[host_id] = self._parse_host(host_id, data)

        # Load settings
        self._default_trainer = config_data.get("default_trainer", "4090")
        self._default_inference = config_data.get("default_inference", "3090")
        self._local_host_id = config_data.get("local_host")

    def _parse_host(self, host_id: str, data: Dict) -> HostConfig:
        """Parse host data into HostConfig."""
        # Parse services
        services = {}
        for svc_name, svc_data in data.get("services", {}).items():
            if isinstance(svc_data, dict):
                services[svc_name] = ServiceConfig(
                    port=svc_data.get("port", 8080),
                    path=svc_data.get("path", ""),
                    protocol=svc_data.get("protocol", "http"),
                    auth_required=svc_data.get("auth_required", False),
                )
            elif isinstance(svc_data, int):
                # Simple port number
                services[svc_name] = ServiceConfig(port=svc_data)

        # Parse role
        role_str = data.get("role", "hybrid")
        try:
            role = HostRole(role_str)
        except ValueError:
            role = HostRole.HYBRID

        return HostConfig(
            host_id=host_id,
            name=data.get("name", host_id),
            host=data.get("host", "localhost"),
            role=role,
            services=services,
            ssh_user=data.get("ssh_user", "user"),
            models_dir=data.get("models_dir", ""),
            checkpoints_dir=data.get("checkpoints_dir", ""),
            capabilities=data.get("capabilities", []),
        )

    def _detect_local_host(self):
        """Detect which host we're running on."""
        if self._local_host_id:
            return  # Already set in config

        # Get local IPs
        local_ips = set()
        try:
            hostname = socket.gethostname()
            local_ips.add(socket.gethostbyname(hostname))
        except Exception:
            pass

        local_ips.add("127.0.0.1")
        local_ips.add("localhost")

        # Try to get all local IPs
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                local_ips.add(info[4][0])
        except Exception:
            pass

        # Match against hosts
        for host_id, host in self._hosts.items():
            if host.host in local_ips or host.host == "localhost":
                self._local_host_id = host_id
                logger.info(f"Detected local host: {host_id}")
                return

        logger.info("Could not detect local host, defaulting to trainer")
        self._local_host_id = self._default_trainer

    def get(self, host_id: str) -> Optional[HostConfig]:
        """Get host by ID."""
        return self._hosts.get(host_id)

    def get_trainer(self) -> Optional[HostConfig]:
        """Get the default trainer host."""
        return self._hosts.get(self._default_trainer)

    def get_inference(self) -> Optional[HostConfig]:
        """Get the default inference host."""
        return self._hosts.get(self._default_inference)

    def get_local(self) -> Optional[HostConfig]:
        """Get the local host (the one we're running on)."""
        if self._local_host_id:
            return self._hosts.get(self._local_host_id)
        return None

    def list_all(self) -> List[HostConfig]:
        """List all hosts."""
        return list(self._hosts.values())

    def list_by_role(self, role: HostRole) -> List[HostConfig]:
        """List hosts with a specific role."""
        return [h for h in self._hosts.values() if h.role == role]

    def list_with_service(self, service: str) -> List[HostConfig]:
        """List hosts that provide a specific service."""
        return [h for h in self._hosts.values() if h.has_service(service)]

    def get_service_url(self, service: str, host_id: Optional[str] = None) -> Optional[str]:
        """
        Get URL for a service.

        Args:
            service: Service name (ledger, inference, vault, etc.)
            host_id: Specific host (default: best available)

        Returns:
            Service URL or None if not found
        """
        if host_id:
            host = self._hosts.get(host_id)
            if host:
                return host.get_service_url(service)
            return None

        # Find best host for this service
        # Priority: local host > trainer > first available
        candidates = self.list_with_service(service)
        if not candidates:
            return None

        # Prefer local host
        local = self.get_local()
        if local and local.has_service(service):
            return local.get_service_url(service)

        # Prefer trainer for ledger/vault services
        if service in ("ledger", "vault", "training"):
            trainer = self.get_trainer()
            if trainer and trainer.has_service(service):
                return trainer.get_service_url(service)

        # Prefer inference host for inference service
        if service == "inference":
            inference = self.get_inference()
            if inference and inference.has_service(service):
                return inference.get_service_url(service)

        # Return first available
        return candidates[0].get_service_url(service)

    def is_local(self, host_id: str) -> bool:
        """Check if a host ID refers to the local machine."""
        return host_id == self._local_host_id

    def is_trainer_local(self) -> bool:
        """Check if we're running on the trainer host."""
        return self._local_host_id == self._default_trainer

    def check_health(self, host_id: str, timeout: int = 5) -> bool:
        """Check if a host is reachable."""
        host = self._hosts.get(host_id)
        if not host:
            return False

        # Try each service endpoint
        for service in host.services.values():
            url = service.get_url(host.host)
            try:
                with urlopen(f"{url}/health", timeout=timeout):
                    return True
            except Exception:
                pass

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all hosts."""
        return {
            "total_hosts": len(self._hosts),
            "local_host": self._local_host_id,
            "default_trainer": self._default_trainer,
            "default_inference": self._default_inference,
            "hosts": {
                host_id: {
                    "name": host.name,
                    "role": host.role.value,
                    "services": list(host.services.keys()),
                }
                for host_id, host in self._hosts.items()
            },
        }


# =============================================================================
# SINGLETON AND HELPERS
# =============================================================================

_registry: Optional[HostRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> HostRegistry:
    """Get or create the host registry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = HostRegistry()
    return _registry


def get_host(host_id: str) -> Optional[HostConfig]:
    """Get a host by ID."""
    return get_registry().get(host_id)


def get_service_url(service: str, host_id: Optional[str] = None) -> Optional[str]:
    """Get URL for a service."""
    return get_registry().get_service_url(service, host_id)


def is_local(host_id: str) -> bool:
    """Check if a host ID is the local machine."""
    return get_registry().is_local(host_id)


def is_trainer_local() -> bool:
    """Check if we're on the trainer host."""
    return get_registry().is_trainer_local()


def get_trainer() -> Optional[HostConfig]:
    """Get the trainer host config."""
    return get_registry().get_trainer()


def get_inference() -> Optional[HostConfig]:
    """Get the inference host config."""
    return get_registry().get_inference()


def get_local_host() -> Optional[HostConfig]:
    """Get the local host config."""
    return get_registry().get_local()
