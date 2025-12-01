"""
Fleet Types - Shared data structures for fleet management.

These types are used by both agents and the controller for
communication and health reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeStatus(str, Enum):
    """Overall status of a node."""
    HEALTHY = "healthy"       # All systems nominal
    WARNING = "warning"       # Some thresholds exceeded
    CRITICAL = "critical"     # Immediate action needed
    OFFLINE = "offline"       # No heartbeat received
    UNKNOWN = "unknown"       # Never seen


class StorageZone(str, Enum):
    """Storage temperature zones."""
    HOT = "hot"       # Fast local NVMe
    WARM = "warm"     # Primary NAS
    COLD = "cold"     # Archive


@dataclass
class StorageHealth:
    """Health metrics for a storage path."""
    path: str
    zone: StorageZone
    total_bytes: int
    used_bytes: int
    free_bytes: int
    checkpoint_count: int = 0
    oldest_checkpoint: Optional[str] = None
    newest_checkpoint: Optional[str] = None

    @property
    def used_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "zone": self.zone.value,
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "used_pct": round(self.used_pct, 1),
            "checkpoint_count": self.checkpoint_count,
            "oldest_checkpoint": self.oldest_checkpoint,
            "newest_checkpoint": self.newest_checkpoint,
        }


@dataclass
class GPUHealth:
    """Health metrics for a GPU."""
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: float
    temperature_c: int
    power_draw_w: float

    @property
    def vram_used_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "vram_used_pct": round(self.vram_used_pct, 1),
            "utilization_pct": round(self.utilization_pct, 1),
            "temperature_c": self.temperature_c,
            "power_draw_w": round(self.power_draw_w, 1),
        }


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    cpu_pct: float
    memory_mb: float
    started_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "cpu_pct": round(self.cpu_pct, 1),
            "memory_mb": round(self.memory_mb, 1),
            "started_at": self.started_at,
        }


@dataclass
class RetentionPolicy:
    """Retention policy for a node."""
    max_checkpoints: Optional[int]
    max_gb: Optional[float]
    keep_strategy: str  # "recently_used", "all", "recent"
    is_vault: bool
    cleanup_threshold_pct: float = 90.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_checkpoints": self.max_checkpoints,
            "max_gb": self.max_gb,
            "keep_strategy": self.keep_strategy,
            "is_vault": self.is_vault,
            "cleanup_threshold_pct": self.cleanup_threshold_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetentionPolicy":
        return cls(
            max_checkpoints=data.get("max_checkpoints"),
            max_gb=data.get("max_gb"),
            keep_strategy=data.get("keep_strategy", "recently_used"),
            is_vault=data.get("is_vault", False),
            cleanup_threshold_pct=data.get("cleanup_threshold_pct", 90.0),
        )


@dataclass
class NodeHealth:
    """Complete health snapshot for a node."""
    host_id: str
    device_id: str
    hostname: str
    status: NodeStatus
    timestamp: str  # ISO format
    uptime_seconds: int

    # System resources
    cpu_pct: float
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int

    # Storage health per zone
    storage: List[StorageHealth] = field(default_factory=list)

    # GPU health (if applicable)
    gpus: List[GPUHealth] = field(default_factory=list)

    # Running processes of interest
    processes: List[ProcessInfo] = field(default_factory=list)

    # Retention policy
    retention_policy: Optional[RetentionPolicy] = None

    # Alerts
    alerts: List[str] = field(default_factory=list)

    # Metrics
    last_retention_run: Optional[str] = None
    checkpoints_deleted_today: int = 0
    bytes_freed_today: int = 0

    @property
    def memory_used_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100

    @property
    def needs_retention(self) -> bool:
        """Check if any storage zone exceeds threshold."""
        if self.retention_policy and self.retention_policy.is_vault:
            return False
        threshold = self.retention_policy.cleanup_threshold_pct if self.retention_policy else 90.0
        return any(s.used_pct > threshold for s in self.storage)

    @property
    def checkpoint_count(self) -> int:
        """Total checkpoints across all storage."""
        return sum(s.checkpoint_count for s in self.storage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "device_id": self.device_id,
            "hostname": self.hostname,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "cpu_pct": round(self.cpu_pct, 1),
            "memory": {
                "total_mb": self.memory_total_mb,
                "used_mb": self.memory_used_mb,
                "free_mb": self.memory_free_mb,
                "used_pct": round(self.memory_used_pct, 1),
            },
            "storage": [s.to_dict() for s in self.storage],
            "gpus": [g.to_dict() for g in self.gpus],
            "processes": [p.to_dict() for p in self.processes],
            "retention_policy": self.retention_policy.to_dict() if self.retention_policy else None,
            "alerts": self.alerts,
            "needs_retention": self.needs_retention,
            "checkpoint_count": self.checkpoint_count,
            "last_retention_run": self.last_retention_run,
            "checkpoints_deleted_today": self.checkpoints_deleted_today,
            "bytes_freed_today": self.bytes_freed_today,
        }

    @classmethod
    def offline(cls, host_id: str, device_id: str, hostname: str) -> "NodeHealth":
        """Create an offline health report."""
        return cls(
            host_id=host_id,
            device_id=device_id,
            hostname=hostname,
            status=NodeStatus.OFFLINE,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=0,
            cpu_pct=0,
            memory_total_mb=0,
            memory_used_mb=0,
            memory_free_mb=0,
            alerts=["Node is offline - no heartbeat received"],
        )


@dataclass
class RetentionResult:
    """Result of a retention run."""
    host_id: str
    device_id: str
    timestamp: str
    dry_run: bool
    checkpoints_before: int
    checkpoints_after: int
    deleted_count: int
    deleted_steps: List[int]
    freed_bytes: int
    errors: List[str]

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host_id": self.host_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "dry_run": self.dry_run,
            "checkpoints_before": self.checkpoints_before,
            "checkpoints_after": self.checkpoints_after,
            "deleted_count": self.deleted_count,
            "deleted_steps": self.deleted_steps,
            "freed_gb": round(self.freed_gb, 2),
            "errors": self.errors,
        }


@dataclass
class FleetStatus:
    """Aggregate status of all nodes in the fleet."""
    timestamp: str
    total_nodes: int
    healthy_nodes: int
    warning_nodes: int
    critical_nodes: int
    offline_nodes: int
    nodes: Dict[str, NodeHealth]

    # Aggregate metrics
    total_checkpoints: int = 0
    total_storage_gb: float = 0
    total_storage_used_gb: float = 0
    total_vram_gb: float = 0
    total_vram_used_gb: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_nodes": self.total_nodes,
                "healthy": self.healthy_nodes,
                "warning": self.warning_nodes,
                "critical": self.critical_nodes,
                "offline": self.offline_nodes,
            },
            "aggregate": {
                "total_checkpoints": self.total_checkpoints,
                "total_storage_gb": round(self.total_storage_gb, 1),
                "storage_used_gb": round(self.total_storage_used_gb, 1),
                "storage_used_pct": round(
                    (self.total_storage_used_gb / self.total_storage_gb * 100)
                    if self.total_storage_gb > 0 else 0, 1
                ),
                "total_vram_gb": round(self.total_vram_gb, 1),
                "vram_used_gb": round(self.total_vram_used_gb, 1),
            },
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }
