"""
Base adapter classes and common utilities.

Adapters provide the bridge between guild framework and existing systems:
- Training daemon and data pipeline
- Inference server (RTX 3090)
- Curriculum management
- Training queue
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Generic, TypeVar

T = TypeVar('T')
R = TypeVar('R')


class AdapterStatus(Enum):
    """Status of adapter operations."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    NOT_AVAILABLE = "not_available"


@dataclass
class AdapterConfig:
    """
    Configuration for adapters.

    Provides common settings like timeouts, retries, and paths.
    """
    base_dir: Path = field(default_factory=lambda: Path("."))

    # Connection settings
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Inference settings
    inference_host: str = "192.168.x.x"
    inference_port: int = 8765

    # Queue settings
    default_priority: str = "normal"

    # Paths
    inbox_dir: Optional[Path] = None
    queue_dir: Optional[Path] = None
    status_dir: Optional[Path] = None

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        if self.inbox_dir is None:
            self.inbox_dir = self.base_dir / "inbox"
        if self.queue_dir is None:
            self.queue_dir = self.base_dir / "queue"
        if self.status_dir is None:
            self.status_dir = self.base_dir / "status"

    @property
    def inference_url(self) -> str:
        return f"http://{self.inference_host}:{self.inference_port}"

    def to_dict(self) -> dict:
        return {
            "base_dir": str(self.base_dir),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "inference_host": self.inference_host,
            "inference_port": self.inference_port,
            "default_priority": self.default_priority,
        }


@dataclass
class AdapterResult(Generic[T]):
    """
    Result of an adapter operation.

    Wraps the result with status and metadata.
    """
    status: AdapterStatus
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == AdapterStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == AdapterStatus.FAILURE

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def ok(cls, data: T, **metadata) -> "AdapterResult[T]":
        """Create a success result."""
        return cls(
            status=AdapterStatus.SUCCESS,
            data=data,
            metadata=metadata,
        )

    @classmethod
    def fail(cls, error: str, **metadata) -> "AdapterResult[T]":
        """Create a failure result."""
        return cls(
            status=AdapterStatus.FAILURE,
            error=error,
            metadata=metadata,
        )

    @classmethod
    def timeout(cls, error: str = "Operation timed out") -> "AdapterResult[T]":
        """Create a timeout result."""
        return cls(
            status=AdapterStatus.TIMEOUT,
            error=error,
        )

    @classmethod
    def not_available(cls, error: str = "Service not available") -> "AdapterResult[T]":
        """Create a not available result."""
        return cls(
            status=AdapterStatus.NOT_AVAILABLE,
            error=error,
        )


class BaseAdapter(ABC):
    """
    Base class for all adapters.

    Provides common functionality like configuration,
    health checks, and result wrapping.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self._healthy = False
        self._last_health_check: Optional[datetime] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name for logging and identification."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the adapter's target system is available.

        Returns:
            True if system is available, False otherwise
        """
        pass

    @property
    def is_healthy(self) -> bool:
        """Current health status."""
        return self._healthy

    def refresh_health(self) -> bool:
        """Refresh health status."""
        self._healthy = self.health_check()
        self._last_health_check = datetime.now()
        return self._healthy

    def require_healthy(self) -> None:
        """
        Ensure adapter is healthy, raise if not.

        Raises:
            RuntimeError: If adapter is not healthy
        """
        if not self.refresh_health():
            raise RuntimeError(f"{self.name} adapter is not healthy")

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status for monitoring."""
        return {
            "name": self.name,
            "healthy": self._healthy,
            "last_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "config": self.config.to_dict(),
        }


class CompositeAdapter:
    """
    Combines multiple adapters for coordinated operations.

    Example:
        composite = CompositeAdapter()
        composite.add(inference_adapter)
        composite.add(queue_adapter)

        if composite.all_healthy():
            # All systems ready
            pass
    """

    def __init__(self):
        self._adapters: Dict[str, BaseAdapter] = {}

    def add(self, adapter: BaseAdapter) -> None:
        """Add an adapter."""
        self._adapters[adapter.name] = adapter

    def remove(self, name: str) -> None:
        """Remove an adapter."""
        self._adapters.pop(name, None)

    def get(self, name: str) -> Optional[BaseAdapter]:
        """Get an adapter by name."""
        return self._adapters.get(name)

    def all_healthy(self) -> bool:
        """Check if all adapters are healthy."""
        return all(a.refresh_health() for a in self._adapters.values())

    def healthy_adapters(self) -> List[str]:
        """Get names of healthy adapters."""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.refresh_health()
        ]

    def unhealthy_adapters(self) -> List[str]:
        """Get names of unhealthy adapters."""
        return [
            name for name, adapter in self._adapters.items()
            if not adapter.refresh_health()
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get status of all adapters."""
        return {
            name: adapter.get_status()
            for name, adapter in self._adapters.items()
        }


# Global adapter registry
_adapters: Dict[str, BaseAdapter] = {}


def register_adapter(adapter: BaseAdapter) -> None:
    """Register a global adapter."""
    _adapters[adapter.name] = adapter


def get_adapter(name: str) -> Optional[BaseAdapter]:
    """Get a global adapter by name."""
    return _adapters.get(name)


def list_adapters() -> List[str]:
    """List all registered adapter names."""
    return list(_adapters.keys())


def reset_adapters() -> None:
    """Reset all global adapters (for testing)."""
    global _adapters
    _adapters = {}
