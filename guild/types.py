"""Common types used across the guild module."""

from enum import Enum
from typing import TypeVar, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from uuid import uuid4

T = TypeVar('T')


class Severity(Enum):
    """Universal severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(Enum):
    """Universal status values."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid4())[:8]
    return f"{prefix}_{uid}" if prefix else uid


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string for JSON serialization."""
    return dt.isoformat() if dt else None


def iso_to_datetime(s: str) -> datetime:
    """Convert ISO string to datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


class SerializableMixin:
    """
    Mixin providing to_dict/from_dict for dataclasses.

    Handles:
    - datetime -> ISO string conversion
    - Enum -> value conversion
    - Nested dataclasses with to_dict methods
    - Lists and dicts with nested objects
    """

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {}
        for k, v in asdict(self).items():
            result[k] = self._serialize_value(v)
        return result

    def _serialize_value(self, v: Any) -> Any:
        """Recursively serialize a value."""
        if isinstance(v, datetime):
            return datetime_to_iso(v)
        elif isinstance(v, Enum):
            return v.value
        elif hasattr(v, 'to_dict'):
            return v.to_dict()
        elif isinstance(v, list):
            return [self._serialize_value(item) for item in v]
        elif isinstance(v, dict):
            return {dk: self._serialize_value(dv) for dk, dv in v.items()}
        return v
