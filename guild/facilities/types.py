"""Facility (hardware) type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import os


class FacilityType(Enum):
    """Types of facilities."""
    HUB = "hub"
    BATTLEFIELD = "battlefield"
    ARCHIVE = "archive"
    OUTPOST = "outpost"
    LABORATORY = "laboratory"


@dataclass
class FacilityResource:
    """A specific resource within a facility."""
    id: str
    type: str  # "gpu", "storage", "network"
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FacilityResource":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            type=data["type"],
            properties=data.get("properties", {}),
        )


@dataclass
class Facility:
    """
    A hardware location in the system.
    Loaded from configs/facilities/*.yaml
    """
    id: str
    name: str
    type: FacilityType
    description: str = ""

    host: str = "localhost"
    port: Optional[int] = None

    base_path: str = ""
    paths: dict[str, str] = field(default_factory=dict)

    resources: list[FacilityResource] = field(default_factory=list)

    is_local: bool = True
    is_available: bool = True

    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def get_path(self, key: str, subpath: str = "") -> str:
        """Get a resolved path within this facility."""
        base = os.path.expandvars(os.path.expanduser(self.base_path))
        if key in self.paths:
            path = os.path.join(base, self.paths[key])
        else:
            path = os.path.join(base, key)
        if subpath:
            path = os.path.join(path, subpath)
        return path

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "host": self.host,
            "port": self.port,
            "base_path": self.base_path,
            "paths": self.paths,
            "resources": [r.to_dict() for r in self.resources],
            "is_local": self.is_local,
            "is_available": self.is_available,
            "rpg_name": self.rpg_name,
            "rpg_description": self.rpg_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Facility":
        """Deserialize from dict."""
        resources = [FacilityResource.from_dict(r) for r in data.get("resources", [])]
        ftype = data.get("type")
        if isinstance(ftype, str):
            ftype = FacilityType(ftype)

        return cls(
            id=data["id"],
            name=data["name"],
            type=ftype,
            description=data.get("description", ""),
            host=data.get("host", "localhost"),
            port=data.get("port"),
            base_path=data.get("base_path", ""),
            paths=data.get("paths", {}),
            resources=resources,
            is_local=data.get("is_local", True),
            is_available=data.get("is_available", True),
            rpg_name=data.get("rpg_name"),
            rpg_description=data.get("rpg_description"),
        )
