"""Incident type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from guild.types import Severity, datetime_to_iso, iso_to_datetime


class IncidentCategory(Enum):
    """Categories of incidents."""
    DATA = "data"
    TRAINING = "training"
    INFRA = "infra"
    LOGIC = "logic"


class IncidentStatus(Enum):
    """Incident lifecycle."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    WONTFIX = "wontfix"


@dataclass
class Incident:
    """A detected problem/bug."""
    id: str
    category: IncidentCategory
    severity: Severity

    title: str
    description: str

    detected_at_step: int
    detected_at_time: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    quest_id: Optional[str] = None
    facility_id: Optional[str] = None

    context: dict = field(default_factory=dict)

    status: IncidentStatus = IncidentStatus.OPEN
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    rpg_name: Optional[str] = None
    rpg_location: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "detected_at_step": self.detected_at_step,
            "detected_at_time": datetime_to_iso(self.detected_at_time),
            "run_id": self.run_id,
            "quest_id": self.quest_id,
            "facility_id": self.facility_id,
            "context": self.context,
            "status": self.status.value,
            "resolution": self.resolution,
            "resolved_at": datetime_to_iso(self.resolved_at),
            "rpg_name": self.rpg_name,
            "rpg_location": self.rpg_location,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Incident":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            category=IncidentCategory(data["category"]),
            severity=Severity(data["severity"]),
            title=data["title"],
            description=data["description"],
            detected_at_step=data["detected_at_step"],
            detected_at_time=iso_to_datetime(data.get("detected_at_time")) or datetime.now(),
            run_id=data.get("run_id"),
            quest_id=data.get("quest_id"),
            facility_id=data.get("facility_id"),
            context=data.get("context", {}),
            status=IncidentStatus(data.get("status", "open")),
            resolution=data.get("resolution"),
            resolved_at=iso_to_datetime(data.get("resolved_at")),
            rpg_name=data.get("rpg_name"),
            rpg_location=data.get("rpg_location"),
        )


@dataclass
class IncidentRule:
    """Rule for detecting incidents."""
    id: str
    name: str
    category: IncidentCategory
    severity: Severity

    detector_type: str
    detector_config: dict = field(default_factory=dict)

    title_template: str = ""
    description_template: str = ""

    rpg_name_template: Optional[str] = None
