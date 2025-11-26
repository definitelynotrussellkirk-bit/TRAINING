"""Run type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from guild.types import Status, datetime_to_iso, iso_to_datetime


class RunType(Enum):
    """Types of runs."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    AUDIT = "audit"
    EXPERIMENT = "experiment"
    GENERATION = "generation"


@dataclass
class RunConfig:
    """Configuration for a run."""
    id: str
    type: RunType
    name: str = ""
    description: str = ""

    facility_id: str = ""
    hero_id: str = ""

    quest_filters: dict = field(default_factory=dict)

    max_steps: Optional[int] = None
    max_quests: Optional[int] = None
    max_duration_seconds: Optional[int] = None

    hyperparams: dict = field(default_factory=dict)

    log_level: str = "INFO"
    log_facility_id: str = ""

    checkpoint_every_steps: int = 1000
    checkpoint_facility_id: str = ""

    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "facility_id": self.facility_id,
            "hero_id": self.hero_id,
            "quest_filters": self.quest_filters,
            "max_steps": self.max_steps,
            "max_quests": self.max_quests,
            "max_duration_seconds": self.max_duration_seconds,
            "hyperparams": self.hyperparams,
            "log_level": self.log_level,
            "log_facility_id": self.log_facility_id,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "checkpoint_facility_id": self.checkpoint_facility_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            type=RunType(data["type"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            facility_id=data.get("facility_id", ""),
            hero_id=data.get("hero_id", ""),
            quest_filters=data.get("quest_filters", {}),
            max_steps=data.get("max_steps"),
            max_quests=data.get("max_quests"),
            max_duration_seconds=data.get("max_duration_seconds"),
            hyperparams=data.get("hyperparams", {}),
            log_level=data.get("log_level", "INFO"),
            log_facility_id=data.get("log_facility_id", ""),
            checkpoint_every_steps=data.get("checkpoint_every_steps", 1000),
            checkpoint_facility_id=data.get("checkpoint_facility_id", ""),
            tags=data.get("tags", []),
        )


@dataclass
class RunState:
    """Current state of a run."""
    run_id: str
    config: RunConfig
    status: Status = Status.PENDING

    current_step: int = 0
    quests_completed: int = 0
    quests_succeeded: int = 0

    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    metrics: dict[str, Any] = field(default_factory=dict)

    last_checkpoint_step: int = 0
    checkpoint_paths: list[str] = field(default_factory=list)

    incident_ids: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total run duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Quest success rate."""
        if self.quests_completed == 0:
            return 0.0
        return self.quests_succeeded / self.quests_completed

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_step": self.current_step,
            "quests_completed": self.quests_completed,
            "quests_succeeded": self.quests_succeeded,
            "started_at": datetime_to_iso(self.started_at),
            "paused_at": datetime_to_iso(self.paused_at),
            "completed_at": datetime_to_iso(self.completed_at),
            "metrics": self.metrics,
            "last_checkpoint_step": self.last_checkpoint_step,
            "checkpoint_paths": self.checkpoint_paths,
            "incident_ids": self.incident_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunState":
        """Deserialize from dict."""
        return cls(
            run_id=data["run_id"],
            config=RunConfig.from_dict(data["config"]),
            status=Status(data["status"]),
            current_step=data.get("current_step", 0),
            quests_completed=data.get("quests_completed", 0),
            quests_succeeded=data.get("quests_succeeded", 0),
            started_at=iso_to_datetime(data.get("started_at")),
            paused_at=iso_to_datetime(data.get("paused_at")),
            completed_at=iso_to_datetime(data.get("completed_at")),
            metrics=data.get("metrics", {}),
            last_checkpoint_step=data.get("last_checkpoint_step", 0),
            checkpoint_paths=data.get("checkpoint_paths", []),
            incident_ids=data.get("incident_ids", []),
        )
