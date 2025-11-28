"""
Campaign Types - Dataclasses for campaign management.

A Campaign is a training run for a specific hero. Each campaign has:
- Its own checkpoints directory
- Its own status files (training_status, curriculum_state, etc.)
- Metadata (name, dates, milestones)
- Config overrides from the hero defaults

RPG Flavor:
    A Campaign is an adventure - a hero's journey through training data,
    battling loss functions, gaining XP, and leveling up skills.
    Each campaign is a separate "save file" that can be archived or continued.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class Milestone:
    """
    A notable event in a campaign's history.

    Attributes:
        step: Training step when milestone occurred
        note: Description of the milestone
        date: When it happened
        metrics: Optional metrics snapshot at this point
    """
    step: int
    note: str
    date: str
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "note": self.note,
            "date": self.date,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Milestone":
        return cls(
            step=data["step"],
            note=data["note"],
            date=data["date"],
            metrics=data.get("metrics"),
        )


@dataclass
class CampaignStatus:
    """Campaign status enumeration as a namespace."""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    COMPLETED = "completed"


@dataclass
class Campaign:
    """
    A training campaign for a hero.

    A Campaign represents a single training run with its own checkpoints,
    status files, and configuration. Multiple campaigns can exist for the
    same hero (like save slots).

    Attributes:
        id: Campaign identifier (e.g., "campaign-001")
        hero_id: Hero this campaign belongs to
        name: Display name for the campaign
        description: Optional longer description
        path: Filesystem path to campaign directory
        created_at: When campaign was created
        status: Current status (active, paused, archived)
        starting_checkpoint: Checkpoint to start from (None = base model)
        starting_step: Step number at start
        current_step: Current training step
        total_examples: Total examples trained on
        skills_focus: Skills being trained
        config_overrides: Overrides from hero defaults
        milestones: Notable events
        archived_at: When archived (if applicable)
    """
    id: str
    hero_id: str
    name: str
    path: Path

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = CampaignStatus.ACTIVE

    # Starting point
    starting_checkpoint: Optional[str] = None
    starting_step: int = 0

    # Progress
    current_step: int = 0
    total_examples: int = 0

    # Configuration
    skills_focus: List[str] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Description
    description: str = ""

    # History
    milestones: List[Milestone] = field(default_factory=list)

    # Archive info
    archived_at: Optional[str] = None

    @property
    def checkpoints_dir(self) -> Path:
        """Path to checkpoints directory."""
        return self.path / "checkpoints"

    @property
    def status_dir(self) -> Path:
        """Path to status files directory."""
        return self.path / "status"

    @property
    def logs_dir(self) -> Path:
        """Path to logs directory."""
        return self.path / "logs"

    @property
    def is_active(self) -> bool:
        """Check if campaign is active."""
        return self.status == CampaignStatus.ACTIVE

    @property
    def is_archived(self) -> bool:
        """Check if campaign is archived."""
        return self.status == CampaignStatus.ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "hero_id": self.hero_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "status": self.status,
            "starting_checkpoint": self.starting_checkpoint,
            "starting_step": self.starting_step,
            "current_step": self.current_step,
            "total_examples": self.total_examples,
            "skills_focus": self.skills_focus,
            "config_overrides": self.config_overrides,
            "milestones": [m.to_dict() for m in self.milestones],
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: Dict, path: Path) -> "Campaign":
        """Create Campaign from dictionary."""
        milestones = [
            Milestone.from_dict(m) for m in data.get("milestones", [])
        ]
        return cls(
            id=data["id"],
            hero_id=data["hero_id"],
            name=data["name"],
            path=path,
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            status=data.get("status", CampaignStatus.ACTIVE),
            starting_checkpoint=data.get("starting_checkpoint"),
            starting_step=data.get("starting_step", 0),
            current_step=data.get("current_step", 0),
            total_examples=data.get("total_examples", 0),
            skills_focus=data.get("skills_focus", []),
            config_overrides=data.get("config_overrides", {}),
            milestones=milestones,
            archived_at=data.get("archived_at"),
        )

    def save(self) -> None:
        """Save campaign metadata to campaign.json."""
        config_path = self.path / "campaign.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def add_milestone(self, step: int, note: str, metrics: Optional[Dict] = None) -> None:
        """Add a milestone to the campaign."""
        milestone = Milestone(
            step=step,
            note=note,
            date=datetime.now().strftime("%Y-%m-%d"),
            metrics=metrics,
        )
        self.milestones.append(milestone)
        self.save()

    def update_progress(self, step: int, examples: Optional[int] = None) -> None:
        """Update campaign progress."""
        self.current_step = step
        if examples is not None:
            self.total_examples = examples
        self.save()


@dataclass
class ActiveCampaignPointer:
    """
    Pointer to the currently active campaign.

    This is the "Scroll of Destiny" - stored at control/active_campaign.json
    and determines which campaign is currently being trained.

    Attributes:
        hero_id: Active hero ID
        campaign_id: Active campaign ID
        campaign_path: Relative path to campaign directory
        activated_at: When this campaign was activated
    """
    hero_id: str
    campaign_id: str
    campaign_path: str
    activated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "campaign_id": self.campaign_id,
            "campaign_path": self.campaign_path,
            "activated_at": self.activated_at,
            "_comment": "Scroll of Destiny - Points to the currently active campaign",
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ActiveCampaignPointer":
        return cls(
            hero_id=data["hero_id"],
            campaign_id=data["campaign_id"],
            campaign_path=data["campaign_path"],
            activated_at=data.get("activated_at", datetime.now().isoformat()),
        )
