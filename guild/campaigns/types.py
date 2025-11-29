"""
Campaign Types - Dataclasses for campaign management.

Mental Model: A Campaign is a Hero's Journey to Maximum Potential
=================================================================

A Campaign represents ONE ATTEMPT to push a hero (model) as far as it can go.
Think of it like a playthrough in an RPG:

- Hero = The model (fixed identity: Qwen3-0.6B, Qwen3-4B, etc.)
- Campaign = One attempt to train that hero to its maximum potential
- Continue = Keep training, keep learning, push further
- Level Cap = The theoretical limit of what this hero can learn

Each campaign has:
- Its own checkpoints directory (save points on the journey)
- Its own status files (current state of the adventure)
- Progress tracking (how far have we pushed this hero?)
- Milestones (notable achievements along the way)

The goal of a campaign is to discover: "How far can this hero go?"
- What skills can it master?
- What's the lowest loss achievable?
- Where does it plateau?

Different heroes (models) have different potentials:
- A 0.6B model might cap at skill level 20
- A 4B model might reach level 50
- We discover the cap by PLAYING (training)

Starting points:
- starting_checkpoint=None → Fresh start (new game)
- starting_checkpoint="checkpoint-X" → Continue from save (or New Game+)
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
    A Hero's Journey to Maximum Potential.

    A Campaign represents ONE ATTEMPT to push a hero as far as it can go.
    The goal is to discover: "What is this hero's level cap?"

    Think of it as a playthrough:
    - Continue training = keep pushing toward max potential
    - Milestones = achievements along the way
    - Peak metrics = personal bests in this run
    - Archived = a completed journey (successful or not)

    Multiple campaigns for the same hero = multiple attempts/playthroughs.

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
        current_step: Current training step (XP earned)
        total_examples: Total examples trained on
        skills_focus: Skills being trained
        config_overrides: Overrides from hero defaults
        milestones: Notable events
        peak_skill_levels: Highest skill levels achieved (journey progress)
        peak_metrics: Best metrics achieved (personal bests)
        archived_at: When archived (if applicable)
    """
    id: str
    hero_id: str
    name: str
    path: Path

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = CampaignStatus.ACTIVE

    # Starting point (None = fresh start, checkpoint = continue/NG+)
    starting_checkpoint: Optional[str] = None
    starting_step: int = 0

    # Progress - how far we've pushed this hero
    current_step: int = 0
    total_examples: int = 0

    # Configuration
    skills_focus: List[str] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Description
    description: str = ""

    # History & Achievements
    milestones: List[Milestone] = field(default_factory=list)

    # Peak achievements - tracking "how far did we push?"
    # e.g., {"sy": 15, "bin": 12} = highest skill levels reached
    peak_skill_levels: Dict[str, int] = field(default_factory=dict)

    # Personal bests - best metrics achieved in this campaign
    # e.g., {"lowest_loss": 0.82, "highest_accuracy": 0.94}
    peak_metrics: Dict[str, float] = field(default_factory=dict)

    # Effort tracking - materials science metaphor
    # Strain = instantaneous difficulty (loss - floor)
    # Effort = cumulative strain (area under curve)
    # This tracks "how much work did it take to get here?"
    skill_effort: Dict[str, float] = field(default_factory=dict)  # cumulative effort per skill

    # Level transition records - effort spent on each level-up
    # List of dicts with: skill_id, from_level, to_level, effort_spent, efficiency
    level_transitions: List[Dict[str, Any]] = field(default_factory=list)

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
            # Peak achievements - how far did we push?
            "peak_skill_levels": self.peak_skill_levels,
            "peak_metrics": self.peak_metrics,
            # Effort tracking - how much work did it take?
            "skill_effort": self.skill_effort,
            "level_transitions": self.level_transitions,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: Dict, path: Path) -> "Campaign":
        """Create Campaign from dictionary (loading a saved journey)."""
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
            # Peak achievements - how far did we push in this journey?
            peak_skill_levels=data.get("peak_skill_levels", {}),
            peak_metrics=data.get("peak_metrics", {}),
            # Effort tracking - how much work did it take?
            skill_effort=data.get("skill_effort", {}),
            level_transitions=data.get("level_transitions", []),
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

    def update_peak_skill(self, skill_id: str, level: int) -> bool:
        """
        Update peak skill level if this is a new personal best.

        This tracks "how far did we push this hero in this skill?"
        Returns True if this was a new peak (we pushed further!).
        """
        current_peak = self.peak_skill_levels.get(skill_id, 0)
        if level > current_peak:
            self.peak_skill_levels[skill_id] = level
            self.save()
            return True
        return False

    def update_peak_metric(self, metric_name: str, value: float, lower_is_better: bool = True) -> bool:
        """
        Update peak metric if this is a new personal best.

        This tracks "what's the best we've achieved on this metric?"
        For loss: lower_is_better=True (0.5 beats 0.8)
        For accuracy: lower_is_better=False (0.95 beats 0.80)

        Returns True if this was a new peak.
        """
        current_peak = self.peak_metrics.get(metric_name)
        is_new_peak = False

        if current_peak is None:
            is_new_peak = True
        elif lower_is_better and value < current_peak:
            is_new_peak = True
        elif not lower_is_better and value > current_peak:
            is_new_peak = True

        if is_new_peak:
            self.peak_metrics[metric_name] = value
            self.save()
            return True
        return False

    @property
    def journey_summary(self) -> str:
        """Get a one-line summary of this hero's journey."""
        skills_str = ", ".join(
            f"{s}:L{l}" for s, l in self.peak_skill_levels.items()
        ) or "No skills yet"
        return f"{self.name} - Step {self.current_step:,} - {skills_str}"

    # =========================================================================
    # Effort Tracking - Materials Science Metaphor
    # =========================================================================

    def add_effort(self, skill_id: str, effort: float) -> None:
        """
        Add effort spent on a skill.

        Effort = cumulative strain = area under (loss - floor) curve.
        This tracks "how much work went into this skill?"
        """
        current = self.skill_effort.get(skill_id, 0.0)
        self.skill_effort[skill_id] = current + effort
        self.save()

    def record_level_transition(
        self,
        skill_id: str,
        from_level: int,
        to_level: int,
        effort_spent: float,
        plastic_gain: float,
        steps_taken: int = 0,
    ) -> None:
        """
        Record effort spent on a level transition.

        This lets us analyze: "Which level-ups were expensive vs cheap?"

        Args:
            skill_id: The skill that leveled up
            from_level: Previous level
            to_level: New level
            effort_spent: Cumulative strain during this transition
            plastic_gain: Loss improvement (start_loss - end_loss)
            steps_taken: Training steps during this transition
        """
        efficiency = plastic_gain / effort_spent if effort_spent > 0 else 0.0

        record = {
            "skill_id": skill_id,
            "from_level": from_level,
            "to_level": to_level,
            "effort_spent": effort_spent,
            "plastic_gain": plastic_gain,
            "efficiency": efficiency,
            "steps_taken": steps_taken,
            "timestamp": datetime.now().isoformat(),
        }
        self.level_transitions.append(record)
        self.save()

    def get_effort_summary(self) -> Dict[str, Any]:
        """
        Get effort summary for this campaign.

        Returns stats like:
        - Total effort across all skills
        - Effort per skill
        - Average efficiency (plastic_gain / effort)
        - Most/least expensive skills
        """
        total_effort = sum(self.skill_effort.values())

        # Aggregate from level transitions
        transitions_by_skill: Dict[str, List[Dict]] = {}
        for t in self.level_transitions:
            skill = t["skill_id"]
            transitions_by_skill.setdefault(skill, []).append(t)

        skill_stats = {}
        for skill_id, effort in self.skill_effort.items():
            transitions = transitions_by_skill.get(skill_id, [])
            total_plastic = sum(t.get("plastic_gain", 0) for t in transitions)
            avg_efficiency = total_plastic / effort if effort > 0 else 0.0

            skill_stats[skill_id] = {
                "total_effort": effort,
                "levels_gained": len(transitions),
                "total_plastic_gain": total_plastic,
                "avg_efficiency": avg_efficiency,
                "effort_per_level": effort / len(transitions) if transitions else effort,
            }

        # Rank skills
        ranked = sorted(
            skill_stats.items(),
            key=lambda x: x[1]["avg_efficiency"],
            reverse=True
        )

        return {
            "total_effort": total_effort,
            "skill_stats": skill_stats,
            "efficiency_ranking": [s for s, _ in ranked],
            "most_expensive_skill": (
                max(skill_stats.items(), key=lambda x: x[1]["effort_per_level"])[0]
                if skill_stats else None
            ),
            "most_efficient_skill": (
                ranked[0][0] if ranked else None
            ),
            "total_level_transitions": len(self.level_transitions),
        }

    @property
    def total_effort(self) -> float:
        """Total effort spent across all skills."""
        return sum(self.skill_effort.values())


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
