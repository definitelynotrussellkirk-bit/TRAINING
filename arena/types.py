"""
Arena Types - Data structures for the Training Arena.

The Arena is where the hero (model) battles quest challenges (training data)
to gain experience and grow stronger.

RPG Mapping:
    Training Step → Combat Round
    Epoch → Campaign
    Loss → Damage Taken
    Accuracy → Hit Rate
    Checkpoint → Hero Snapshot
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class BattleState(Enum):
    """State of the current battle (training run)."""
    IDLE = "idle"               # No battle active
    PREPARING = "preparing"     # Loading gear and supplies
    FIGHTING = "fighting"       # Training in progress
    PAUSED = "paused"           # Battle paused
    VICTORY = "victory"         # Training completed successfully
    RETREAT = "retreat"         # Training stopped early
    DEFEATED = "defeated"       # Training crashed/failed


class CombatResult(Enum):
    """Outcome of evaluating a single quest."""
    CRIT = "crit"       # Perfect answer (exact match)
    HIT = "hit"         # Good answer (acceptable)
    GRAZE = "graze"     # Partial answer
    MISS = "miss"       # Wrong answer
    FUMBLE = "fumble"   # Error during evaluation


@dataclass
class BattleStatus:
    """
    Current status of the battle (training run).

    Displayed on the War Room dashboard.
    """
    # Battle state
    state: BattleState = BattleState.IDLE
    quest_file: Optional[str] = None

    # Combat progress
    current_round: int = 0          # Training step
    total_rounds: int = 0           # Total steps
    campaign: int = 0               # Current epoch
    total_campaigns: int = 1        # Total epochs

    # Combat metrics
    damage_taken: float = 0.0       # Training loss
    validation_damage: float = 0.0  # Validation loss
    hit_rate: Optional[float] = None  # Accuracy

    # Timing
    rounds_per_second: float = 0.0  # Steps/sec
    time_remaining: Optional[str] = None
    eta: Optional[str] = None

    # Hero state
    hero_checkpoint: Optional[str] = None
    hero_vram_mb: float = 0.0

    # Timestamps
    battle_started: Optional[datetime] = None
    last_update: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "state": self.state.value,
            "quest_file": self.quest_file,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "campaign": self.campaign,
            "total_campaigns": self.total_campaigns,
            "damage_taken": self.damage_taken,
            "validation_damage": self.validation_damage,
            "hit_rate": self.hit_rate,
            "rounds_per_second": self.rounds_per_second,
            "time_remaining": self.time_remaining,
            "eta": self.eta,
            "hero_checkpoint": self.hero_checkpoint,
            "hero_vram_mb": self.hero_vram_mb,
            "battle_started": self.battle_started.isoformat() if self.battle_started else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class QuestBoardStatus:
    """Status of the Quest Board (training queue)."""
    # Queue counts
    urgent: int = 0         # High priority
    standard: int = 0       # Normal priority
    reserves: int = 0       # Low priority
    active_duty: int = 0    # Currently processing
    fallen: int = 0         # Failed

    # Totals
    total_pending: int = 0
    inbox: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "urgent": self.urgent,
            "standard": self.standard,
            "reserves": self.reserves,
            "active_duty": self.active_duty,
            "fallen": self.fallen,
            "total_pending": self.total_pending,
            "inbox": self.inbox,
        }


@dataclass
class HeroSnapshot:
    """
    A snapshot of the hero (model checkpoint).

    Saved periodically during battle to allow resurrection if defeated.
    """
    checkpoint_path: str
    step: int
    campaign: int               # Epoch
    damage_taken: float         # Loss
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional metrics
    validation_damage: Optional[float] = None
    hit_rate: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "checkpoint_path": self.checkpoint_path,
            "step": self.step,
            "campaign": self.campaign,
            "damage_taken": self.damage_taken,
            "validation_damage": self.validation_damage,
            "hit_rate": self.hit_rate,
            "timestamp": self.timestamp.isoformat(),
        }
