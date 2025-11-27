"""
Herald event types.

Events are the messages broadcast through the realm. Each event has a type,
timestamp, and optional data payload.

Event types follow a hierarchical naming convention:
    category.action (e.g., "quest.started", "hero.level_up")

This allows subscribers to listen to specific events or entire categories
using wildcard patterns (e.g., "quest.*" for all quest events).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """
    All event types in the realm.

    Naming: category.action
    """

    # === Quest Lifecycle ===
    QUEST_RECEIVED = "quest.received"       # Quest dropped in inbox
    QUEST_QUEUED = "quest.queued"           # Quest added to queue
    QUEST_STARTED = "quest.started"         # Battle begins
    QUEST_COMPLETED = "quest.completed"     # Quest finished successfully
    QUEST_FAILED = "quest.failed"           # Quest failed (validation, etc.)
    QUEST_ABANDONED = "quest.abandoned"     # Quest stopped by user

    # === Combat (Training Steps) ===
    COMBAT_ROUND = "combat.round"           # Training step completed
    COMBAT_CRIT = "combat.crit"             # Exceptional performance (low loss)
    COMBAT_HIT = "combat.hit"               # Normal performance
    COMBAT_MISS = "combat.miss"             # Poor performance (high loss)
    COMBAT_CHECKPOINT = "combat.checkpoint" # Checkpoint saved

    # === Hero Progression ===
    XP_GAINED = "hero.xp_gained"            # XP awarded
    LEVEL_UP = "hero.level_up"              # Hero leveled up
    SKILL_LEVEL_UP = "hero.skill_level_up"  # Skill level increased
    SKILL_UNLOCKED = "hero.skill_unlocked"  # New skill unlocked
    EFFECT_APPLIED = "hero.effect_applied"  # Status effect applied
    EFFECT_EXPIRED = "hero.effect_expired"  # Status effect expired

    # === Champions (Checkpoints) ===
    CHAMPION_EVALUATED = "champion.evaluated"   # Checkpoint evaluated
    CHAMPION_CROWNED = "champion.crowned"       # New best checkpoint
    CHAMPION_DEPLOYED = "champion.deployed"     # Deployed to Oracle (3090)
    CHAMPION_RETIRED = "champion.retired"       # Old checkpoint cleaned up

    # === Training System ===
    TRAINING_STARTED = "training.started"   # Daemon started
    TRAINING_PAUSED = "training.paused"     # Training paused
    TRAINING_RESUMED = "training.resumed"   # Training resumed
    TRAINING_STOPPED = "training.stopped"   # Training stopped
    TRAINING_IDLE = "training.idle"         # No quests, waiting

    # === System Health ===
    SYSTEM_HEALTHY = "system.healthy"       # Health check passed
    SYSTEM_WARNING = "system.warning"       # Warning condition
    SYSTEM_ERROR = "system.error"           # Error occurred
    INCIDENT_RAISED = "system.incident"     # Incident created
    INCIDENT_RESOLVED = "system.resolved"   # Incident resolved

    # === Vault (Storage) ===
    VAULT_ASSET_REGISTERED = "vault.registered"   # New asset tracked
    VAULT_ASSET_FETCHED = "vault.fetched"         # Asset fetched from remote
    VAULT_CLEANUP = "vault.cleanup"               # Old assets cleaned

    # === Oracle (Inference) ===
    ORACLE_LOADED = "oracle.loaded"         # Model loaded
    ORACLE_INFERENCE = "oracle.inference"   # Inference completed
    ORACLE_ERROR = "oracle.error"           # Inference error


@dataclass
class Event:
    """
    An event broadcast through the Herald.

    Attributes:
        type: The event type (from EventType enum or custom string)
        data: Event payload (varies by event type)
        timestamp: When the event occurred
        source: Optional identifier of event source
    """
    type: EventType | str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None

    def __post_init__(self):
        # Convert enum to string for consistent handling
        if isinstance(self.type, EventType):
            self.type = self.type.value

    @property
    def category(self) -> str:
        """Get the event category (first part before '.')."""
        return self.type.split('.')[0] if '.' in self.type else self.type

    @property
    def action(self) -> str:
        """Get the event action (second part after '.')."""
        return self.type.split('.')[1] if '.' in self.type else self.type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            type=d["type"],
            data=d.get("data", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            source=d.get("source"),
        )


# Type alias for event callbacks
EventCallback = "Callable[[Event], None]"
