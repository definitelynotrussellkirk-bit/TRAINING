"""
Saga types - The narrative log of the realm.

A Saga is a persistent, append-only log of tales (events) that happened
in the realm. Unlike the Herald (which is ephemeral), the Saga persists
to disk and can be displayed in the Tavern UI as a scrolling chat log.

Each TaleEntry is a single line in the narrative, with:
- Timestamp
- Icon (emoji)
- Message (human-readable)
- Event type (for filtering)
- Optional data payload
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import json


class TaleCategory(str, Enum):
    """Categories for filtering tales in the UI."""
    QUEST = "quest"
    COMBAT = "combat"
    HERO = "hero"
    CHAMPION = "champion"
    TRAINING = "training"
    SYSTEM = "system"
    VAULT = "vault"
    ORACLE = "oracle"


@dataclass
class TaleEntry:
    """
    A single entry in the Saga - one line of narrative.

    Displayed in the Tavern as:
        [14:23:01] ⚔️ DIO begins quest: binary_L5.jsonl

    Attributes:
        timestamp: When this tale occurred
        icon: Emoji icon for visual display
        message: Human-readable narrative message
        event_type: Original event type (for filtering/analysis)
        category: High-level category (for UI tabs)
        data: Additional structured data
    """
    timestamp: datetime
    icon: str
    message: str
    event_type: str = ""
    category: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ts": self.timestamp.isoformat(),
            "icon": self.icon,
            "msg": self.message,
            "type": self.event_type,
            "cat": self.category,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaleEntry":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            timestamp=datetime.fromisoformat(d["ts"]),
            icon=d.get("icon", ""),
            message=d.get("msg", ""),
            event_type=d.get("type", ""),
            category=d.get("cat", ""),
            data=d.get("data", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "TaleEntry":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))

    def format_display(self, show_date: bool = False) -> str:
        """
        Format for display in the UI.

        Args:
            show_date: Include date (for multi-day views)

        Returns:
            Formatted string like "[14:23:01] ⚔️ DIO begins quest..."
        """
        if show_date:
            time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = self.timestamp.strftime("%H:%M:%S")

        return f"[{time_str}] {self.icon} {self.message}"

    @property
    def age_seconds(self) -> float:
        """Seconds since this tale occurred."""
        return (datetime.now() - self.timestamp).total_seconds()

    @property
    def is_recent(self) -> bool:
        """Is this tale from the last 5 minutes?"""
        return self.age_seconds < 300


# Icon mapping for event types -> display icons
TALE_ICONS: Dict[str, str] = {
    # Quest lifecycle
    "quest.received": "📥",
    "quest.queued": "📋",
    "quest.started": "⚔️",
    "quest.completed": "✅",
    "quest.failed": "❌",
    "quest.abandoned": "🏃",

    # Combat
    "combat.round": "⚡",
    "combat.crit": "💥",
    "combat.hit": "🎯",
    "combat.miss": "💨",
    "combat.checkpoint": "💾",

    # Hero progression
    "hero.xp_gained": "✨",
    "hero.level_up": "⬆️",
    "hero.skill_level_up": "📈",
    "hero.skill_unlocked": "🔓",
    "hero.effect_applied": "🌟",
    "hero.effect_expired": "💫",

    # Champions
    "champion.evaluated": "📊",
    "champion.crowned": "🏆",
    "champion.deployed": "🚀",
    "champion.retired": "📦",

    # Training system
    "training.started": "🏁",
    "training.paused": "⏸️",
    "training.resumed": "▶️",
    "training.stopped": "⏹️",
    "training.idle": "😴",

    # System
    "system.healthy": "💚",
    "system.warning": "⚠️",
    "system.error": "🔴",
    "system.incident": "🚨",
    "system.resolved": "✅",

    # Vault
    "vault.registered": "📁",
    "vault.fetched": "📥",
    "vault.cleanup": "🧹",

    # Oracle
    "oracle.loaded": "🔮",
    "oracle.inference": "💬",
    "oracle.error": "💀",
}


def get_icon(event_type: str) -> str:
    """Get the icon for an event type."""
    return TALE_ICONS.get(event_type, "📜")


def get_category(event_type: str) -> str:
    """Get the category from an event type."""
    if "." in event_type:
        return event_type.split(".")[0]
    return event_type
