"""
Herald - The Battle Announcer for the Realm.

The Herald announces important events as they happen:
    - Battle starts/ends
    - Level ups
    - Quest completions
    - Milestones
    - Alerts and warnings

The game UI subscribes to the Herald to show events in the battle log.

RPG Flavor:
    The Herald stands at the edge of the Arena, calling out each
    significant moment in the hero's journey. When DIO levels up,
    the Herald's voice booms across the realm!

Usage:
    herald = Herald(base_dir)

    # Announce events
    herald.announce("Battle started!", "battle", "info")
    herald.announce("Level up! DIO is now level 50!", "level", "success")

    # Get recent events for UI
    events = herald.get_recent(limit=20)

    # Watch for new events (for streaming)
    for event in herald.watch():
        print(event)
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Generator, List, Optional


class EventType(Enum):
    """Types of herald announcements."""
    BATTLE = "battle"       # Battle start/end/pause
    LEVEL = "level"         # Level ups
    QUEST = "quest"         # Quest complete
    SKILL = "skill"         # Skill level up
    MILESTONE = "milestone" # Achievement/milestone
    ALERT = "alert"         # Warning/error
    SYSTEM = "system"       # System events
    IDLE = "idle"           # Idle rewards


class EventSeverity(Enum):
    """Severity of the announcement."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    EPIC = "epic"  # Major achievement


@dataclass
class HeraldEvent:
    """A single announcement from the Herald."""
    id: int
    timestamp: datetime
    message: str
    event_type: EventType
    severity: EventSeverity
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "details": self.details or {},
        }


class Herald:
    """
    The Herald - announces game events.

    Maintains an event log that the game UI can read from.
    Events are stored in memory (recent) and optionally persisted to disk.
    """

    MAX_EVENTS = 1000  # Keep last 1000 events in memory
    PERSIST_FILE = "status/herald_log.json"

    def __init__(self, base_dir: str | Path):
        """
        Initialize the Herald.

        Args:
            base_dir: Base training directory
        """
        self.base_dir = Path(base_dir)
        self.log_file = self.base_dir / self.PERSIST_FILE

        # Event storage
        self._events: Deque[HeraldEvent] = deque(maxlen=self.MAX_EVENTS)
        self._event_counter = 0
        self._lock = threading.Lock()

        # Watchers waiting for new events
        self._watchers: List[threading.Event] = []

        # State tracking for auto-announcements
        self._last_step = 0
        self._last_level = 0
        self._last_battle_state = None
        self._last_quest = None

        # Load persisted events
        self._load_events()

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def announce(
        self,
        message: str,
        event_type: str | EventType = EventType.SYSTEM,
        severity: str | EventSeverity = EventSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
    ) -> HeraldEvent:
        """
        Make an announcement.

        Args:
            message: The announcement text
            event_type: Type of event (battle, level, quest, etc.)
            severity: Severity (info, success, warning, error, epic)
            details: Optional extra data

        Returns:
            The created HeraldEvent
        """
        # Normalize enums
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        if isinstance(severity, str):
            severity = EventSeverity(severity)

        with self._lock:
            self._event_counter += 1
            event = HeraldEvent(
                id=self._event_counter,
                timestamp=datetime.now(),
                message=message,
                event_type=event_type,
                severity=severity,
                details=details,
            )
            self._events.append(event)

            # Notify watchers
            for watcher in self._watchers:
                watcher.set()

        # Persist periodically
        if self._event_counter % 10 == 0:
            self._persist_events()

        return event

    def get_recent(self, limit: int = 20, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent announcements.

        Args:
            limit: Max number of events to return
            event_type: Filter by event type (optional)

        Returns:
            List of event dicts, newest first
        """
        with self._lock:
            events = list(self._events)

        # Filter by type if specified
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]

        # Return newest first, limited
        return [e.to_dict() for e in reversed(events)][:limit]

    def get_since(self, event_id: int) -> List[Dict[str, Any]]:
        """
        Get events since a specific event ID.

        Args:
            event_id: Last seen event ID

        Returns:
            List of new events since that ID
        """
        with self._lock:
            events = [e for e in self._events if e.id > event_id]
        return [e.to_dict() for e in events]

    def watch(self, timeout: float = 30.0) -> Generator[Dict[str, Any], None, None]:
        """
        Watch for new events (generator).

        Args:
            timeout: How long to wait for each event

        Yields:
            Event dicts as they arrive
        """
        last_id = self._event_counter
        watcher = threading.Event()

        with self._lock:
            self._watchers.append(watcher)

        try:
            while True:
                # Wait for new events
                watcher.wait(timeout=timeout)
                watcher.clear()

                # Get new events
                new_events = self.get_since(last_id)
                for event in new_events:
                    last_id = event["id"]
                    yield event

        finally:
            with self._lock:
                self._watchers.remove(watcher)

    # =========================================================================
    # AUTO-ANNOUNCEMENT (call from game loop)
    # =========================================================================

    def check_and_announce(self, game_state: Dict[str, Any]):
        """
        Check game state and auto-announce significant events.

        Call this periodically from the game loop to auto-generate events.

        Args:
            game_state: Current game state dict with keys:
                - step: Current training step
                - level: Hero level
                - is_training: Whether training is active
                - quest_name: Current quest name
                - loss: Current loss
        """
        step = game_state.get("step", 0)
        level = game_state.get("level", 1)
        is_training = game_state.get("is_training", False)
        quest_name = game_state.get("quest_name")
        loss = game_state.get("loss", 0)

        # Check for level up
        if level > self._last_level and self._last_level > 0:
            self.announce(
                f"Level up! DIO is now level {level}!",
                EventType.LEVEL,
                EventSeverity.EPIC,
                {"old_level": self._last_level, "new_level": level},
            )
        self._last_level = level

        # Check for battle state change
        if is_training and self._last_battle_state != "training":
            quest_display = quest_name or "a new quest"
            self.announce(
                f"Battle started! DIO engages {quest_display}",
                EventType.BATTLE,
                EventSeverity.INFO,
                {"quest": quest_name},
            )
        elif not is_training and self._last_battle_state == "training":
            self.announce(
                "Battle complete! DIO returns victorious.",
                EventType.BATTLE,
                EventSeverity.SUCCESS,
            )
        self._last_battle_state = "training" if is_training else "idle"

        # Check for quest change
        if quest_name and quest_name != self._last_quest and is_training:
            self.announce(
                f"New quest accepted: {quest_name}",
                EventType.QUEST,
                EventSeverity.INFO,
                {"quest": quest_name},
            )
        self._last_quest = quest_name

        # Step milestones (every 10k)
        if step > 0 and step // 10000 > self._last_step // 10000:
            milestone = (step // 10000) * 10000
            self.announce(
                f"Milestone reached: {milestone:,} steps!",
                EventType.MILESTONE,
                EventSeverity.SUCCESS,
                {"steps": milestone},
            )
        self._last_step = step

        # Loss warnings
        if loss > 5.0:
            self.announce(
                f"Warning: High damage! Loss = {loss:.4f}",
                EventType.ALERT,
                EventSeverity.WARNING,
                {"loss": loss},
            )

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def battle_started(self, quest_name: str):
        """Announce battle start."""
        self.announce(
            f"Battle started! DIO engages {quest_name}",
            EventType.BATTLE,
            EventSeverity.INFO,
            {"quest": quest_name},
        )

    def battle_ended(self, victory: bool = True, quest_name: Optional[str] = None):
        """Announce battle end."""
        if victory:
            self.announce(
                f"Victory! Quest '{quest_name}' complete!" if quest_name else "Victory! Quest complete!",
                EventType.BATTLE,
                EventSeverity.SUCCESS,
                {"victory": True, "quest": quest_name},
            )
        else:
            self.announce(
                "Battle ended - DIO retreats to recover.",
                EventType.BATTLE,
                EventSeverity.WARNING,
                {"victory": False},
            )

    def level_up(self, new_level: int):
        """Announce level up."""
        self.announce(
            f"LEVEL UP! DIO reached level {new_level}!",
            EventType.LEVEL,
            EventSeverity.EPIC,
            {"level": new_level},
        )

    def skill_up(self, skill_name: str, new_level: int):
        """Announce skill level up."""
        self.announce(
            f"Skill improved! {skill_name} is now level {new_level}!",
            EventType.SKILL,
            EventSeverity.SUCCESS,
            {"skill": skill_name, "level": new_level},
        )

    def idle_reward(self, xp_amount: int):
        """Announce idle XP reward."""
        self.announce(
            f"+{xp_amount} XP (idle bonus)",
            EventType.IDLE,
            EventSeverity.INFO,
            {"xp": xp_amount},
        )

    def system_message(self, message: str, severity: str = "info"):
        """Announce system message."""
        self.announce(message, EventType.SYSTEM, EventSeverity(severity))

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _persist_events(self):
        """Save recent events to disk."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                events = list(self._events)

            # Keep last 100 for persistence
            data = {
                "last_id": self._event_counter,
                "events": [e.to_dict() for e in events[-100:]],
            }

            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception:
            pass  # Fail silently

    def _load_events(self):
        """Load events from disk."""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file) as f:
                data = json.load(f)

            self._event_counter = data.get("last_id", 0)

            for event_data in data.get("events", []):
                event = HeraldEvent(
                    id=event_data["id"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    message=event_data["message"],
                    event_type=EventType(event_data["event_type"]),
                    severity=EventSeverity(event_data["severity"]),
                    details=event_data.get("details"),
                )
                self._events.append(event)

        except Exception:
            pass  # Start fresh on error


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_herald: Optional[Herald] = None


def get_herald(base_dir: str | Path = "/path/to/training") -> Herald:
    """Get or create the global Herald instance."""
    global _herald
    if _herald is None:
        _herald = Herald(base_dir)
    return _herald


def announce(message: str, event_type: str = "system", severity: str = "info"):
    """Quick announce using global herald."""
    return get_herald().announce(message, event_type, severity)
