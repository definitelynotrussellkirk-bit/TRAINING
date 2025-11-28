"""
HeraldBridge - Connects Herald events to the Saga.

The bridge subscribes to Herald events and automatically records them
as tales in the Saga. This means any event emitted through the Herald
will appear in the Tavern's scrolling log.

Message Templates:
    Each event type has a template that formats event data into
    human-readable messages. Templates use Python format strings
    with access to all event data fields.

Usage:
    from guild.herald import Herald
    from guild.saga import SagaWriter
    from guild.saga.bridge import HeraldBridge

    herald = Herald.instance()
    saga = SagaWriter(base_dir)
    bridge = HeraldBridge(saga, herald)

    # Now any event emitted will be recorded in the Saga
    herald.emit(EventType.LEVEL_UP, {"hero": "DIO", "level": 42})
    # -> Saga records: "[14:23:01] ⬆️ LEVEL UP! DIO reached Level 42"
"""

import logging
from typing import Callable, Dict, Optional, Tuple

from guild.herald.types import Event, EventType
from guild.herald.bus import Herald
from guild.saga.types import get_icon, get_category
from guild.saga.writer import SagaWriter

logger = logging.getLogger(__name__)


# Message templates for each event type
# Format: (icon_override or None, template_string)
# Template has access to all event.data fields
MESSAGE_TEMPLATES: Dict[str, Tuple[Optional[str], str]] = {
    # Quest lifecycle
    "quest.received": (None, "Quest received: {quest_file}"),
    "quest.queued": (None, "Quest queued: {quest_file} (priority: {priority})"),
    "quest.started": (None, "DIO begins quest: {quest_file}"),
    "quest.completed": (None, "Quest complete: {quest_file} (+{xp} XP)"),
    "quest.failed": (None, "Quest failed: {quest_file} - {reason}"),
    "quest.abandoned": (None, "Quest abandoned: {quest_file}"),

    # Combat
    "combat.round": (None, "Round {step} - Damage: {loss:.4f}"),
    "combat.crit": ("💥", "CRIT! Step {step} - Loss dropped to {loss:.4f}"),
    "combat.hit": (None, "Step {step} - Loss: {loss:.4f}"),
    "combat.miss": (None, "Step {step} - High loss: {loss:.4f}"),
    "combat.checkpoint": (None, "Checkpoint saved: {checkpoint}"),

    # Hero progression
    "hero.xp_gained": (None, "+{xp} XP"),
    "hero.level_up": ("⬆️", "LEVEL UP! {hero} reached Level {level}"),
    "hero.skill_level_up": (None, "Skill up! {skill} is now Level {level}"),
    "hero.skill_unlocked": ("🔓", "New skill unlocked: {skill}"),
    "hero.effect_applied": (None, "Effect applied: {effect}"),
    "hero.effect_expired": (None, "Effect expired: {effect}"),

    # Champions
    "champion.evaluated": (None, "Champion evaluated: {checkpoint} (score: {score:.3f})"),
    "champion.crowned": ("🏆", "New champion! {checkpoint}"),
    "champion.deployed": ("🚀", "Champion deployed to {target}: {checkpoint}"),
    "champion.retired": (None, "Champion retired: {checkpoint}"),

    # Training system
    "training.started": ("🏁", "Training daemon started"),
    "training.paused": (None, "Training paused"),
    "training.resumed": (None, "Training resumed"),
    "training.stopped": (None, "Training stopped"),
    "training.idle": ("😴", "DIO rests at the tavern... ({reason})"),

    # System
    "system.healthy": (None, "System healthy"),
    "system.warning": (None, "Warning: {message}"),
    "system.error": ("🔴", "Error: {message}"),
    "system.incident": ("🚨", "Incident: {title}"),
    "system.resolved": (None, "Incident resolved: {title}"),

    # Vault
    "vault.registered": (None, "Asset registered: {asset_id}"),
    "vault.fetched": (None, "Asset fetched: {asset_id}"),
    "vault.cleanup": (None, "Vault cleanup: removed {count} old assets"),

    # Oracle
    "oracle.loaded": ("🔮", "Oracle loaded model: {model}"),
    "oracle.inference": (None, "Inference: {prompt_preview}..."),
    "oracle.error": (None, "Oracle error: {error}"),
}


class HeraldBridge:
    """
    Connects Herald events to the Saga.

    Subscribes to all Herald events and writes them as tales.
    Uses templates to format event data into human-readable messages.
    """

    def __init__(
        self,
        saga: SagaWriter,
        herald: Optional[Herald] = None,
        subscribe_all: bool = True,
    ):
        """
        Initialize the bridge.

        Args:
            saga: SagaWriter to write tales to
            herald: Herald to subscribe to (default: singleton)
            subscribe_all: If True, subscribe to all events immediately
        """
        self.saga = saga
        self.herald = herald or Herald.instance()
        self._subscribed = False

        if subscribe_all:
            self.subscribe_all()

    def subscribe_all(self) -> None:
        """Subscribe to all event types."""
        if self._subscribed:
            return

        # Subscribe to wildcard to catch ALL events
        self.herald.subscribe("*", self._on_event)
        self._subscribed = True
        logger.info("HeraldBridge: subscribed to all events")

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        self.herald.unsubscribe("*", self._on_event)
        self._subscribed = False
        logger.info("HeraldBridge: unsubscribed from all events")

    def _on_event(self, event: Event) -> None:
        """Handle an event from the Herald."""
        try:
            icon, message = self._format_event(event)
            self.saga.tell(
                event_type=event.type,
                message=message,
                icon=icon,
                **event.data,
            )
        except Exception as e:
            logger.warning(f"HeraldBridge: failed to record event {event.type}: {e}")

    def _format_event(self, event: Event) -> Tuple[str, str]:
        """
        Format an event into icon and message.

        Returns:
            Tuple of (icon, formatted_message)
        """
        # Look up template
        template_entry = MESSAGE_TEMPLATES.get(event.type)

        if template_entry:
            icon_override, template = template_entry
            icon = icon_override or get_icon(event.type)

            # Format message with event data
            try:
                message = template.format(**event.data)
            except KeyError as e:
                # Missing data field - use fallback
                logger.debug(f"HeraldBridge: missing field {e} for {event.type}")
                message = self._fallback_message(event)
        else:
            # No template - use generic format
            icon = get_icon(event.type)
            message = self._fallback_message(event)

        return icon, message

    def _fallback_message(self, event: Event) -> str:
        """Generate a fallback message when template fails."""
        # Use 'message' field if present
        if "message" in event.data:
            return event.data["message"]

        # Otherwise, format event type nicely
        event_name = event.type.replace(".", " ").replace("_", " ").title()

        # Add key data fields
        key_fields = ["name", "file", "quest_file", "checkpoint", "skill", "level"]
        extras = []
        for field in key_fields:
            if field in event.data:
                extras.append(f"{field}={event.data[field]}")

        if extras:
            return f"{event_name}: {', '.join(extras)}"

        return event_name


# Module-level bridge singleton
_default_bridge: Optional[HeraldBridge] = None


def init_bridge(saga: SagaWriter, herald: Optional[Herald] = None) -> HeraldBridge:
    """Initialize the default HeraldBridge."""
    global _default_bridge
    _default_bridge = HeraldBridge(saga, herald)
    return _default_bridge


def get_bridge() -> Optional[HeraldBridge]:
    """Get the default HeraldBridge (None if not initialized)."""
    return _default_bridge


def connect_herald_to_saga(
    base_dir,
    herald: Optional[Herald] = None,
) -> Tuple[SagaWriter, HeraldBridge]:
    """
    Convenience function to wire up Herald -> Saga.

    Args:
        base_dir: Base training directory
        herald: Herald instance (default: singleton)

    Returns:
        Tuple of (SagaWriter, HeraldBridge)

    Example:
        from core.paths import get_base_dir
        saga, bridge = connect_herald_to_saga(get_base_dir())
        # Now all Herald events will be recorded in the Saga
    """
    from guild.saga.writer import SagaWriter

    saga = SagaWriter(base_dir)
    bridge = HeraldBridge(saga, herald)
    return saga, bridge
