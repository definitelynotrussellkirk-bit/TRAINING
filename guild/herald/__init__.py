"""
Herald - The event bus of the realm.

The Herald announces events throughout the realm. Any system can emit events,
and any system can subscribe to receive them.

Quick Start:
    from guild.herald import Herald, EventType, emit, subscribe

    # Subscribe to events
    def on_level_up(event):
        print(f"Level up! {event.data}")

    subscribe(EventType.LEVEL_UP, on_level_up)

    # Emit events (from anywhere in the codebase)
    emit(EventType.LEVEL_UP, {"hero": "DIO", "level": 42})

    # Wildcard subscriptions
    subscribe("quest.*", on_any_quest_event)  # All quest events
    subscribe("*", on_any_event)              # ALL events

Singleton Access:
    herald = Herald.instance()
    herald.emit(EventType.QUEST_STARTED, {"quest_file": "data.jsonl"})

Event Types:
    See EventType enum for all available event types.
    Custom strings also supported: emit("my.custom.event", data)
"""

from guild.herald.types import Event, EventType
from guild.herald.bus import (
    Herald,
    get_herald,
    emit,
    subscribe,
    unsubscribe,
)

__all__ = [
    # Types
    "Event",
    "EventType",
    # Herald class
    "Herald",
    # Convenience functions
    "get_herald",
    "emit",
    "subscribe",
    "unsubscribe",
]
