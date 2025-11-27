"""
Herald - The central event bus of the realm.

The Herald announces events to all who listen. Any system can emit events,
and any system can subscribe to receive them.

Usage:
    from guild.herald import Herald, EventType

    # Get the herald (singleton)
    herald = Herald.instance()

    # Subscribe to events
    def on_level_up(event):
        print(f"Level up! New level: {event.data['level']}")

    herald.subscribe(EventType.LEVEL_UP, on_level_up)

    # Or subscribe to all events in a category
    herald.subscribe("quest.*", on_any_quest_event)

    # Emit events (from anywhere)
    herald.emit(EventType.LEVEL_UP, {"hero": "DIO", "level": 42})

Thread Safety:
    Herald is thread-safe. Events can be emitted from any thread,
    and callbacks will be invoked synchronously in the emitting thread.
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set
from weakref import WeakSet

from guild.herald.types import Event, EventType

logger = logging.getLogger(__name__)


class Herald:
    """
    Central event bus for the realm.

    Singleton pattern - use Herald.instance() to get the shared instance.

    Features:
        - Type-safe event emission
        - Wildcard subscriptions (e.g., "quest.*")
        - Thread-safe
        - Callback error isolation (one bad callback doesn't break others)
    """

    _instance: Optional["Herald"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize Herald. Use Herald.instance() instead of direct construction."""
        # Map of event pattern -> list of callbacks
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        # Lock for thread safety
        self._sub_lock = threading.RLock()
        # Track event history (limited buffer for debugging)
        self._history: List[Event] = []
        self._history_limit = 100
        self._history_lock = threading.Lock()
        # Paused state (for testing/debugging)
        self._paused = False

    @classmethod
    def instance(cls) -> "Herald":
        """Get the singleton Herald instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def subscribe(
        self,
        event_type: EventType | str,
        callback: Callable[[Event], None],
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: EventType enum or string pattern.
                        Supports wildcards: "quest.*" matches all quest events.
            callback: Function to call when event is emitted.
                     Signature: callback(event: Event) -> None

        Example:
            herald.subscribe(EventType.LEVEL_UP, my_callback)
            herald.subscribe("quest.*", on_any_quest)
        """
        pattern = event_type.value if isinstance(event_type, EventType) else str(event_type)

        with self._sub_lock:
            if callback not in self._subscribers[pattern]:
                self._subscribers[pattern].append(callback)
                logger.debug(f"Herald: subscribed to '{pattern}'")

    def unsubscribe(
        self,
        event_type: EventType | str,
        callback: Callable[[Event], None],
    ) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: EventType enum or string pattern
            callback: The callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        pattern = event_type.value if isinstance(event_type, EventType) else str(event_type)

        with self._sub_lock:
            if pattern in self._subscribers and callback in self._subscribers[pattern]:
                self._subscribers[pattern].remove(callback)
                logger.debug(f"Herald: unsubscribed from '{pattern}'")
                return True
        return False

    def emit(
        self,
        event_type: EventType | str,
        data: Optional[Dict] = None,
        source: Optional[str] = None,
    ) -> Event:
        """
        Emit an event to all subscribers.

        Args:
            event_type: The type of event
            data: Event payload (optional)
            source: Identifier of the event source (optional)

        Returns:
            The Event object that was emitted

        Example:
            herald.emit(EventType.LEVEL_UP, {"hero": "DIO", "level": 42})
            herald.emit("custom.event", {"custom": "data"}, source="my_module")
        """
        if self._paused:
            logger.debug(f"Herald paused, dropping event: {event_type}")
            return None

        event = Event(
            type=event_type,
            data=data or {},
            timestamp=datetime.now(),
            source=source,
        )

        # Add to history
        self._add_to_history(event)

        # Get all matching callbacks
        callbacks = self._get_matching_callbacks(event.type)

        # Invoke callbacks (errors isolated)
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(
                    f"Herald: callback error for '{event.type}': {e}",
                    exc_info=True
                )

        logger.debug(f"Herald: emitted '{event.type}' to {len(callbacks)} subscribers")
        return event

    def _get_matching_callbacks(self, event_type: str) -> List[Callable]:
        """Get all callbacks that match the event type."""
        callbacks = []

        with self._sub_lock:
            # Direct match
            if event_type in self._subscribers:
                callbacks.extend(self._subscribers[event_type])

            # Wildcard matches (e.g., "quest.*" matches "quest.started")
            category = event_type.split('.')[0] if '.' in event_type else event_type
            wildcard = f"{category}.*"
            if wildcard in self._subscribers:
                callbacks.extend(self._subscribers[wildcard])

            # Global wildcard
            if "*" in self._subscribers:
                callbacks.extend(self._subscribers["*"])

        return callbacks

    def _add_to_history(self, event: Event):
        """Add event to history buffer."""
        with self._history_lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit:]

    def recent_events(self, limit: int = 20) -> List[Event]:
        """Get recent events from history."""
        with self._history_lock:
            return list(self._history[-limit:])

    def pause(self):
        """Pause event emission (for testing/debugging)."""
        self._paused = True
        logger.info("Herald: paused")

    def resume(self):
        """Resume event emission."""
        self._paused = False
        logger.info("Herald: resumed")

    def clear_subscribers(self):
        """Remove all subscribers (for testing)."""
        with self._sub_lock:
            self._subscribers.clear()
        logger.debug("Herald: cleared all subscribers")

    @property
    def subscriber_count(self) -> int:
        """Total number of subscriptions."""
        with self._sub_lock:
            return sum(len(cbs) for cbs in self._subscribers.values())

    def get_subscription_patterns(self) -> List[str]:
        """Get all patterns with active subscriptions."""
        with self._sub_lock:
            return [p for p, cbs in self._subscribers.items() if cbs]


# Convenience functions for module-level access
_default_herald: Optional[Herald] = None


def get_herald() -> Herald:
    """Get the default Herald instance."""
    global _default_herald
    if _default_herald is None:
        _default_herald = Herald.instance()
    return _default_herald


def emit(
    event_type: EventType | str,
    data: Optional[Dict] = None,
    source: Optional[str] = None,
) -> Event:
    """Emit an event using the default Herald."""
    return get_herald().emit(event_type, data, source)


def subscribe(
    event_type: EventType | str,
    callback: Callable[[Event], None],
) -> None:
    """Subscribe to events using the default Herald."""
    get_herald().subscribe(event_type, callback)


def unsubscribe(
    event_type: EventType | str,
    callback: Callable[[Event], None],
) -> bool:
    """Unsubscribe from events using the default Herald."""
    return get_herald().unsubscribe(event_type, callback)
