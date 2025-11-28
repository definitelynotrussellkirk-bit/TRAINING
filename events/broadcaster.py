"""
Global Event Broadcaster

Central hub for all system events. Components emit events here,
and subscribers (like Tavern UI) receive them in real-time.

Architecture:
- Singleton broadcaster (get_broadcaster())
- Thread-safe event queue
- File-backed persistence (events.jsonl)
- In-memory buffer for recent events
- SSE-compatible output format

Usage:
    from events import emit, subscribe, get_recent

    # Emit an event
    emit(queue_empty_event(0, 2))

    # Get recent events
    events = get_recent(limit=50)

    # Subscribe to events (for SSE streaming)
    for event in subscribe():
        yield event.to_sse()
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Generator, List, Optional, Set

from .types import Event, EventType, Severity

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """
    Central event hub for the training system.

    Thread-safe singleton that:
    - Accepts events from any component
    - Stores recent events in memory
    - Persists to events.jsonl
    - Streams to SSE subscribers
    """

    _instance: Optional["EventBroadcaster"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, base_dir: Optional[str] = None, max_buffer: int = 500):
        if self._initialized:
            return

        self._initialized = True

        # Find base directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Auto-detect
            self.base_dir = Path(__file__).parent.parent

        # Event storage
        self.events_dir = self.base_dir / "status"
        self.events_dir.mkdir(exist_ok=True)
        self.events_file = self.events_dir / "events.jsonl"

        # In-memory buffer (recent events)
        self.buffer: deque = deque(maxlen=max_buffer)
        self.max_buffer = max_buffer

        # Subscriber queues (for SSE streaming)
        self.subscribers: List[Queue] = []
        self.subscriber_lock = threading.Lock()

        # Event handlers (callbacks)
        self.handlers: dict[EventType, List[Callable]] = {}
        self.global_handlers: List[Callable] = []

        # Thread safety
        self.emit_lock = threading.Lock()

        # Load recent events from file
        self._load_recent_events()

        logger.info(f"EventBroadcaster initialized (buffer: {max_buffer}, file: {self.events_file})")

    def _load_recent_events(self):
        """Load recent events from file into buffer."""
        if not self.events_file.exists():
            return

        try:
            with open(self.events_file, "r") as f:
                lines = f.readlines()
                # Only load last max_buffer lines
                recent_lines = lines[-self.max_buffer:]
                for line in recent_lines:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            event = Event.from_dict(data)
                            self.buffer.append(event)
                        except (json.JSONDecodeError, KeyError):
                            pass  # Skip malformed lines
            logger.debug(f"Loaded {len(self.buffer)} events from {self.events_file}")
        except Exception as e:
            logger.warning(f"Failed to load events: {e}")

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers and handlers.

        Thread-safe. Can be called from any component.
        """
        with self.emit_lock:
            # Add to buffer
            self.buffer.append(event)

            # Persist to file
            self._persist_event(event)

            # Notify subscribers (SSE)
            self._notify_subscribers(event)

            # Call handlers
            self._call_handlers(event)

            # Also log
            log_level = {
                Severity.DEBUG: logging.DEBUG,
                Severity.INFO: logging.INFO,
                Severity.WARNING: logging.WARNING,
                Severity.ERROR: logging.ERROR,
                Severity.SUCCESS: logging.INFO,
            }.get(event.severity, logging.INFO)

            logger.log(log_level, f"[{event.source}] {event.message}")

    def _persist_event(self, event: Event) -> None:
        """Append event to events.jsonl file."""
        try:
            with open(self.events_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def _notify_subscribers(self, event: Event) -> None:
        """Push event to all SSE subscriber queues."""
        with self.subscriber_lock:
            dead_subscribers = []
            for q in self.subscribers:
                try:
                    q.put_nowait(event)
                except Exception:
                    dead_subscribers.append(q)

            # Clean up dead subscribers
            for q in dead_subscribers:
                self.subscribers.remove(q)

    def _call_handlers(self, event: Event) -> None:
        """Call registered handlers for this event type."""
        # Type-specific handlers
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event.type}: {e}")

        # Global handlers
        for handler in self.global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")

    def subscribe(self, timeout: float = 30.0) -> Generator[Event, None, None]:
        """
        Subscribe to event stream (for SSE).

        Yields events as they arrive. Times out after `timeout` seconds
        of inactivity, yielding None to keep connection alive.
        """
        q: Queue = Queue()

        with self.subscriber_lock:
            self.subscribers.append(q)

        try:
            while True:
                try:
                    event = q.get(timeout=timeout)
                    yield event
                except Empty:
                    # Timeout - yield None for keepalive
                    yield None
        finally:
            # Cleanup on disconnect
            with self.subscriber_lock:
                if q in self.subscribers:
                    self.subscribers.remove(q)

    def on(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def on_all(self, handler: Callable[[Event], None]) -> None:
        """Register a handler for all events."""
        self.global_handlers.append(handler)

    def get_recent(self, limit: int = 50, event_type: Optional[EventType] = None,
                   severity: Optional[Severity] = None) -> List[Event]:
        """Get recent events from buffer, optionally filtered."""
        events = list(self.buffer)

        if event_type:
            events = [e for e in events if e.type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        # Return most recent first
        return list(reversed(events[-limit:]))

    def get_since(self, since_id: str) -> List[Event]:
        """Get all events since a specific event ID."""
        events = list(self.buffer)
        result = []
        found = False

        for event in events:
            if found:
                result.append(event)
            elif event.id == since_id:
                found = True

        return result

    def clear(self) -> None:
        """Clear all events (for testing)."""
        with self.emit_lock:
            self.buffer.clear()
            # Truncate file
            with open(self.events_file, "w") as f:
                pass

    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        with self.subscriber_lock:
            return len(self.subscribers)

    def stats(self) -> dict:
        """Get broadcaster statistics."""
        return {
            "buffer_size": len(self.buffer),
            "max_buffer": self.max_buffer,
            "subscribers": self.subscriber_count(),
            "events_file": str(self.events_file),
            "handlers": {str(k): len(v) for k, v in self.handlers.items()},
            "global_handlers": len(self.global_handlers),
        }


# Singleton accessor
_broadcaster: Optional[EventBroadcaster] = None


def get_broadcaster(base_dir: Optional[str] = None) -> EventBroadcaster:
    """Get the global EventBroadcaster singleton."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = EventBroadcaster(base_dir=base_dir)
    return _broadcaster


# Convenience functions
def emit(event: Event) -> None:
    """Emit an event to the global broadcaster."""
    get_broadcaster().emit(event)


def subscribe(timeout: float = 30.0) -> Generator[Event, None, None]:
    """Subscribe to global event stream."""
    return get_broadcaster().subscribe(timeout)


def get_recent(limit: int = 50, event_type: Optional[EventType] = None) -> List[Event]:
    """Get recent events from global broadcaster."""
    return get_broadcaster().get_recent(limit=limit, event_type=event_type)


def on(event_type: EventType, handler: Callable[[Event], None]) -> None:
    """Register a handler for an event type."""
    get_broadcaster().on(event_type, handler)


def on_all(handler: Callable[[Event], None]) -> None:
    """Register a handler for all events."""
    get_broadcaster().on_all(handler)


# Announcement helper - high-level API for components
def announce(message: str, event_type: EventType = EventType.DAEMON_HEARTBEAT,
             severity: Severity = Severity.INFO, source: str = "system",
             data: Optional[dict] = None) -> None:
    """
    Announce something to the global channel.

    Convenience function for simple announcements.
    """
    event = Event(
        type=event_type,
        message=message,
        severity=severity,
        source=source,
        data=data or {},
    )
    emit(event)
