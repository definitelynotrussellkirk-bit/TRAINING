"""
Global Event System

Central announcement channel for all training system components.

Quick Start:
    from events import emit, announce, get_recent
    from events.types import queue_empty_event, data_generating_event

    # Simple announcement
    announce("Queue is empty!", source="daemon")

    # Typed event
    emit(queue_empty_event(0, 2))

    # Get recent events
    events = get_recent(limit=50)

For SSE streaming in Flask:
    from events import subscribe

    @app.route('/api/events/stream')
    def event_stream():
        def generate():
            for event in subscribe(timeout=30):
                if event is None:
                    yield ": keepalive\n\n"
                else:
                    yield event.to_sse()
        return Response(generate(), mimetype='text/event-stream')
"""

from .types import (
    Event,
    EventType,
    Severity,
    # Event factories
    queue_empty_event,
    data_need_event,
    data_generating_event,
    data_generated_event,
    data_queued_event,
    quality_pass_event,
    quality_fail_event,
    training_started_event,
    training_completed_event,
    checkpoint_saved_event,
    level_up_event,
    daemon_heartbeat_event,
)

from .broadcaster import (
    EventBroadcaster,
    get_broadcaster,
    emit,
    subscribe,
    get_recent,
    on,
    on_all,
    announce,
)

__all__ = [
    # Types
    "Event",
    "EventType",
    "Severity",
    # Event factories
    "queue_empty_event",
    "data_need_event",
    "data_generating_event",
    "data_generated_event",
    "data_queued_event",
    "quality_pass_event",
    "quality_fail_event",
    "training_started_event",
    "training_completed_event",
    "checkpoint_saved_event",
    "level_up_event",
    "daemon_heartbeat_event",
    # Broadcaster
    "EventBroadcaster",
    "get_broadcaster",
    "emit",
    "subscribe",
    "get_recent",
    "on",
    "on_all",
    "announce",
]
