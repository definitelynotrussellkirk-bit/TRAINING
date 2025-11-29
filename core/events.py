"""
Events System - Central event stream for the Realm.

All significant events are logged to status/events.jsonl for:
- Battle log display
- Debugging
- Audit trail

Usage:
    from core.events import emit_event, get_recent_events

    # Emit an event
    emit_event("job_started", job_id="xyz", job_type="train", worker_id="training_daemon")

    # Read recent events
    events = get_recent_events(limit=50)
    for e in events:
        print(f"[{e['ts']}] {e['kind']}: {e.get('job_id', '')}")

Event Kinds:
    - job_submitted, job_started, job_completed, job_failed
    - checkpoint_saved
    - eval_suite_triggered, eval_completed
    - mode_changed
    - worker_started, worker_stopped, worker_stale
    - reset_performed
    - warning_raised
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Iterator

logger = logging.getLogger(__name__)

_events_lock = Lock()


def _get_events_file() -> Path:
    """Get path to events file."""
    try:
        from core.paths import get_base_dir
        base_dir = get_base_dir()
    except ImportError:
        base_dir = Path(__file__).parent.parent
    return base_dir / "status" / "events.jsonl"


# =============================================================================
# EVENT EMISSION
# =============================================================================

def emit_event(kind: str, **fields) -> Dict[str, Any]:
    """
    Emit an event to the central event stream.

    Args:
        kind: Event kind (e.g., "job_started", "checkpoint_saved")
        **fields: Additional event fields

    Returns:
        The event dict that was written

    Common event kinds:
        job_submitted: job_id, job_type, payload
        job_started: job_id, job_type, worker_id
        job_completed: job_id, job_type, worker_id, result
        job_failed: job_id, job_type, worker_id, error
        checkpoint_saved: hero_id, campaign_id, step, path
        eval_suite_triggered: suite_id, run_id, jobs_count
        eval_completed: suite_id, skill_id, level, accuracy
        mode_changed: from_mode, to_mode, changed_by, reason
        worker_started: worker_id, role, device
        worker_stopped: worker_id, role
        worker_stale: worker_id, role, last_seen
        reset_performed: jobs_cancelled, reason
        warning_raised: warning_type, message, details
    """
    event = {
        "ts": datetime.now().isoformat(),
        "kind": kind,
    }
    event.update(fields)

    events_file = _get_events_file()

    try:
        with _events_lock:
            events_file.parent.mkdir(parents=True, exist_ok=True)
            with open(events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to emit event {kind}: {e}")

    return event


# =============================================================================
# CONVENIENCE EMITTERS
# =============================================================================

def emit_job_submitted(job_id: str, job_type: str, payload: Optional[Dict] = None):
    """Emit job_submitted event."""
    emit_event("job_submitted", job_id=job_id, job_type=job_type, payload=payload or {})


def emit_job_started(job_id: str, job_type: str, worker_id: str):
    """Emit job_started event."""
    emit_event("job_started", job_id=job_id, job_type=job_type, worker_id=worker_id)


def emit_job_completed(job_id: str, job_type: str, worker_id: str, result: Optional[Dict] = None):
    """Emit job_completed event."""
    emit_event("job_completed", job_id=job_id, job_type=job_type, worker_id=worker_id, result=result or {})


def emit_job_failed(job_id: str, job_type: str, worker_id: str, error: str):
    """Emit job_failed event."""
    emit_event("job_failed", job_id=job_id, job_type=job_type, worker_id=worker_id, error=error)


def emit_checkpoint_saved(hero_id: str, campaign_id: str, step: int, path: str):
    """Emit checkpoint_saved event."""
    emit_event("checkpoint_saved", hero_id=hero_id, campaign_id=campaign_id, step=step, path=path)


def emit_training_started(job_id: str, job_name: str, total_steps: int):
    """Emit training_started event."""
    emit_event("training_started", job_id=job_id, job_name=job_name, total_steps=total_steps)


def emit_training_completed(job_id: str, job_name: str, final_step: int, final_loss: Optional[float] = None):
    """Emit training_completed event."""
    emit_event("training_completed", job_id=job_id, job_name=job_name, final_step=final_step, final_loss=final_loss)


def emit_eval_suite_triggered(suite_id: str, run_id: str, jobs_count: int):
    """Emit eval_suite_triggered event."""
    emit_event("eval_suite_triggered", suite_id=suite_id, run_id=run_id, jobs_count=jobs_count)


def emit_eval_completed(suite_id: str, skill_id: str, level: int, accuracy: float):
    """Emit eval_completed event."""
    emit_event("eval_completed", suite_id=suite_id, skill_id=skill_id, level=level, accuracy=accuracy)


def emit_warning(warning_type: str, message: str, details: Optional[Dict] = None):
    """Emit warning_raised event."""
    emit_event("warning_raised", warning_type=warning_type, message=message, details=details or {})


# =============================================================================
# EVENT READING
# =============================================================================

def iter_events(
    since: Optional[datetime] = None,
    kinds: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over events from the event file.

    Args:
        since: Only return events after this timestamp
        kinds: Only return events of these kinds

    Yields:
        Event dicts
    """
    events_file = _get_events_file()

    if not events_file.exists():
        return

    try:
        with open(events_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by timestamp
                if since:
                    try:
                        event_ts = datetime.fromisoformat(event.get("ts", ""))
                        if event_ts < since:
                            continue
                    except (ValueError, TypeError):
                        continue

                # Filter by kind
                if kinds and event.get("kind") not in kinds:
                    continue

                yield event
    except Exception as e:
        logger.error(f"Failed to read events: {e}")


def get_recent_events(
    limit: int = 50,
    since: Optional[datetime] = None,
    kinds: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent events, newest first.

    Args:
        limit: Maximum number of events to return
        since: Only return events after this timestamp
        kinds: Only return events of these kinds

    Returns:
        List of event dicts, newest first
    """
    events_file = _get_events_file()

    if not events_file.exists():
        return []

    # Read all matching events, then take the last N
    all_events = list(iter_events(since=since, kinds=kinds))
    return list(reversed(all_events[-limit:]))


def get_events_since(timestamp: datetime) -> List[Dict[str, Any]]:
    """Get all events since a timestamp."""
    return list(iter_events(since=timestamp))


def get_events_last_n_minutes(minutes: int = 30) -> List[Dict[str, Any]]:
    """Get events from the last N minutes."""
    since = datetime.now() - timedelta(minutes=minutes)
    return get_recent_events(limit=1000, since=since)


# =============================================================================
# EVENT FILE MANAGEMENT
# =============================================================================

def get_event_count() -> int:
    """Get total number of events in the file."""
    events_file = _get_events_file()
    if not events_file.exists():
        return 0

    try:
        with open(events_file, "r") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def rotate_events(max_events: int = 10000):
    """
    Rotate events file if it gets too large.

    Keeps the most recent max_events events.
    """
    events_file = _get_events_file()

    if not events_file.exists():
        return

    count = get_event_count()
    if count <= max_events:
        return

    logger.info(f"Rotating events file ({count} events -> {max_events})")

    # Read all events
    all_events = list(iter_events())

    # Keep only the most recent
    keep_events = all_events[-max_events:]

    # Write back
    with _events_lock:
        # Backup old file
        backup_file = events_file.with_suffix(".jsonl.bak")
        if backup_file.exists():
            backup_file.unlink()
        events_file.rename(backup_file)

        # Write new file
        with open(events_file, "w") as f:
            for event in keep_events:
                f.write(json.dumps(event) + "\n")

    logger.info(f"Events file rotated: {len(keep_events)} events kept")


def clear_events():
    """Clear all events (use with caution)."""
    events_file = _get_events_file()
    with _events_lock:
        if events_file.exists():
            events_file.unlink()


# =============================================================================
# BATTLE LOG FORMATTING
# =============================================================================

def format_event_for_battle_log(event: Dict[str, Any]) -> str:
    """
    Format an event for display in the battle log.

    Returns a human-readable string.
    """
    kind = event.get("kind", "unknown")
    ts = event.get("ts", "")[:19]  # Trim to seconds

    formatters = {
        "job_submitted": lambda e: f"Job submitted: {e.get('job_type', '?')} {e.get('job_id', '?')[:8]}",
        "job_started": lambda e: f"Job started: {e.get('job_type', '?')} {e.get('job_id', '?')[:8]} on {e.get('worker_id', '?')}",
        "job_completed": lambda e: f"Job completed: {e.get('job_type', '?')} {e.get('job_id', '?')[:8]}",
        "job_failed": lambda e: f"Job FAILED: {e.get('job_type', '?')} {e.get('job_id', '?')[:8]} - {e.get('error', '?')[:50]}",
        "training_started": lambda e: f"Training started: {e.get('job_name', '?')} ({e.get('total_steps', '?')} steps)",
        "training_completed": lambda e: f"Training completed: {e.get('job_name', '?')} (step {e.get('final_step', '?')}, loss {e.get('final_loss', '?')})",
        "checkpoint_saved": lambda e: f"Checkpoint saved: step {e.get('step', '?')}",
        "eval_suite_triggered": lambda e: f"Eval suite triggered: {e.get('suite_id', '?')} ({e.get('jobs_count', '?')} jobs)",
        "eval_completed": lambda e: f"Eval completed: {e.get('skill_id', '?')} L{e.get('level', '?')} = {e.get('accuracy', 0)*100:.1f}%",
        "mode_changed": lambda e: f"Mode changed: {e.get('from_mode', '?')} -> {e.get('to_mode', '?')} ({e.get('reason', '')})",
        "worker_started": lambda e: f"Worker started: {e.get('worker_id', '?')} ({e.get('role', '?')})",
        "worker_stopped": lambda e: f"Worker stopped: {e.get('worker_id', '?')}",
        "worker_stale": lambda e: f"Worker STALE: {e.get('worker_id', '?')} (last seen {e.get('last_seen', '?')})",
        "reset_performed": lambda e: f"Reset performed: {e.get('jobs_cancelled', 0)} jobs cancelled",
        "warning_raised": lambda e: f"Warning: {e.get('warning_type', '?')} - {e.get('message', '?')}",
    }

    formatter = formatters.get(kind, lambda e: f"{kind}: {json.dumps(e)[:60]}")
    return f"[{ts}] {formatter(event)}"


def get_battle_log(limit: int = 50, since_minutes: Optional[int] = None) -> List[str]:
    """
    Get formatted battle log entries.

    Args:
        limit: Maximum entries
        since_minutes: Only show events from last N minutes

    Returns:
        List of formatted strings for display
    """
    since = None
    if since_minutes:
        since = datetime.now() - timedelta(minutes=since_minutes)

    events = get_recent_events(limit=limit, since=since)
    return [format_event_for_battle_log(e) for e in events]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Events System")
    parser.add_argument("--list", action="store_true", help="List recent events")
    parser.add_argument("--battle-log", action="store_true", help="Show formatted battle log")
    parser.add_argument("--count", action="store_true", help="Show event count")
    parser.add_argument("--rotate", action="store_true", help="Rotate events file")
    parser.add_argument("--emit", type=str, help="Emit a test event of given kind")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to show")
    parser.add_argument("--since-minutes", type=int, help="Only show events from last N minutes")

    args = parser.parse_args()

    if args.emit:
        event = emit_event(args.emit, test=True, message="Test event from CLI")
        print(f"Emitted: {json.dumps(event)}")

    elif args.count:
        count = get_event_count()
        print(f"Total events: {count}")

    elif args.rotate:
        rotate_events()
        print("Events rotated")

    elif args.battle_log:
        log = get_battle_log(limit=args.limit, since_minutes=args.since_minutes)
        if not log:
            print("No events")
        else:
            for line in log:
                print(line)

    else:
        events = get_recent_events(limit=args.limit)
        if not events:
            print("No events")
        else:
            print(f"Recent Events ({len(events)}):")
            print("-" * 60)
            for e in events:
                print(f"[{e.get('ts', '')[:19]}] {e.get('kind', '?')}")
                for k, v in e.items():
                    if k not in ("ts", "kind"):
                        print(f"  {k}: {v}")
                print()
