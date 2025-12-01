"""
Events System - LEGACY COMPATIBILITY SHIM

⚠️  DEPRECATED: This module is retained for backward compatibility only.
⚠️  New code should use core.event_bus instead.

This module now delegates all calls to core.event_bus. The JSONL storage
has been retired in favor of dual-write to RealmState (live) and BattleLogger (persistent).

Migration Guide:
    OLD:
        from core.events import emit_job_started
        emit_job_started(job_id, job_type, worker_id)

    NEW:
        from core.event_bus import job_started
        job_started(job_id, job_type, worker_id)

For consumption:
    OLD:
        from core.events import get_recent_events
        events = get_recent_events(limit=50)

    NEW (live view):
        from core.realm_store import get_events
        events = get_events(limit=50)

    NEW (historic view):
        from core.battle_log import get_battle_logger
        logger = get_battle_logger()
        events = logger.get_events(limit=100)
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

# Emit deprecation warning on import
warnings.warn(
    "core.events is deprecated. Use core.event_bus for emission "
    "and core.realm_store or core.battle_log for consumption. "
    "JSONL storage has been retired.",
    DeprecationWarning,
    stacklevel=2,
)


# =============================================================================
# EVENT EMISSION (delegates to event_bus)
# =============================================================================

def emit_event(kind: str, **fields) -> Dict[str, Any]:
    """
    Emit an event (legacy interface).

    ⚠️ DEPRECATED: Use core.event_bus.emit_realm_event() instead.

    This function now delegates to the unified event bus.
    """
    from core.event_bus import emit_realm_event

    # Extract optional channel/severity if provided
    channel = fields.pop("channel", None)
    severity = fields.pop("severity", None)

    # Build human-readable message from kind and fields
    message = _format_message_from_fields(kind, fields)

    return emit_realm_event(
        kind=kind,
        message=message,
        channel=channel,
        severity=severity,
        details=fields,
    )


def _format_message_from_fields(kind: str, fields: Dict) -> str:
    """Format a message from legacy event fields."""
    import json

    formatters = {
        "job_submitted": lambda f: f"Job {f.get('job_type', '?')} submitted: {f.get('job_id', '?')[:8]}",
        "job_started": lambda f: f"Job {f.get('job_type', '?')} started on {f.get('worker_id', '?')}",
        "job_completed": lambda f: f"Job {f.get('job_type', '?')} completed: {f.get('job_id', '?')[:8]}",
        "job_failed": lambda f: f"Job {f.get('job_type', '?')} FAILED: {f.get('job_id', '?')[:8]} - {f.get('error', '?')[:50]}",
        "training_started": lambda f: f"Training started: {f.get('job_name', '?')} ({f.get('total_steps', '?')} steps)",
        "training_completed": lambda f: f"Training completed: {f.get('job_name', '?')} (step {f.get('final_step', '?')}, loss {f.get('final_loss', '?')})",
        "checkpoint_saved": lambda f: f"Checkpoint {f.get('step', '?'):,} saved",
        "eval_suite_triggered": lambda f: f"Eval suite {f.get('suite_id', '?')} triggered ({f.get('jobs_count', '?')} jobs)",
        "eval_completed": lambda f: f"Eval {f.get('skill_id', '?')} L{f.get('level', '?')}: {f.get('accuracy', 0)*100:.1f}%",
        "mode_changed": lambda f: f"Mode changed: {f.get('from_mode', '?')} → {f.get('to_mode', '?')}",
        "worker_started": lambda f: f"Worker {f.get('worker_id', '?')} started ({f.get('role', '?')})",
        "worker_stopped": lambda f: f"Worker {f.get('worker_id', '?')} stopped",
        "worker_stale": lambda f: f"Worker {f.get('worker_id', '?')} STALE (last seen {f.get('last_seen', '?')})",
        "reset_performed": lambda f: f"Reset performed: {f.get('jobs_cancelled', 0)} jobs cancelled",
        "warning_raised": lambda f: f"Warning: {f.get('warning_type', '?')} - {f.get('message', '?')}",
    }

    formatter = formatters.get(kind, lambda f: f"{kind}: {json.dumps(f)[:60]}")
    return formatter(fields)


# =============================================================================
# TYPED EMITTERS (delegate to event_bus typed helpers)
# =============================================================================

def emit_job_submitted(job_id: str, job_type: str, payload: Optional[Dict] = None):
    """⚠️ DEPRECATED: Use core.event_bus.job_submitted()"""
    from core.event_bus import job_submitted
    return job_submitted(job_id, job_type, payload)


def emit_job_started(job_id: str, job_type: str, worker_id: str):
    """⚠️ DEPRECATED: Use core.event_bus.job_started()"""
    from core.event_bus import job_started
    return job_started(job_id, job_type, worker_id)


def emit_job_completed(job_id: str, job_type: str, worker_id: str, result: Optional[Dict] = None):
    """⚠️ DEPRECATED: Use core.event_bus.job_completed()"""
    from core.event_bus import job_completed
    # Note: event_bus.job_completed expects duration_sec, not result
    # For compat, we'll use emit_realm_event directly
    from core.event_bus import emit_realm_event
    return emit_realm_event(
        kind="job_completed",
        message=f"Job {job_type} completed",
        details={"job_id": job_id, "job_type": job_type, "worker_id": worker_id, "result": result or {}},
    )


def emit_job_failed(job_id: str, job_type: str, worker_id: str, error: str):
    """⚠️ DEPRECATED: Use core.event_bus.job_failed()"""
    from core.event_bus import job_failed
    return job_failed(job_id, job_type, error, worker_id)


def emit_checkpoint_saved(hero_id: str, campaign_id: str, step: int, path: str):
    """⚠️ DEPRECATED: Use core.event_bus.checkpoint_saved()"""
    from core.event_bus import checkpoint_saved
    # Note: event_bus.checkpoint_saved requires loss, which we don't have here
    # Use emit_realm_event with full details
    from core.event_bus import emit_realm_event
    return emit_realm_event(
        kind="checkpoint_saved",
        message=f"Checkpoint {step:,} saved",
        hero_id=hero_id,
        campaign_id=campaign_id,
        details={"step": step, "path": path},
    )


def emit_training_started(job_id: str, job_name: str, total_steps: int):
    """⚠️ DEPRECATED: Use core.event_bus.training_started()"""
    from core.event_bus import training_started
    return training_started(job_id, job_name, total_steps)


def emit_training_completed(job_id: str, job_name: str, final_step: int, final_loss: Optional[float] = None):
    """⚠️ DEPRECATED: Use core.event_bus.training_completed()"""
    from core.event_bus import training_completed
    return training_completed(job_id, job_name, final_step, final_loss or 0.0)


def emit_eval_suite_triggered(suite_id: str, run_id: str, jobs_count: int):
    """⚠️ DEPRECATED: Use core.event_bus.emit_realm_event()"""
    from core.event_bus import emit_realm_event
    return emit_realm_event(
        kind="eval_suite_triggered",
        message=f"Eval suite {suite_id} triggered ({jobs_count} jobs)",
        details={"suite_id": suite_id, "run_id": run_id, "jobs_count": jobs_count},
    )


def emit_eval_completed(suite_id: str, skill_id: str, level: int, accuracy: float):
    """⚠️ DEPRECATED: Use core.event_bus.eval_completed()"""
    from core.event_bus import eval_completed
    # Note: event_bus.eval_completed has different signature (skill, not suite_id)
    return eval_completed(skill_id, level, accuracy)


def emit_warning(warning_type: str, message: str, details: Optional[Dict] = None):
    """⚠️ DEPRECATED: Use core.event_bus.warning_raised()"""
    from core.event_bus import warning_raised
    return warning_raised(warning_type, message, details)


# =============================================================================
# EVENT READING (delegates to realm_store or battle_log)
# =============================================================================

def get_recent_events(
    limit: int = 50,
    since: Optional[datetime] = None,
    kinds: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent events (legacy interface).

    ⚠️ DEPRECATED: Use core.realm_store.get_events() for live view
                  or core.battle_log for historic queries.

    This now reads from RealmState (last N events, in-memory).
    For historic queries with filtering, use BattleLogger directly.
    """
    from core.realm_store import get_events

    # RealmState doesn't support since/kinds filtering, so we filter post-fetch
    events = get_events(limit=limit * 2)  # Fetch more to account for filtering

    filtered = []
    for event in events:
        # Filter by timestamp if provided
        if since:
            try:
                event_ts = datetime.fromisoformat(event.get("timestamp", ""))
                if event_ts < since:
                    continue
            except (ValueError, TypeError):
                continue

        # Filter by kind if provided
        if kinds and event.get("kind") not in kinds:
            continue

        filtered.append({
            "ts": event.get("timestamp", ""),
            "kind": event.get("kind", ""),
            **event.get("details", {}),
        })

        if len(filtered) >= limit:
            break

    return filtered


def get_events_since(timestamp: datetime) -> List[Dict[str, Any]]:
    """⚠️ DEPRECATED: Use core.realm_store.get_events()"""
    return get_recent_events(limit=1000, since=timestamp)


def get_events_last_n_minutes(minutes: int = 30) -> List[Dict[str, Any]]:
    """⚠️ DEPRECATED: Use core.battle_log for time-based queries"""
    from datetime import timedelta
    since = datetime.now() - timedelta(minutes=minutes)
    return get_recent_events(limit=1000, since=since)


# =============================================================================
# LEGACY JSONL FUNCTIONS (no-ops or raise errors)
# =============================================================================

def get_event_count() -> int:
    """
    ⚠️ DEPRECATED: JSONL storage removed. Use BattleLogger for counts.

    Returns 0 for backward compatibility.
    """
    warnings.warn("JSONL storage removed. Use core.battle_log for event counts.", DeprecationWarning)
    return 0


def rotate_events(max_events: int = 10000):
    """
    ⚠️ DEPRECATED: JSONL storage removed. No-op.
    """
    warnings.warn("JSONL storage removed. This is a no-op.", DeprecationWarning)
    pass


def clear_events():
    """
    ⚠️ DEPRECATED: JSONL storage removed. No-op.
    """
    warnings.warn("JSONL storage removed. This is a no-op.", DeprecationWarning)
    pass


def iter_events(*args, **kwargs):
    """
    ⚠️ DEPRECATED: JSONL storage removed. Use BattleLogger.get_events().
    """
    raise DeprecationWarning(
        "iter_events() removed with JSONL storage. "
        "Use core.battle_log.BattleLogger.get_events() for historic queries."
    )


# =============================================================================
# BATTLE LOG FORMATTING (moved to event_bus)
# =============================================================================

def format_event_for_battle_log(event: Dict[str, Any]) -> str:
    """
    ⚠️ DEPRECATED: Formatting now handled by event_bus.

    Returns a simple formatted string for backward compat.
    """
    kind = event.get("kind", "unknown")
    ts = event.get("ts", event.get("timestamp", ""))[:19]
    message = event.get("message", _format_message_from_fields(kind, event))
    return f"[{ts}] {message}"


def get_battle_log(limit: int = 50, since_minutes: Optional[int] = None) -> List[str]:
    """
    ⚠️ DEPRECATED: Use core.battle_log directly.

    Returns formatted battle log entries from BattleLogger.
    """
    from core.battle_log import get_battle_logger
    from datetime import timedelta

    logger = get_battle_logger()

    since = None
    if since_minutes:
        from datetime import datetime
        since = (datetime.utcnow() - timedelta(minutes=since_minutes)).isoformat() + "Z"

    events = logger.get_events(since=since, limit=limit)

    from core.battle_log import CHANNEL_ICONS
    return [
        f"[{e.timestamp[:19]}] {CHANNEL_ICONS.get(e.channel, '📢')} [{e.channel}] {e.message}"
        for e in reversed(events)
    ]


# =============================================================================
# CLI (minimal - redirect to event_bus)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Events System (Legacy Shim)")
    parser.add_argument("--list", action="store_true", help="List recent events")
    parser.add_argument("--battle-log", action="store_true", help="Show formatted battle log")
    parser.add_argument("--emit", type=str, help="Emit a test event of given kind")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to show")

    args = parser.parse_args()

    print("⚠️  WARNING: core.events is deprecated. Use core.event_bus or core.battle_log instead.\n")

    if args.emit:
        event = emit_event(args.emit, test=True, message="Test event from CLI")
        print(f"Emitted: {event}")

    elif args.battle_log:
        log = get_battle_log(limit=args.limit)
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
