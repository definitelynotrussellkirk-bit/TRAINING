"""
Event Bus - Canonical event emission for the Realm.

This is the ONLY way to emit events system-wide. All other event systems
(core/events.py, direct battle_log calls) should delegate to this module.

Architecture:
    Producer → event_bus.emit_realm_event()
                    │
                    ├──→ RealmStore (live events, SSE broadcast)
                    └──→ BattleLogger (persistent SQLite, queryable)

Both consumers automatically receive all events from the unified producer.

Usage:
    from core.event_bus import emit_realm_event

    # Generic emission
    emit_realm_event(
        kind="checkpoint_saved",
        message="Checkpoint 183,000 saved (loss: 0.2340)",
        channel="checkpoint",  # optional - auto-detected from kind
        severity="success",    # optional - auto-detected from kind
        details={"step": 183000, "loss": 0.234, "path": "/path/to/ckpt"},
    )

    # Typed helpers (recommended)
    from core.event_bus import checkpoint_saved, job_started, eval_completed

    checkpoint_saved(step=183000, loss=0.234, path="/path/to/ckpt")
    job_started(job_id="abc123", job_type="eval", worker_id="macmini_1")
    eval_completed(skill="sy", level=5, accuracy=0.942)

Consuming Events:
    1. LIVE VIEW (last N events, real-time SSE)
       from core.realm_store import get_events
       events = get_events(limit=50)

    2. HISTORIC VIEW (queryable, filterable, 24h+)
       from core.battle_log import get_battle_logger
       logger = get_battle_logger()
       events = logger.get_events(channels=["jobs"], limit=100)

Design Decisions:
    - Sync writes (7.5ms overhead per event, negligible at current volume)
    - Auto-detect channel/severity from kind
    - Persist to both RealmState (live) and BattleLog (historic)
    - No JSONL storage (use export CLI if needed)
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import storage backends (lazy to avoid circular imports)
_realm_store = None
_battle_logger = None


def _get_realm_store():
    """Lazy import RealmStore to avoid circular dependencies."""
    global _realm_store
    if _realm_store is None:
        try:
            from core.realm_store import get_store
            _realm_store = get_store()
        except Exception as e:
            logger.error(f"Failed to initialize RealmStore: {e}")
            _realm_store = False  # Mark as failed to avoid retry spam
    return _realm_store if _realm_store is not False else None


def _get_battle_logger():
    """Lazy import BattleLogger to avoid circular dependencies."""
    global _battle_logger
    if _battle_logger is None:
        try:
            from core.battle_log import get_battle_logger
            _battle_logger = get_battle_logger()
        except Exception as e:
            logger.error(f"Failed to initialize BattleLogger: {e}")
            _battle_logger = False  # Mark as failed
    return _battle_logger if _battle_logger is not False else None


# =============================================================================
# KIND METADATA - Auto-detect channel/severity from event kind
# =============================================================================

KIND_METADATA = {
    # Jobs
    "job_submitted": ("jobs", "info"),
    "job_started": ("jobs", "info"),
    "job_completed": ("jobs", "success"),
    "job_failed": ("jobs", "error"),
    "job_cancelled": ("jobs", "warning"),

    # Training
    "training_started": ("training", "info"),
    "training_step": ("training", "info"),
    "training_completed": ("training", "success"),
    "training_paused": ("training", "warning"),
    "training_resumed": ("training", "info"),
    "training_failed": ("training", "error"),
    "lr_changed": ("training", "info"),

    # Checkpoints
    "checkpoint_saved": ("checkpoint", "success"),
    "checkpoint_promoted": ("checkpoint", "success"),
    "checkpoint_deleted": ("checkpoint", "info"),
    "checkpoint_archived": ("checkpoint", "info"),

    # Evaluations
    "eval_started": ("eval", "info"),
    "eval_completed": ("eval", "success"),
    "eval_regression": ("eval", "warning"),
    "eval_failed": ("eval", "error"),
    "eval_suite_triggered": ("eval", "info"),

    # Data/Curriculum
    "dataset_generated": ("data", "success"),
    "curriculum_updated": ("data", "info"),
    "forge_batch_ready": ("data", "info"),
    "quest_created": ("data", "success"),
    "skill_data_generated": ("data", "info"),

    # System
    "server_started": ("system", "success"),
    "server_stopped": ("system", "info"),
    "config_reloaded": ("system", "info"),
    "worker_joined": ("system", "success"),
    "worker_left": ("system", "warning"),
    "worker_stale": ("system", "warning"),
    "mode_changed": ("system", "info"),
    "warning_raised": ("system", "warning"),
    "error_occurred": ("system", "error"),
    "reset_performed": ("system", "warning"),

    # Vault
    "vault_sync": ("vault", "info"),
    "vault_cleanup": ("vault", "info"),
    "retention_applied": ("vault", "info"),

    # Guild
    "title_earned": ("guild", "success"),
    "skill_level_up": ("guild", "success"),
    "achievement_unlocked": ("guild", "success"),
    "campaign_milestone": ("guild", "success"),
}


# =============================================================================
# CANONICAL EVENT EMITTER
# =============================================================================

def emit_realm_event(
    *,
    kind: str,
    message: str,
    channel: Optional[str] = None,
    severity: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    persist: bool = True,
    hero_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Canonical event emitter for the Realm.

    Writes to:
    1. RealmState (for live UIs, SSE broadcast, subscriptions)
    2. BattleLogger SQLite (for historic queries, if persist=True)

    Args:
        kind: Event kind (e.g., "checkpoint_saved", "job_started")
        message: Human-readable message for display
        channel: Event channel (auto-detected from kind if None)
        severity: Event severity (auto-detected from kind if None)
        details: Structured data (stored as JSON)
        persist: Whether to write to persistent SQLite log (default: True)
        hero_id: Hero ID if event is hero-specific
        campaign_id: Campaign ID if event is campaign-specific

    Returns:
        The event dict that was emitted

    Channels:
        system, jobs, training, eval, vault, guild, debug, data, checkpoint

    Severities:
        info, success, warning, error

    Example:
        emit_realm_event(
            kind="checkpoint_saved",
            message="Checkpoint 183,000 saved (loss: 0.2340)",
            details={"step": 183000, "loss": 0.234},
        )
    """
    # Auto-detect channel/severity from kind
    if channel is None or severity is None:
        defaults = KIND_METADATA.get(kind, ("system", "info"))
        channel = channel or defaults[0]
        severity = severity or defaults[1]

    details_dict = details or {}

    # Emit to RealmState (live events)
    event = None
    realm_store = _get_realm_store()
    if realm_store:
        try:
            event = realm_store.emit_event(
                kind=kind,
                message=message,
                channel=channel,
                severity=severity,
                details=details_dict,
            )
        except Exception as e:
            logger.error(f"Failed to emit to RealmState: {e}")

    # Persist to BattleLogger (historic events)
    if persist:
        battle_log = _get_battle_logger()
        if battle_log:
            try:
                battle_log.log(
                    channel=channel,
                    message=message,
                    severity=severity,
                    source=f"event.{kind}",
                    hero_id=hero_id,
                    campaign_id=campaign_id,
                    details=details_dict,
                )
            except Exception as e:
                logger.error(f"Failed to persist to BattleLog: {e}")

    # Return event dict (from RealmState if available, else construct)
    if event is None:
        from datetime import datetime
        event = {
            "kind": kind,
            "message": message,
            "channel": channel,
            "severity": severity,
            "details": details_dict,
            "timestamp": datetime.now().isoformat(),
        }

    return event


# =============================================================================
# TYPED HELPERS - Recommended interface for producers
# =============================================================================

def job_submitted(job_id: str, job_type: str, payload: Optional[Dict] = None):
    """Emit job_submitted event."""
    return emit_realm_event(
        kind="job_submitted",
        message=f"Job {job_type} submitted: {job_id[:8]}",
        details={"job_id": job_id, "job_type": job_type, "payload": payload or {}},
    )


def job_started(job_id: str, job_type: str, worker_id: str):
    """Emit job_started event."""
    return emit_realm_event(
        kind="job_started",
        message=f"Job {job_type} started on {worker_id}",
        details={"job_id": job_id, "job_type": job_type, "worker_id": worker_id},
    )


def job_completed(job_id: str, job_type: str, duration_sec: float, worker_id: Optional[str] = None):
    """Emit job_completed event."""
    return emit_realm_event(
        kind="job_completed",
        message=f"Job {job_type} completed in {duration_sec:.1f}s",
        details={
            "job_id": job_id,
            "job_type": job_type,
            "duration": duration_sec,
            "worker_id": worker_id,
        },
    )


def job_failed(job_id: str, job_type: str, error: str, worker_id: Optional[str] = None):
    """Emit job_failed event."""
    return emit_realm_event(
        kind="job_failed",
        message=f"Job {job_type} failed: {error[:80]}",
        details={
            "job_id": job_id,
            "job_type": job_type,
            "error": error,
            "worker_id": worker_id,
        },
    )


def checkpoint_saved(step: int, loss: float, path: str, hero_id: Optional[str] = None, campaign_id: Optional[str] = None):
    """Emit checkpoint_saved event."""
    return emit_realm_event(
        kind="checkpoint_saved",
        message=f"Checkpoint {step:,} saved (loss: {loss:.4f})",
        details={"step": step, "loss": loss, "path": path},
        hero_id=hero_id,
        campaign_id=campaign_id,
    )


def checkpoint_promoted(step: int, metric: str, value: float):
    """Emit checkpoint_promoted event (became champion)."""
    return emit_realm_event(
        kind="checkpoint_promoted",
        message=f"Checkpoint {step:,} promoted to champion ({metric}: {value:.4f})",
        details={"step": step, "metric": metric, "value": value},
    )


def training_started(job_id: str, file: str, total_steps: int, hero_id: Optional[str] = None):
    """Emit training_started event."""
    return emit_realm_event(
        kind="training_started",
        message=f"Training started: {file} ({total_steps:,} steps)",
        details={"job_id": job_id, "file": file, "total_steps": total_steps},
        hero_id=hero_id,
    )


def training_completed(job_id: str, file: str, final_step: int, final_loss: float):
    """Emit training_completed event."""
    return emit_realm_event(
        kind="training_completed",
        message=f"Training completed: {file} (step {final_step:,}, loss {final_loss:.4f})",
        details={
            "job_id": job_id,
            "file": file,
            "final_step": final_step,
            "final_loss": final_loss,
        },
    )


def eval_started(skill: str, level: int, count: int):
    """Emit eval_started event."""
    return emit_realm_event(
        kind="eval_started",
        message=f"Eval started: {skill} L{level} ({count} samples)",
        details={"skill": skill, "level": level, "count": count},
    )


def eval_completed(skill: str, level: int, accuracy: float, delta: Optional[float] = None):
    """Emit eval_completed event."""
    msg = f"Eval {skill} L{level}: {accuracy*100:.1f}%"
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        msg += f" ({sign}{delta*100:.1f}%)"

    return emit_realm_event(
        kind="eval_completed",
        message=msg,
        details={"skill": skill, "level": level, "accuracy": accuracy, "delta": delta},
    )


def eval_regression(skill: str, level: int, old_acc: float, new_acc: float):
    """Emit eval_regression event (performance dropped)."""
    delta = new_acc - old_acc
    return emit_realm_event(
        kind="eval_regression",
        message=f"Regression: {skill} L{level} dropped {abs(delta)*100:.1f}% ({old_acc*100:.1f}% → {new_acc*100:.1f}%)",
        details={"skill": skill, "level": level, "old_accuracy": old_acc, "new_accuracy": new_acc},
    )


def title_earned(hero: str, title: str, hero_id: Optional[str] = None):
    """Emit title_earned event."""
    return emit_realm_event(
        kind="title_earned",
        message=f"{hero} earned the title '{title}'",
        details={"hero": hero, "title": title},
        hero_id=hero_id,
    )


def skill_level_up(skill: str, old_level: int, new_level: int, hero_id: Optional[str] = None):
    """Emit skill_level_up event."""
    return emit_realm_event(
        kind="skill_level_up",
        message=f"Skill level up: {skill} L{old_level} → L{new_level}",
        details={"skill": skill, "old_level": old_level, "new_level": new_level},
        hero_id=hero_id,
    )


def worker_joined(worker_id: str, role: str, device: str):
    """Emit worker_joined event."""
    return emit_realm_event(
        kind="worker_joined",
        message=f"Worker {worker_id} joined ({role} on {device})",
        details={"worker_id": worker_id, "role": role, "device": device},
    )


def worker_left(worker_id: str, reason: str = "disconnected"):
    """Emit worker_left event."""
    return emit_realm_event(
        kind="worker_left",
        message=f"Worker {worker_id} left ({reason})",
        details={"worker_id": worker_id, "reason": reason},
    )


def warning_raised(warning_type: str, message: str, details: Optional[Dict] = None):
    """Emit warning_raised event."""
    return emit_realm_event(
        kind="warning_raised",
        message=f"Warning: {warning_type} - {message}",
        details={"warning_type": warning_type, **(details or {})},
    )


def mode_changed(from_mode: str, to_mode: str, reason: str = ""):
    """Emit mode_changed event."""
    msg = f"Mode changed: {from_mode} → {to_mode}"
    if reason:
        msg += f" ({reason})"
    return emit_realm_event(
        kind="mode_changed",
        message=msg,
        details={"from_mode": from_mode, "to_mode": to_mode, "reason": reason},
    )


def dataset_generated(dataset_type: str, count: int, path: str):
    """Emit dataset_generated event."""
    return emit_realm_event(
        kind="dataset_generated",
        message=f"Dataset generated: {dataset_type} ({count:,} samples)",
        details={"dataset_type": dataset_type, "count": count, "path": path},
    )


def curriculum_updated(skill: str, level: int, action: str):
    """Emit curriculum_updated event."""
    return emit_realm_event(
        kind="curriculum_updated",
        message=f"Curriculum updated: {skill} L{level} ({action})",
        details={"skill": skill, "level": level, "action": action},
    )


# =============================================================================
# EXPORT FUNCTIONALITY (optional JSONL export for analysis)
# =============================================================================

def export_to_jsonl(output_path: str, since: Optional[str] = None, limit: int = 10000):
    """
    Export events from BattleLog to JSONL file for analysis.

    Args:
        output_path: Path to output JSONL file
        since: Only export events after this timestamp (ISO format)
        limit: Maximum events to export

    Example:
        export_to_jsonl("events_last_week.jsonl", since="2025-11-23T00:00:00Z")
    """
    import json
    from pathlib import Path

    battle_log = _get_battle_logger()
    if not battle_log:
        raise RuntimeError("BattleLogger not available")

    events = battle_log.get_events(since=since, limit=limit)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for event in reversed(events):  # Oldest first
            f.write(json.dumps(event.to_dict()) + "\n")

    logger.info(f"Exported {len(events)} events to {output}")
    return len(events)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Event Bus CLI")
    parser.add_argument("--test", action="store_true", help="Emit test events")
    parser.add_argument("--export", type=str, help="Export events to JSONL file")
    parser.add_argument("--since", type=str, help="Export events since timestamp")
    parser.add_argument("--limit", type=int, default=10000, help="Max events to export")

    args = parser.parse_args()

    if args.test:
        print("Emitting test events...")
        job_started("test123", "eval", "test_worker")
        checkpoint_saved(step=183000, loss=0.234, path="/tmp/ckpt")
        eval_completed(skill="sy", level=5, accuracy=0.942, delta=0.021)
        title_earned(hero="DIO", title="The Unwavering")
        print("✓ Test events emitted to RealmState and BattleLog")

    elif args.export:
        count = export_to_jsonl(args.export, since=args.since, limit=args.limit)
        print(f"✓ Exported {count} events to {args.export}")

    else:
        parser.print_help()
