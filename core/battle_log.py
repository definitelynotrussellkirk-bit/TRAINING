"""
Battle Log - Unified event stream with MMO-style channels.

The Battle Log is your lab's global event bus. Every significant action
gets logged here, creating a real-time narrative of what the system is doing.

Channels (like MMO chat):
    system   - Server start/stop, config reload, errors affecting the whole system
    jobs     - Job claimed/started/completed/failed, retries, queue warnings
    training - Checkpoints, LR changes, campaign milestones
    eval     - Evaluation results, regressions, thresholds
    vault    - Archive/retention/sync operations (hot->warm, etc.)
    guild    - Titles, lore events, hero progression, fun stuff
    debug    - Internal assertions, edge-case logs (dev only)

Usage:
    from core.battle_log import log_event, get_battle_logger

    # Simple logging
    log_event(
        channel="jobs",
        message="Eval job claimed by macmini_eval_1",
        source="jobs.store",
        severity="info",
        details={"job_id": "abc123"}
    )

    # Or use the logger directly
    logger = get_battle_logger()
    logger.log("guild", "DIO earned the title 'The Unwavering'", source="titles")

    # Query events
    events = logger.get_events(channels=["jobs", "training"], limit=50)

RPG Flavor:
    The Battle Log is the Chronicle of the Realm - every quest completed,
    every battle won, every checkpoint saved is recorded for posterity.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("battle_log")


# =============================================================================
# ENUMS
# =============================================================================

class BattleChannel(str, Enum):
    """MMO-style chat channels."""
    SYSTEM = "system"      # Server-level events
    JOBS = "jobs"          # Job lifecycle
    TRAINING = "training"  # Training progress
    EVAL = "eval"          # Evaluation results
    VAULT = "vault"        # Storage operations
    GUILD = "guild"        # Titles, lore, fun stuff
    DEBUG = "debug"        # Developer-only


class BattleSeverity(str, Enum):
    """Event severity levels."""
    INFO = "info"          # Normal events
    SUCCESS = "success"    # Positive outcomes
    WARNING = "warning"    # Attention needed
    ERROR = "error"        # Something went wrong


# Channel icons for display
CHANNEL_ICONS = {
    "system": "⚙️",
    "jobs": "⚔️",
    "training": "📈",
    "eval": "📊",
    "vault": "🗃️",
    "guild": "🏰",
    "debug": "🔧",
}

# Severity colors (for UI)
SEVERITY_COLORS = {
    "info": "#888",
    "success": "#4CAF50",
    "warning": "#ffa726",
    "error": "#ff6b6b",
}


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class BattleLogEntry:
    """A single event in the battle log."""
    id: int
    timestamp: str
    channel: str
    source: str
    severity: str
    message: str
    hero_id: Optional[str] = None
    campaign_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "channel": self.channel,
            "source": self.source,
            "severity": self.severity,
            "message": self.message,
            "hero_id": self.hero_id,
            "campaign_id": self.campaign_id,
            "details": self.details,
            "icon": CHANNEL_ICONS.get(self.channel, "📝"),
        }


# =============================================================================
# BATTLE LOGGER
# =============================================================================

class BattleLogger:
    """
    Unified event logger with MMO-style channels.

    Uses SQLite for persistence, same DB as job store for simplicity.
    Thread-safe for concurrent writes from different modules.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS battle_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        channel TEXT NOT NULL,
        source TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'info',
        hero_id TEXT,
        campaign_id TEXT,
        message TEXT NOT NULL,
        details_json TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_battle_log_time
        ON battle_log(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_battle_log_channel
        ON battle_log(channel, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_battle_log_hero
        ON battle_log(hero_id, timestamp DESC);
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the battle logger.

        Args:
            db_path: Path to SQLite database. Defaults to vault/jobs.db
        """
        if db_path is None:
            from core.paths import get_base_dir
            db_path = get_base_dir() / "vault" / "jobs.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def log(
        self,
        channel: str,
        message: str,
        *,
        source: str = "unknown",
        severity: str = "info",
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log an event to the battle log.

        Args:
            channel: Event channel (system, jobs, training, eval, vault, guild, debug)
            message: Human-readable message (keep it short and flavorful!)
            source: Module/component that generated the event
            severity: Event severity (info, success, warning, error)
            hero_id: Hero ID if event is hero-specific
            campaign_id: Campaign ID if event is campaign-specific
            details: Optional structured data (stored as JSON)

        Returns:
            ID of the created log entry
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        details_json = json.dumps(details) if details else None

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO battle_log
                (timestamp, channel, source, severity, hero_id, campaign_id, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, channel, source, severity, hero_id, campaign_id, message, details_json),
        )
        conn.commit()

        entry_id = cursor.lastrowid
        logger.debug(f"[{channel}] {message}")
        return entry_id

    def get_events(
        self,
        channels: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: int = 50,
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[BattleLogEntry]:
        """
        Get events from the battle log.

        Args:
            channels: Filter by channels (None = all)
            since: Only events after this timestamp (ISO format)
            limit: Maximum events to return
            hero_id: Filter by hero
            campaign_id: Filter by campaign
            severity: Filter by severity

        Returns:
            List of BattleLogEntry objects, most recent first
        """
        conn = self._get_conn()

        # Build query
        conditions = []
        params = []

        if channels:
            placeholders = ",".join("?" * len(channels))
            conditions.append(f"channel IN ({placeholders})")
            params.extend(channels)

        if since:
            conditions.append("timestamp > ?")
            params.append(since)

        if hero_id:
            conditions.append("hero_id = ?")
            params.append(hero_id)

        if campaign_id:
            conditions.append("campaign_id = ?")
            params.append(campaign_id)

        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor = conn.execute(
            f"""
            SELECT id, timestamp, channel, source, severity, hero_id,
                   campaign_id, message, details_json
            FROM battle_log
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        )

        entries = []
        for row in cursor:
            details = json.loads(row["details_json"]) if row["details_json"] else None
            entries.append(BattleLogEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                channel=row["channel"],
                source=row["source"],
                severity=row["severity"],
                hero_id=row["hero_id"],
                campaign_id=row["campaign_id"],
                message=row["message"],
                details=details,
            ))

        return entries

    def get_recent(self, limit: int = 20) -> List[BattleLogEntry]:
        """Get most recent events across all channels."""
        return self.get_events(limit=limit)

    def get_channel_counts(self, hours: int = 24) -> Dict[str, int]:
        """Get event counts per channel for the last N hours."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"

        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT channel, COUNT(*) as count
            FROM battle_log
            WHERE timestamp > ?
            GROUP BY channel
            """,
            (cutoff,),
        )

        return {row["channel"]: row["count"] for row in cursor}

    def cleanup_old(self, max_age_days: int = 7) -> int:
        """
        Remove old log entries.

        Args:
            max_age_days: Remove entries older than this

        Returns:
            Number of entries removed
        """
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat() + "Z"

        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM battle_log WHERE timestamp < ?",
            (cutoff,),
        )
        conn.commit()

        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} old battle log entries")
        return count


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_battle_logger: Optional[BattleLogger] = None
_lock = threading.Lock()


def get_battle_logger() -> BattleLogger:
    """Get the global battle logger instance."""
    global _battle_logger
    if _battle_logger is None:
        with _lock:
            if _battle_logger is None:
                _battle_logger = BattleLogger()
    return _battle_logger


def log_event(
    channel: str,
    message: str,
    *,
    source: str = "unknown",
    severity: str = "info",
    hero_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Log an event to the battle log (convenience function).

    See BattleLogger.log() for full documentation.
    """
    return get_battle_logger().log(
        channel=channel,
        message=message,
        source=source,
        severity=severity,
        hero_id=hero_id,
        campaign_id=campaign_id,
        details=details,
    )


# Convenience functions for each channel
def log_system(message: str, **kwargs) -> int:
    """Log a system event."""
    return log_event("system", message, source=kwargs.pop("source", "system"), **kwargs)


def log_jobs(message: str, **kwargs) -> int:
    """Log a jobs event."""
    return log_event("jobs", message, source=kwargs.pop("source", "jobs.store"), **kwargs)


def log_training(message: str, **kwargs) -> int:
    """Log a training event."""
    return log_event("training", message, source=kwargs.pop("source", "training"), **kwargs)


def log_eval(message: str, **kwargs) -> int:
    """Log an eval event."""
    return log_event("eval", message, source=kwargs.pop("source", "eval_runner"), **kwargs)


def log_vault(message: str, **kwargs) -> int:
    """Log a vault event."""
    return log_event("vault", message, source=kwargs.pop("source", "vault"), **kwargs)


def log_guild(message: str, **kwargs) -> int:
    """Log a guild event (titles, lore, fun stuff)."""
    return log_event("guild", message, source=kwargs.pop("source", "guild"), **kwargs)


def log_debug(message: str, **kwargs) -> int:
    """Log a debug event."""
    return log_event("debug", message, source=kwargs.pop("source", "debug"), **kwargs)


# =============================================================================
# MESSAGE FORMATTERS (for consistent MMO-style messages)
# =============================================================================

def format_job_claimed(job_type: str, job_id: str, worker_id: str) -> str:
    """Format a job claimed message."""
    return f"Job {job_type} claimed by {worker_id}"


def format_job_completed(job_type: str, job_id: str, duration_sec: float) -> str:
    """Format a job completed message."""
    return f"Job {job_type} completed in {duration_sec:.1f}s"


def format_job_failed(job_type: str, job_id: str, error_code: str) -> str:
    """Format a job failed message."""
    return f"Job {job_type} failed [{error_code}]"


def format_checkpoint_saved(step: int, loss: Optional[float] = None) -> str:
    """Format a checkpoint saved message."""
    if loss:
        return f"Checkpoint {step:,} saved (loss: {loss:.4f})"
    return f"Checkpoint {step:,} saved"


def format_eval_result(skill: str, level: int, accuracy: float, delta: Optional[float] = None) -> str:
    """Format an eval result message."""
    base = f"Eval {skill} L{level}: {accuracy*100:.1f}%"
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        base += f" ({sign}{delta*100:.1f}%)"
    return base


def format_title_earned(hero: str, title: str) -> str:
    """Format a title earned message."""
    return f"{hero} earned the title '{title}'"


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # Add parent to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Battle Log CLI")
    parser.add_argument("--tail", type=int, default=20, help="Show last N events")
    parser.add_argument("--channel", type=str, help="Filter by channel")
    parser.add_argument("--test", action="store_true", help="Add test events")

    args = parser.parse_args()

    blog = get_battle_logger()

    if args.test:
        # Add some test events
        log_system("VaultKeeper started", severity="success")
        log_jobs("Job eval claimed by macmini_eval_1", details={"job_id": "test123"})
        log_training("Checkpoint 183000 saved", severity="success", details={"step": 183000, "loss": 0.234})
        log_eval("Eval bin L5: 94.2% (+2.1%)", severity="success", details={"skill": "bin", "level": 5, "accuracy": 0.942})
        log_guild("DIO earned the title 'The Unwavering'", severity="success")
        log_vault("Synced 3 checkpoints to warm zone", details={"count": 3, "zone": "warm"})
        print("Added test events")
        print()

    # Show recent events
    channels = [args.channel] if args.channel else None
    events = blog.get_events(channels=channels, limit=args.tail)

    print(f"Battle Log ({len(events)} events)")
    print("=" * 60)

    for event in reversed(events):  # Show oldest first
        icon = CHANNEL_ICONS.get(event.channel, "📝")
        ts = event.timestamp[:19].replace("T", " ")
        sev = event.severity[0].upper()
        print(f"[{ts}] {icon} [{event.channel:8}] [{sev}] {event.message}")

    print()
    print("Channel counts (24h):")
    for ch, count in sorted(blog.get_channel_counts().items()):
        print(f"  {ch}: {count}")
