#!/usr/bin/env python3
"""
Chronicle - Persistent Battle Log for the Realm.

The Chronicle records all significant events in the realm:
    - Training start/stop
    - Level ups
    - Skill evaluations
    - Checkpoint saves
    - Errors and warnings

Events are written to a daily log file that rotates at midnight.
Old chronicles are archived to the vault.

Usage:
    from watchtower.chronicle import Chronicle

    chronicle = Chronicle()
    chronicle.record("training_start", {"file": "data.jsonl", "step": 1000})
    chronicle.record("level_up", {"level": 10, "step": 10000})

    # Get recent entries for UI
    recent = chronicle.recent(limit=50)
"""

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import shutil


class Chronicle:
    """Persistent event log for the realm."""

    # Event types and their display info
    EVENT_TYPES = {
        "training_start": {"icon": "⚔️", "color": "success"},
        "training_stop": {"icon": "💤", "color": "info"},
        "training_complete": {"icon": "✅", "color": "success"},
        "level_up": {"icon": "🎉", "color": "success"},
        "skill_eval": {"icon": "🎯", "color": "info"},
        "skill_level_up": {"icon": "📈", "color": "success"},
        "checkpoint_save": {"icon": "💾", "color": "info"},
        "checkpoint_deploy": {"icon": "🚀", "color": "success"},
        "error": {"icon": "❌", "color": "error"},
        "warning": {"icon": "⚠️", "color": "warning"},
        "system": {"icon": "⚙️", "color": "info"},
        "milestone": {"icon": "🏆", "color": "success"},
    }

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the Chronicle.

        Args:
            base_dir: Base directory for logs. Defaults to TRAINING dir.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)

        # Current log directory
        self.log_dir = self.base_dir / "logs" / "chronicle"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Archive directory (in vault)
        self.archive_dir = self.base_dir / "vault" / "chronicles"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Current day's log file
        self._current_date: Optional[date] = None
        self._log_file: Optional[Path] = None
        # RLock (reentrant) because _ensure_current_log() calls record() while holding lock
        self._lock = threading.RLock()

        # Rotate on init if needed
        self._ensure_current_log()

    def _ensure_current_log(self) -> Path:
        """Ensure we have a log file for today, rotating if needed."""
        today = date.today()

        if self._current_date != today:
            with self._lock:
                # Archive yesterday's log if it exists
                if self._log_file and self._log_file.exists():
                    self._archive_log(self._log_file)

                # Create new log file
                self._current_date = today
                self._log_file = self.log_dir / f"{today.isoformat()}.jsonl"

                # Write header if new file
                if not self._log_file.exists():
                    self.record("system", {
                        "message": f"Chronicle opened for {today.isoformat()}"
                    })

        return self._log_file

    def _archive_log(self, log_file: Path):
        """Archive an old log file to the vault."""
        if not log_file.exists():
            return

        archive_path = self.archive_dir / log_file.name

        # Don't overwrite existing archives
        if archive_path.exists():
            return

        try:
            shutil.copy2(log_file, archive_path)
            # Keep original in logs/ for a week, then clean up
        except Exception as e:
            print(f"[Chronicle] Failed to archive {log_file}: {e}")

    def record(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Record an event to the chronicle.

        Args:
            event_type: Type of event (see EVENT_TYPES)
            data: Additional event data
        """
        log_file = self._ensure_current_log()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {},
        }

        # Add display info
        type_info = self.EVENT_TYPES.get(event_type, {"icon": "📝", "color": "info"})
        entry["icon"] = type_info["icon"]
        entry["color"] = type_info["color"]

        with self._lock:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def recent(self, limit: int = 50, event_types: Optional[List[str]] = None) -> List[Dict]:
        """Get recent chronicle entries.

        Args:
            limit: Maximum entries to return
            event_types: Filter by event types (None = all)

        Returns:
            List of entries, newest first
        """
        log_file = self._ensure_current_log()
        entries = []

        if not log_file.exists():
            return entries

        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if event_types is None or entry.get("type") in event_types:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Chronicle] Error reading log: {e}")

        # Return newest first, limited
        return list(reversed(entries[-limit:]))

    def get_stats(self) -> Dict[str, Any]:
        """Get chronicle statistics."""
        log_file = self._ensure_current_log()

        stats = {
            "current_file": str(log_file),
            "today": self._current_date.isoformat() if self._current_date else None,
            "entries_today": 0,
            "by_type": {},
            "archived_days": 0,
        }

        # Count today's entries
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            stats["entries_today"] += 1
                            try:
                                entry = json.loads(line)
                                etype = entry.get("type", "unknown")
                                stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1
                            except:
                                pass
            except:
                pass

        # Count archived files
        stats["archived_days"] = len(list(self.archive_dir.glob("*.jsonl")))

        return stats


# Global chronicle instance
_chronicle: Optional[Chronicle] = None


def get_chronicle() -> Chronicle:
    """Get the global chronicle instance."""
    global _chronicle
    if _chronicle is None:
        _chronicle = Chronicle()
    return _chronicle


# Convenience functions
def record_event(event_type: str, data: Optional[Dict[str, Any]] = None):
    """Record an event to the chronicle."""
    get_chronicle().record(event_type, data)


def record_training_start(file: str, step: int):
    """Record training start."""
    record_event("training_start", {"file": file, "step": step})


def record_training_stop(step: int, reason: str = "completed"):
    """Record training stop."""
    record_event("training_stop", {"step": step, "reason": reason})


def record_level_up(level: int, step: int):
    """Record a level up."""
    record_event("level_up", {"level": level, "step": step})


def record_skill_eval(skill: str, level: int, accuracy: float, step: int):
    """Record a skill evaluation."""
    record_event("skill_eval", {
        "skill": skill,
        "level": level,
        "accuracy": accuracy,
        "step": step
    })


def record_checkpoint(checkpoint: str, step: int, metrics: Optional[Dict] = None):
    """Record a checkpoint save."""
    record_event("checkpoint_save", {
        "checkpoint": checkpoint,
        "step": step,
        "metrics": metrics or {}
    })


def record_error(message: str, details: Optional[Dict] = None):
    """Record an error."""
    record_event("error", {"message": message, "details": details or {}})


def record_milestone(message: str, step: int):
    """Record a milestone."""
    record_event("milestone", {"message": message, "step": step})


if __name__ == "__main__":
    # Test the chronicle
    chronicle = Chronicle()

    # Record some test events
    chronicle.record("system", {"message": "Chronicle test started"})
    chronicle.record("training_start", {"file": "test.jsonl", "step": 1000})
    chronicle.record("level_up", {"level": 5, "step": 5000})
    chronicle.record("skill_eval", {"skill": "SYLLO", "level": 2, "accuracy": 85.5})

    # Get recent entries
    print("\nRecent entries:")
    for entry in chronicle.recent(limit=10):
        print(f"  {entry['icon']} [{entry['type']}] {entry.get('data', {})}")

    # Get stats
    print("\nStats:")
    stats = chronicle.get_stats()
    print(f"  Entries today: {stats['entries_today']}")
    print(f"  By type: {stats['by_type']}")
