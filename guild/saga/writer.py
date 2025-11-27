"""
SagaWriter - Append-only narrative log writer.

Writes tales to JSONL files, organized by date:
    logs/saga/2025-11-27.jsonl
    logs/saga/2025-11-26.jsonl
    ...

The Saga is append-only - tales are never modified or deleted
(except by retention policies cleaning old files).

Thread-safe for concurrent writes from multiple sources.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from guild.saga.types import TaleEntry, get_icon, get_category

logger = logging.getLogger(__name__)


class SagaWriter:
    """
    Writes tales to the Saga (append-only JSONL).

    Usage:
        saga = SagaWriter(base_dir)

        # Write a tale
        saga.tell("quest.started", "DIO begins quest: binary_L5.jsonl")

        # With extra data
        saga.tell(
            "hero.level_up",
            "LEVEL UP! DIO reached Level 42",
            level=42,
            xp_total=42000,
        )

        # Custom icon
        saga.tell(
            "custom.event",
            "Something happened",
            icon="🎉",
        )
    """

    def __init__(self, base_dir: Path | str):
        """
        Initialize SagaWriter.

        Args:
            base_dir: Base training directory. Saga files go in {base_dir}/logs/saga/
        """
        self.base_dir = Path(base_dir)
        self.saga_dir = self.base_dir / "logs" / "saga"
        self.saga_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._current_date: Optional[str] = None

    def _get_file_for_date(self, date: datetime) -> Path:
        """Get the saga file path for a given date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.saga_dir / f"{date_str}.jsonl"

    def _get_current_file(self) -> Path:
        """Get the current day's saga file, handling day rollover."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Check for day rollover
        if self._current_date != today:
            self._current_date = today
            self._current_file = self.saga_dir / f"{today}.jsonl"

        return self._current_file

    def tell(
        self,
        event_type: str,
        message: str,
        icon: Optional[str] = None,
        **data: Any,
    ) -> TaleEntry:
        """
        Record a tale in the Saga.

        Args:
            event_type: Event type (e.g., "quest.started", "hero.level_up")
            message: Human-readable narrative message
            icon: Optional icon override (default: auto from event_type)
            **data: Additional structured data to store

        Returns:
            The TaleEntry that was recorded

        Example:
            saga.tell("quest.started", "DIO begins quest: binary_L5.jsonl")
            saga.tell("hero.level_up", "LEVEL UP!", level=42, icon="🎊")
        """
        entry = TaleEntry(
            timestamp=datetime.now(),
            icon=icon or get_icon(event_type),
            message=message,
            event_type=event_type,
            category=get_category(event_type),
            data=data,
        )

        self._write_entry(entry)
        logger.debug(f"Saga: {entry.format_display()}")
        return entry

    def _write_entry(self, entry: TaleEntry) -> None:
        """Write a tale entry to the current file."""
        with self._lock:
            file_path = self._get_current_file()
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")

    def tell_raw(self, entry: TaleEntry) -> None:
        """Write a pre-constructed TaleEntry."""
        self._write_entry(entry)
        logger.debug(f"Saga: {entry.format_display()}")

    # === Convenience methods for common events ===

    def quest_started(self, quest_file: str, skill: str = "", level: int = 0) -> TaleEntry:
        """Record a quest starting."""
        msg = f"DIO begins quest: {quest_file}"
        if skill and level:
            msg = f"DIO begins {skill} L{level} quest: {quest_file}"
        return self.tell("quest.started", msg, quest_file=quest_file, skill=skill, level=level)

    def quest_completed(self, quest_file: str, xp: int = 0, duration_s: float = 0) -> TaleEntry:
        """Record a quest completing."""
        msg = f"Quest complete: {quest_file}"
        if xp:
            msg += f" (+{xp} XP)"
        return self.tell("quest.completed", msg, quest_file=quest_file, xp=xp, duration_s=duration_s)

    def quest_failed(self, quest_file: str, reason: str = "") -> TaleEntry:
        """Record a quest failing."""
        msg = f"Quest failed: {quest_file}"
        if reason:
            msg += f" ({reason})"
        return self.tell("quest.failed", msg, quest_file=quest_file, reason=reason)

    def level_up(self, hero: str, new_level: int, xp_total: int = 0) -> TaleEntry:
        """Record a hero leveling up."""
        msg = f"LEVEL UP! {hero} reached Level {new_level}"
        return self.tell("hero.level_up", msg, hero=hero, level=new_level, xp_total=xp_total)

    def skill_level_up(self, skill: str, new_level: int) -> TaleEntry:
        """Record a skill leveling up."""
        msg = f"Skill level up! {skill} is now Level {new_level}"
        return self.tell("hero.skill_level_up", msg, skill=skill, level=new_level)

    def champion_crowned(self, checkpoint: str, score: float = 0) -> TaleEntry:
        """Record a new champion checkpoint."""
        msg = f"New champion! {checkpoint}"
        if score:
            msg += f" (score: {score:.3f})"
        return self.tell("champion.crowned", msg, checkpoint=checkpoint, score=score)

    def champion_deployed(self, checkpoint: str, target: str = "Oracle") -> TaleEntry:
        """Record champion deployment."""
        msg = f"Champion deployed to {target}: {checkpoint}"
        return self.tell("champion.deployed", msg, checkpoint=checkpoint, target=target)

    def combat_crit(self, step: int, loss: float) -> TaleEntry:
        """Record exceptional training performance."""
        msg = f"CRIT! Step {step} - Loss dropped to {loss:.4f}"
        return self.tell("combat.crit", msg, step=step, loss=loss)

    def checkpoint_saved(self, checkpoint: str, step: int) -> TaleEntry:
        """Record checkpoint save."""
        msg = f"Checkpoint saved: {checkpoint}"
        return self.tell("combat.checkpoint", msg, checkpoint=checkpoint, step=step)

    def training_idle(self, reason: str = "No quests in queue") -> TaleEntry:
        """Record training going idle."""
        msg = f"DIO rests at the tavern... ({reason})"
        return self.tell("training.idle", msg, reason=reason)

    def system_error(self, error: str, source: str = "") -> TaleEntry:
        """Record a system error."""
        msg = f"Error: {error}"
        if source:
            msg = f"[{source}] {msg}"
        return self.tell("system.error", msg, error=error, source=source)


# Module-level singleton
_default_writer: Optional[SagaWriter] = None


def init_saga(base_dir: Path | str) -> SagaWriter:
    """Initialize the default Saga writer."""
    global _default_writer
    _default_writer = SagaWriter(base_dir)
    return _default_writer


def get_saga() -> Optional[SagaWriter]:
    """Get the default Saga writer (None if not initialized)."""
    return _default_writer


def tell(event_type: str, message: str, **data) -> Optional[TaleEntry]:
    """Write to the default Saga (no-op if not initialized)."""
    if _default_writer:
        return _default_writer.tell(event_type, message, **data)
    return None
