"""Skill state management and persistence."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

from guild.skills.types import SkillState, SkillConfig
from guild.skills.registry import get_registry


logger = logging.getLogger(__name__)


@dataclass
class AccuracyRecord:
    """Record of an accuracy evaluation."""
    step: int
    accuracy: float
    level: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "accuracy": self.accuracy,
            "level": self.level,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccuracyRecord":
        return cls(
            step=data["step"],
            accuracy=data["accuracy"],
            level=data["level"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProgressionRecord:
    """Record of a level progression."""
    from_level: int
    to_level: int
    timestamp: str
    triggered_by: str = "auto"  # "auto" or "manual"

    def to_dict(self) -> dict:
        return {
            "from_level": self.from_level,
            "to_level": self.to_level,
            "timestamp": self.timestamp,
            "triggered_by": self.triggered_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressionRecord":
        return cls(
            from_level=data["from_level"],
            to_level=data["to_level"],
            timestamp=data["timestamp"],
            triggered_by=data.get("triggered_by", "auto"),
        )


class SkillStateManager:
    """
    Manages skill state persistence.

    Stores per-skill state including:
    - Current level
    - Accuracy history
    - Progression history
    - Rolling accuracy window

    State is persisted to JSON files in the status directory.
    """

    def __init__(
        self,
        state_dir: Path,
        state_file: str = "skill_states.json",
        history_limit: int = 100,
    ):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
            state_file: Name of state file
            history_limit: Max accuracy records to keep per skill
        """
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / state_file
        self.history_limit = history_limit

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._states: Dict[str, SkillState] = {}
        self._accuracy_history: Dict[str, list[AccuracyRecord]] = {}
        self._progression_history: Dict[str, list[ProgressionRecord]] = {}
        self._metadata: Dict[str, Any] = {}

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if not self.state_file.exists():
            self._metadata = {
                "created_at": datetime.now().isoformat(),
            }
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse state file: {e}")
            return

        # Load metadata
        self._metadata = data.get("metadata", {})

        # Load per-skill states
        for skill_id, skill_data in data.get("skills", {}).items():
            # Load SkillState
            state_data = skill_data.get("state", {})
            state_data["skill_id"] = skill_id
            self._states[skill_id] = SkillState.from_dict(state_data)

            # Load accuracy history
            history = skill_data.get("accuracy_history", [])
            self._accuracy_history[skill_id] = [
                AccuracyRecord.from_dict(r) for r in history
            ]

            # Load progression history
            progressions = skill_data.get("progression_history", [])
            self._progression_history[skill_id] = [
                ProgressionRecord.from_dict(r) for r in progressions
            ]

        logger.debug(f"Loaded state for {len(self._states)} skills")

    def _save_state(self):
        """Save state to disk."""
        data = {
            "metadata": {
                **self._metadata,
                "last_updated": datetime.now().isoformat(),
            },
            "skills": {},
        }

        for skill_id in set(self._states.keys()) | set(self._accuracy_history.keys()):
            skill_data = {}

            if skill_id in self._states:
                skill_data["state"] = self._states[skill_id].to_dict()

            if skill_id in self._accuracy_history:
                skill_data["accuracy_history"] = [
                    r.to_dict() for r in self._accuracy_history[skill_id]
                ]

            if skill_id in self._progression_history:
                skill_data["progression_history"] = [
                    r.to_dict() for r in self._progression_history[skill_id]
                ]

            data["skills"][skill_id] = skill_data

        # Atomic write
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(self.state_file)

        logger.debug("Skill state saved")

    def get_state(self, skill_id: str) -> SkillState:
        """
        Get state for a skill, creating if needed.

        Args:
            skill_id: Skill identifier

        Returns:
            SkillState instance
        """
        if skill_id not in self._states:
            self._states[skill_id] = SkillState(skill_id=skill_id)
        return self._states[skill_id]

    def get_accuracy_history(self, skill_id: str) -> list[AccuracyRecord]:
        """Get accuracy history for a skill."""
        return self._accuracy_history.get(skill_id, []).copy()

    def get_progression_history(self, skill_id: str) -> list[ProgressionRecord]:
        """Get progression history for a skill."""
        return self._progression_history.get(skill_id, []).copy()

    def record_accuracy(
        self,
        skill_id: str,
        accuracy: float,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record an accuracy measurement.

        Args:
            skill_id: Skill identifier
            accuracy: Accuracy score (0-1)
            step: Training step
            metadata: Optional metadata
        """
        state = self.get_state(skill_id)

        record = AccuracyRecord(
            step=step,
            accuracy=accuracy,
            level=state.level,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        if skill_id not in self._accuracy_history:
            self._accuracy_history[skill_id] = []

        self._accuracy_history[skill_id].append(record)

        # Trim history
        if len(self._accuracy_history[skill_id]) > self.history_limit:
            self._accuracy_history[skill_id] = \
                self._accuracy_history[skill_id][-self.history_limit:]

        # Update rolling accuracy
        state.record_result(accuracy >= 0.5)  # Simple success/fail

        self._save_state()
        logger.info(f"[{skill_id}] Recorded accuracy: {accuracy:.1%} at step {step}")

    def record_progression(
        self,
        skill_id: str,
        from_level: int,
        to_level: int,
        triggered_by: str = "auto",
    ):
        """
        Record a level progression.

        Args:
            skill_id: Skill identifier
            from_level: Previous level
            to_level: New level
            triggered_by: What triggered the progression
        """
        record = ProgressionRecord(
            from_level=from_level,
            to_level=to_level,
            timestamp=datetime.now().isoformat(),
            triggered_by=triggered_by,
        )

        if skill_id not in self._progression_history:
            self._progression_history[skill_id] = []

        self._progression_history[skill_id].append(record)

        # Update state
        state = self.get_state(skill_id)
        state.record_level_up()

        self._save_state()
        logger.info(f"[{skill_id}] PROGRESSION: Level {from_level} -> {to_level}")

    def set_level(self, skill_id: str, level: int, triggered_by: str = "manual"):
        """
        Set skill level directly.

        Args:
            skill_id: Skill identifier
            level: New level
            triggered_by: What triggered the change
        """
        state = self.get_state(skill_id)
        old_level = state.level

        if old_level != level:
            self.record_progression(skill_id, old_level, level, triggered_by)

        state.level = level
        self._save_state()

    def check_progression(
        self,
        skill_id: str,
        min_evals: int = 3,
    ) -> tuple[bool, str]:
        """
        Check if skill should progress to next level.

        Args:
            skill_id: Skill identifier
            min_evals: Minimum evaluations needed

        Returns:
            (should_progress, reason)
        """
        registry = get_registry()

        if not registry.exists(skill_id):
            return False, f"Unknown skill: {skill_id}"

        skill_config = registry.get(skill_id)
        state = self.get_state(skill_id)
        current_level = state.level

        # Get threshold for current level
        threshold = skill_config.get_threshold(current_level)

        # Get recent accuracy at current level
        history = self.get_accuracy_history(skill_id)
        at_level = [r for r in history if r.level == current_level]

        if len(at_level) < min_evals:
            return False, f"Need {min_evals} evals at level (have {len(at_level)})"

        # Check last N evals
        recent = at_level[-min_evals:]
        avg_accuracy = sum(r.accuracy for r in recent) / len(recent)

        if avg_accuracy >= threshold:
            return True, f"Avg accuracy {avg_accuracy:.1%} >= threshold {threshold:.0%}"

        return False, f"Avg accuracy {avg_accuracy:.1%} < threshold {threshold:.0%}"

    def progress_if_ready(
        self,
        skill_id: str,
        min_evals: int = 3,
    ) -> tuple[bool, Optional[int]]:
        """
        Progress to next level if criteria are met.

        Returns:
            (progressed, new_level or None)
        """
        should_progress, reason = self.check_progression(skill_id, min_evals)
        logger.info(f"[{skill_id}] Progression check: {reason}")

        if not should_progress:
            return False, None

        state = self.get_state(skill_id)
        old_level = state.level
        new_level = old_level + 1

        self.record_progression(skill_id, old_level, new_level, "auto")

        return True, new_level

    def reset_skill(self, skill_id: str):
        """Reset a skill to level 1."""
        # Get old level for audit trail
        old_level = self.get_state(skill_id).level

        # Reset state
        self._states[skill_id] = SkillState(skill_id=skill_id)
        self._accuracy_history[skill_id] = []

        # Record reset as progression (without calling record_level_up)
        if skill_id not in self._progression_history:
            self._progression_history[skill_id] = []

        self._progression_history[skill_id].append(ProgressionRecord(
            from_level=old_level,
            to_level=1,
            timestamp=datetime.now().isoformat(),
            triggered_by="reset",
        ))

        self._save_state()
        logger.info(f"[{skill_id}] Reset to Level 1")

    def get_status(self, skill_id: str) -> Dict[str, Any]:
        """Get detailed status for a skill."""
        registry = get_registry()
        state = self.get_state(skill_id)
        history = self.get_accuracy_history(skill_id)
        progressions = self.get_progression_history(skill_id)

        # Get config if available
        skill_config = registry.get_or_none(skill_id)
        skill_name = skill_config.name if skill_config else skill_id

        # Recent accuracy
        at_level = [r for r in history if r.level == state.level]
        recent = at_level[-3:] if at_level else []
        avg_accuracy = sum(r.accuracy for r in recent) / len(recent) if recent else None

        # Check progression
        should_progress, reason = self.check_progression(skill_id)

        return {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "level": state.level,
            "xp_total": state.xp_total,
            "accuracy": state.accuracy,
            "recent_accuracy": avg_accuracy,
            "evals_at_level": len(at_level),
            "total_evals": len(history),
            "progressions": len(progressions),
            "should_progress": should_progress,
            "reason": reason,
            "eligible_for_trial": state.eligible_for_trial,
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all skills."""
        all_skills = set(self._states.keys()) | set(self._accuracy_history.keys())
        return {skill_id: self.get_status(skill_id) for skill_id in all_skills}


# Global state manager
_manager: Optional[SkillStateManager] = None


def init_state_manager(
    state_dir: Path,
    state_file: str = "skill_states.json",
) -> SkillStateManager:
    """Initialize the global skill state manager."""
    global _manager
    _manager = SkillStateManager(state_dir, state_file)
    return _manager


def get_state_manager() -> SkillStateManager:
    """Get the global skill state manager."""
    global _manager
    if _manager is None:
        raise RuntimeError(
            "Skill state manager not initialized. "
            "Call init_state_manager() first."
        )
    return _manager


def reset_state_manager():
    """Reset the global skill state manager (for testing)."""
    global _manager
    _manager = None
