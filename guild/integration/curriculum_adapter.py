"""
Curriculum adapter for skill/level synchronization.

Bridges guild skill progression with the existing curriculum system.

Features:
- Sync guild skill levels ↔ curriculum state
- Record combat results as accuracy entries
- Check and trigger progression
- Bidirectional state management
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from guild.integration.adapters import (
    BaseAdapter,
    AdapterConfig,
    AdapterResult,
)

logger = logging.getLogger(__name__)


@dataclass
class CurriculumSkillState:
    """State of a skill in the curriculum system."""
    skill_id: str
    current_level: int = 0  # 0 = nothing mastered, training on level 1
    accuracy_history: List[Dict[str, Any]] = field(default_factory=list)
    progression_history: List[Dict[str, Any]] = field(default_factory=list)

    def recent_accuracy(self, count: int = 3) -> float:
        """Get average accuracy over recent evaluations."""
        if not self.accuracy_history:
            return 0.0
        recent = self.accuracy_history[-count:]
        if not recent:
            return 0.0
        return sum(e.get("accuracy", 0) for e in recent) / len(recent)

    def to_dict(self) -> dict:
        return {
            "current_level": self.current_level,
            "accuracy_history": self.accuracy_history,
            "progression_history": self.progression_history,
        }


@dataclass
class CurriculumState:
    """Full curriculum state across all skills."""
    skills: Dict[str, CurriculumSkillState] = field(default_factory=dict)
    active_skill: str = "syllo"
    started_at: str = ""
    last_updated: str = ""

    def get_skill(self, skill_id: str) -> CurriculumSkillState:
        """Get or create skill state."""
        if skill_id not in self.skills:
            self.skills[skill_id] = CurriculumSkillState(skill_id=skill_id)
        return self.skills[skill_id]

    def to_dict(self) -> dict:
        return {
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "active_skill": self.active_skill,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CurriculumState":
        """Load from dictionary."""
        state = cls(
            active_skill=data.get("active_skill", "syllo"),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
        )

        skills_data = data.get("skills", {})
        for skill_id, skill_data in skills_data.items():
            state.skills[skill_id] = CurriculumSkillState(
                skill_id=skill_id,
                current_level=skill_data.get("current_level", 0),  # 0 = nothing mastered
                accuracy_history=skill_data.get("accuracy_history", []),
                progression_history=skill_data.get("progression_history", []),
            )

        return state


# Skill level configurations (mirrors curriculum_manager.py)
SKILL_CONFIGS = {
    "syllo": {
        "name": "SYLLO Puzzles",
        "max_level": 10,
        "threshold": 0.80,
        "min_evals": 3,
    },
    "binary": {
        "name": "Binary Arithmetic",
        "max_level": 7,
        "threshold": 0.80,
        "min_evals": 3,
    },
    "logic": {
        "name": "Logic",
        "max_level": 10,
        "threshold": 0.80,
        "min_evals": 3,
    },
}


class CurriculumAdapter(BaseAdapter):
    """
    Adapter for curriculum state synchronization.

    Bridges guild skill system with existing curriculum_state.json.

    Features:
    - Read/write curriculum state
    - Record combat results as accuracy entries
    - Check progression conditions
    - Sync guild skill levels to curriculum
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self._state: Optional[CurriculumState] = None
        self._state_file: Optional[Path] = None

    @property
    def name(self) -> str:
        return "curriculum"

    @property
    def state_file(self) -> Path:
        """Path to curriculum state file."""
        if self._state_file is None:
            self._state_file = self.config.base_dir / "data_manager" / "curriculum_state.json"
        return self._state_file

    def health_check(self) -> bool:
        """Check if curriculum state file is accessible."""
        try:
            state_dir = self.state_file.parent
            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Curriculum adapter health check failed: {e}")
            return False

    def load_state(self) -> AdapterResult[CurriculumState]:
        """Load curriculum state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    data = json.load(f)
                self._state = CurriculumState.from_dict(data)
            else:
                # Create default state
                self._state = CurriculumState(
                    started_at=datetime.now().isoformat(),
                )
                # Initialize default skills
                for skill_id in SKILL_CONFIGS:
                    self._state.get_skill(skill_id)

            return AdapterResult.ok(self._state)

        except Exception as e:
            logger.error(f"Failed to load curriculum state: {e}")
            return AdapterResult.fail(str(e))

    def save_state(self) -> AdapterResult[Path]:
        """Save curriculum state to disk."""
        if self._state is None:
            return AdapterResult.fail("No state loaded")

        try:
            self._state.last_updated = datetime.now().isoformat()

            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, 'w') as f:
                json.dump(self._state.to_dict(), f, indent=2)

            logger.debug(f"Saved curriculum state to {self.state_file}")
            return AdapterResult.ok(self.state_file)

        except Exception as e:
            logger.error(f"Failed to save curriculum state: {e}")
            return AdapterResult.fail(str(e))

    def get_state(self) -> Optional[CurriculumState]:
        """Get current state (load if not loaded)."""
        if self._state is None:
            self.load_state()
        return self._state

    def get_skill_level(self, skill_id: str) -> int:
        """Get current level for a skill."""
        state = self.get_state()
        if state is None:
            return 1
        return state.get_skill(skill_id).current_level

    def set_skill_level(self, skill_id: str, level: int) -> AdapterResult[int]:
        """Set level for a skill."""
        state = self.get_state()
        if state is None:
            return AdapterResult.fail("State not loaded")

        skill_config = SKILL_CONFIGS.get(skill_id, {"max_level": 10})
        max_level = skill_config.get("max_level", 10)

        # Clamp to valid range
        level = max(1, min(level, max_level))

        skill_state = state.get_skill(skill_id)
        old_level = skill_state.current_level
        skill_state.current_level = level

        self.save_state()

        return AdapterResult.ok(
            level,
            old_level=old_level,
            changed=old_level != level,
        )

    def record_accuracy(
        self,
        skill_id: str,
        accuracy: float,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult[Dict[str, Any]]:
        """
        Record an accuracy measurement for a skill.

        Args:
            skill_id: Skill identifier
            accuracy: Accuracy value (0.0 - 1.0)
            step: Training step
            metadata: Additional metadata

        Returns:
            AdapterResult with recorded entry
        """
        state = self.get_state()
        if state is None:
            return AdapterResult.fail("State not loaded")

        skill_state = state.get_skill(skill_id)

        entry = {
            "step": step,
            "accuracy": accuracy,
            "level": skill_state.current_level,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        skill_state.accuracy_history.append(entry)

        self.save_state()

        return AdapterResult.ok(entry)

    def record_combat_result(
        self,
        skill_id: str,
        combat_result: str,
        correct: bool,
        step: int = 0,
    ) -> AdapterResult[Dict[str, Any]]:
        """
        Record a combat result for skill tracking.

        Accumulates results and records accuracy when batch complete.

        Args:
            skill_id: Skill identifier
            combat_result: Combat result string
            correct: Whether the answer was correct
            step: Training step

        Returns:
            AdapterResult with recorded data
        """
        # This is a simplified version - in practice you'd batch these
        accuracy = 1.0 if correct else 0.0
        return self.record_accuracy(
            skill_id=skill_id,
            accuracy=accuracy,
            step=step,
            metadata={
                "combat_result": combat_result,
                "single_result": True,
            }
        )

    def check_progression(self, skill_id: str) -> AdapterResult[Dict[str, Any]]:
        """
        Check if a skill should progress to the next level.

        Uses recent accuracy history to determine if threshold is met.

        Args:
            skill_id: Skill identifier

        Returns:
            AdapterResult with progression check result
        """
        state = self.get_state()
        if state is None:
            return AdapterResult.fail("State not loaded")

        skill_state = state.get_skill(skill_id)
        skill_config = SKILL_CONFIGS.get(skill_id, {})

        threshold = skill_config.get("threshold", 0.80)
        min_evals = skill_config.get("min_evals", 3)
        max_level = skill_config.get("max_level", 10)

        current_level = skill_state.current_level
        recent_accuracy = skill_state.recent_accuracy(min_evals)
        eval_count = len(skill_state.accuracy_history)

        # Check if already at max level
        if current_level >= max_level:
            return AdapterResult.ok({
                "should_progress": False,
                "reason": "already_at_max",
                "current_level": current_level,
                "max_level": max_level,
            })

        # Check if enough evaluations
        if eval_count < min_evals:
            return AdapterResult.ok({
                "should_progress": False,
                "reason": "insufficient_evals",
                "eval_count": eval_count,
                "min_evals": min_evals,
                "recent_accuracy": recent_accuracy,
            })

        # Check threshold
        should_progress = recent_accuracy >= threshold

        return AdapterResult.ok({
            "should_progress": should_progress,
            "reason": "threshold_met" if should_progress else "below_threshold",
            "current_level": current_level,
            "recent_accuracy": recent_accuracy,
            "threshold": threshold,
            "eval_count": eval_count,
        })

    def progress_if_ready(self, skill_id: str) -> AdapterResult[Dict[str, Any]]:
        """
        Check and apply progression if conditions are met.

        Args:
            skill_id: Skill identifier

        Returns:
            AdapterResult with progression result
        """
        check_result = self.check_progression(skill_id)
        if not check_result.success:
            return check_result

        check_data = check_result.data
        if not check_data.get("should_progress", False):
            return AdapterResult.ok({
                "progressed": False,
                **check_data
            })

        # Apply progression
        state = self.get_state()
        skill_state = state.get_skill(skill_id)

        old_level = skill_state.current_level
        new_level = old_level + 1
        skill_state.current_level = new_level

        # Record progression
        skill_state.progression_history.append({
            "from_level": old_level,
            "to_level": new_level,
            "accuracy": check_data.get("recent_accuracy", 0),
            "timestamp": datetime.now().isoformat(),
        })

        self.save_state()

        logger.info(f"Skill {skill_id} progressed from level {old_level} to {new_level}")

        return AdapterResult.ok({
            "progressed": True,
            "old_level": old_level,
            "new_level": new_level,
            "accuracy": check_data.get("recent_accuracy", 0),
        })

    def sync_from_guild_skill(
        self,
        guild_skill,  # SkillState from guild.skills
    ) -> AdapterResult[bool]:
        """
        Sync curriculum state from guild skill state.

        Args:
            guild_skill: SkillState from guild skills module

        Returns:
            AdapterResult indicating sync success
        """
        try:
            skill_id = guild_skill.id
            state = self.get_state()
            skill_state = state.get_skill(skill_id)

            # Sync level
            if hasattr(guild_skill, 'level'):
                skill_state.current_level = guild_skill.level

            # Sync recent accuracy if available
            if hasattr(guild_skill, 'accuracy') and guild_skill.accuracy is not None:
                self.record_accuracy(
                    skill_id=skill_id,
                    accuracy=guild_skill.accuracy,
                    metadata={"source": "guild_sync"}
                )

            self.save_state()
            return AdapterResult.ok(True)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def sync_to_guild_skill(
        self,
        skill_id: str,
        skill_state_manager,  # SkillStateManager from guild.skills
    ) -> AdapterResult[bool]:
        """
        Sync guild skill from curriculum state.

        Args:
            skill_id: Skill to sync
            skill_state_manager: Guild skill state manager

        Returns:
            AdapterResult indicating sync success
        """
        try:
            state = self.get_state()
            curriculum_skill = state.get_skill(skill_id)

            # Set level in guild
            skill_state_manager.set_level(skill_id, curriculum_skill.current_level)

            # Sync recent accuracy
            recent_acc = curriculum_skill.recent_accuracy(3)
            if recent_acc > 0:
                skill_state_manager.record_accuracy(
                    skill_id=skill_id,
                    accuracy=recent_acc,
                    metadata={"source": "curriculum_sync"}
                )

            return AdapterResult.ok(True)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def get_active_skill(self) -> str:
        """Get the currently active skill."""
        state = self.get_state()
        return state.active_skill if state else "syllo"

    def set_active_skill(self, skill_id: str) -> AdapterResult[str]:
        """Set the active skill."""
        state = self.get_state()
        if state is None:
            return AdapterResult.fail("State not loaded")

        state.active_skill = skill_id
        self.save_state()

        return AdapterResult.ok(skill_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum state for all skills."""
        state = self.get_state()
        if state is None:
            return {}

        return {
            "active_skill": state.active_skill,
            "skills": {
                skill_id: {
                    "level": skill.current_level,
                    "recent_accuracy": skill.recent_accuracy(3),
                    "eval_count": len(skill.accuracy_history),
                    "progressions": len(skill.progression_history),
                }
                for skill_id, skill in state.skills.items()
            }
        }


# Global adapter instance
_curriculum_adapter: Optional[CurriculumAdapter] = None


def init_curriculum_adapter(config: Optional[AdapterConfig] = None) -> CurriculumAdapter:
    """Initialize the global curriculum adapter."""
    global _curriculum_adapter
    _curriculum_adapter = CurriculumAdapter(config)
    return _curriculum_adapter


def get_curriculum_adapter() -> CurriculumAdapter:
    """Get the global curriculum adapter."""
    global _curriculum_adapter
    if _curriculum_adapter is None:
        _curriculum_adapter = CurriculumAdapter()
    return _curriculum_adapter


def reset_curriculum_adapter() -> None:
    """Reset the global curriculum adapter (for testing)."""
    global _curriculum_adapter
    _curriculum_adapter = None
