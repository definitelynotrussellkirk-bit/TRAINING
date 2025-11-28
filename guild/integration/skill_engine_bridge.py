"""
Skill Engine Bridge - Connects SkillEngine to existing Guild infrastructure.

This module bridges:
- SkillEngine ↔ CurriculumAdapter (for level/accuracy sync)
- SkillEngine ↔ Training generators (for data generation)
- SkillEngine ↔ Passives (for eval)

Design: SkillEngine is the central truth for skill operations.
This bridge ensures existing systems (curriculum, queue, training) integrate cleanly.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingRequest:
    """Request for training data from a skill."""
    skill_id: str
    level: int
    count: int
    difficulty: Optional[str] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    """Batch of training examples from a skill."""
    skill_id: str
    level: int
    examples: List[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.examples)


@dataclass
class EvalSyncResult:
    """Result of syncing eval results to curriculum."""
    skill_id: str
    accuracy: float
    level: int
    level_changed: bool = False
    new_level: Optional[int] = None
    primitives_updated: List[str] = field(default_factory=list)


class SkillEngineBridge:
    """
    Bridges SkillEngine with existing Guild infrastructure.

    Responsibilities:
    1. Sync eval results from SkillEngine → CurriculumAdapter
    2. Fetch training data via skill generators
    3. Coordinate state between SkillEngine and curriculum
    4. Provide unified interface for skill operations

    Usage:
        bridge = SkillEngineBridge()

        # Sync eval results
        result = bridge.sync_eval_result("bin", eval_result)

        # Get training data
        batch = bridge.get_training_batch("bin", level=3, count=100)

        # Check progression
        should_level = bridge.check_progression("bin")
    """

    def __init__(self, base_dir: Optional[Path] = None):
        from core.paths import get_base_dir
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()

        self._engine = None
        self._curriculum_adapter = None

    @property
    def engine(self):
        """Lazy-load SkillEngine."""
        if self._engine is None:
            from guild.skills import get_engine
            self._engine = get_engine()
        return self._engine

    @property
    def curriculum(self):
        """Lazy-load CurriculumAdapter."""
        if self._curriculum_adapter is None:
            from guild.integration.curriculum_adapter import CurriculumAdapter
            from guild.integration.adapters import AdapterConfig
            config = AdapterConfig(base_dir=self.base_dir)
            self._curriculum_adapter = CurriculumAdapter(config)
        return self._curriculum_adapter

    def sync_eval_result(
        self,
        skill_id: str,
        accuracy: float,
        level: int,
        step: int = 0,
        per_primitive: Optional[Dict[str, float]] = None,
        hero_id: Optional[str] = None,
    ) -> EvalSyncResult:
        """
        Sync an eval result from SkillEngine to curriculum system.

        Args:
            skill_id: Skill identifier
            accuracy: Overall accuracy (0.0-1.0)
            level: Level that was evaluated
            step: Training step (for tracking)
            per_primitive: Per-primitive accuracy breakdown
            hero_id: Optional hero ID for multi-hero support

        Returns:
            EvalSyncResult with sync status
        """
        # Update curriculum adapter
        result = self.curriculum.record_accuracy(
            skill_id=skill_id,
            accuracy=accuracy,
            step=step,
            metadata={
                "level": level,
                "per_primitive": per_primitive,
                "hero_id": hero_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Check for level-up
        level_changed = False
        new_level = None

        if result.success:
            # Check progression via curriculum
            progression = self.curriculum.check_progression(skill_id)
            if progression.success and progression.data.get("should_progress"):
                new_level = progression.data.get("new_level")
                level_changed = True

                # Sync level to curriculum
                self.curriculum.set_skill_level(skill_id, new_level)

                logger.info(f"Skill {skill_id} leveled up: {level} → {new_level}")

        return EvalSyncResult(
            skill_id=skill_id,
            accuracy=accuracy,
            level=level,
            level_changed=level_changed,
            new_level=new_level,
            primitives_updated=list(per_primitive.keys()) if per_primitive else [],
        )

    def get_training_batch(
        self,
        skill_id: str,
        level: int,
        count: int = 100,
        difficulty: Optional[str] = None,
    ) -> Optional[TrainingBatch]:
        """
        Get training data for a skill via the skill's generator.

        Uses SkillEngine to route to the appropriate generator.

        Args:
            skill_id: Skill identifier
            level: Difficulty level
            count: Number of examples
            difficulty: Optional difficulty override

        Returns:
            TrainingBatch with examples, or None if generation failed
        """
        try:
            skill = self.engine.get(skill_id)

            # Use skill's training generator
            examples = skill.generate_training_batch(
                level=level,
                count=count,
                difficulty=difficulty,
            )

            return TrainingBatch(
                skill_id=skill_id,
                level=level,
                examples=examples,
                metadata={
                    "difficulty": difficulty,
                    "generator": "skill_engine",
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate training batch for {skill_id}: {e}")
            return None

    def get_current_level(self, skill_id: str) -> int:
        """Get current mastered level for a skill."""
        return self.curriculum.get_skill_level(skill_id)

    def get_training_level(self, skill_id: str) -> int:
        """Get level currently being trained (mastered + 1)."""
        return self.curriculum.get_skill_level(skill_id) + 1

    def check_progression(self, skill_id: str) -> Dict[str, Any]:
        """
        Check if skill should progress to next level.

        Returns dict with:
        - should_progress: bool
        - current_level: int
        - new_level: int (if should_progress)
        - reason: str
        """
        result = self.curriculum.check_progression(skill_id)
        if result.success:
            return result.data
        return {
            "should_progress": False,
            "current_level": self.get_current_level(skill_id),
            "reason": result.error or "Check failed",
        }

    def get_skill_summary(self, skill_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a skill's state.

        Combines data from SkillEngine and CurriculumAdapter.
        """
        # From curriculum
        curriculum_level = self.get_current_level(skill_id)

        # From engine
        try:
            engine_state = self.engine.get_state(skill_id)
            primitive_accuracy = engine_state.primitive_accuracy
            total_evals = engine_state.total_evals
            last_accuracy = engine_state.last_eval_accuracy
        except Exception:
            primitive_accuracy = {}
            total_evals = 0
            last_accuracy = None

        # From curriculum adapter
        curriculum_state = self.curriculum.get_state()
        if curriculum_state:
            skill_state = curriculum_state.get_skill(skill_id)
            recent_accuracy = skill_state.recent_accuracy()
            accuracy_history = skill_state.accuracy_history[-10:]
        else:
            recent_accuracy = 0.0
            accuracy_history = []

        return {
            "skill_id": skill_id,
            "current_level": curriculum_level,
            "training_level": curriculum_level + 1,
            "recent_accuracy": recent_accuracy,
            "last_eval_accuracy": last_accuracy,
            "total_evals": total_evals,
            "primitive_accuracy": primitive_accuracy,
            "accuracy_history": accuracy_history,
        }

    def list_skills(self) -> List[str]:
        """List all available skills."""
        return self.engine.list_skills()


# Singleton instance
_bridge: Optional[SkillEngineBridge] = None


def get_skill_engine_bridge() -> SkillEngineBridge:
    """Get the singleton SkillEngineBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = SkillEngineBridge()
    return _bridge


def reset_skill_engine_bridge():
    """Reset the singleton (for testing)."""
    global _bridge
    _bridge = None
