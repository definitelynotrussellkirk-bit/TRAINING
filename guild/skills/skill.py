"""
Skill ABC - The core abstraction for trainable skills.

A Skill is THE interface for all skill operations:
- Training data generation
- Eval problem generation
- Answer scoring
- State updates (XP, leveling)

This is the unified interface that Guild, DataManager, Monitoring,
and UI all talk to.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time

from guild.skills.types import SkillConfig, SkillState
from guild.skills.primitives import PrimitiveId
from guild.skills.eval_types import EvalBatch, EvalResult


class Skill(ABC):
    """
    Abstract base class for trainable skills.

    A Skill knows how to:
    1. Generate training examples at a given level
    2. Generate eval problems at a given level
    3. Score model answers against expected answers
    4. Update skill state based on eval results

    Subclasses implement specific skill behaviors. The standard
    implementation is CompositeSkill which combines a generator
    (for training) with a passive (for eval).

    Example:
        skill = engine.get("binary")

        # Training
        training_batch = skill.generate_training_batch(level=5, count=100)

        # Eval
        eval_batch = skill.generate_eval_batch(level=5, count=5)
        # ... run model on eval_batch.prompts ...
        result = skill.score_eval(eval_batch, model_answers)

        # Update state
        new_state = skill.update_state_from_eval(old_state, result)
    """

    def __init__(self, config: SkillConfig):
        """
        Initialize skill with configuration.

        Args:
            config: SkillConfig loaded from YAML
        """
        self.config = config

    @property
    def id(self) -> str:
        """Unique identifier for this skill."""
        return self.config.id

    @property
    def name(self) -> str:
        """Display name for this skill."""
        return self.config.name

    @property
    def max_level(self) -> int:
        """Maximum level for this skill."""
        return self.config.max_level

    @property
    def primitives(self) -> list[PrimitiveId]:
        """
        List of primitives this skill covers.

        Override in subclasses to provide skill-specific primitives.
        Default returns empty list.
        """
        return []

    @abstractmethod
    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        """
        Generate training examples at given level.

        Args:
            level: Skill level (1 to max_level)
            count: Number of examples to generate
            seed: Optional random seed for reproducibility

        Returns:
            List of training examples in inbox format:
            [{"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
              "metadata": {...}}, ...]
        """
        ...

    @abstractmethod
    def generate_eval_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> EvalBatch:
        """
        Generate evaluation problems at given level.

        Args:
            level: Skill level (1 to max_level)
            count: Number of problems to generate
            seed: Optional random seed for reproducibility

        Returns:
            EvalBatch containing problems for evaluation
        """
        ...

    @abstractmethod
    def score_eval(
        self,
        batch: EvalBatch,
        model_answers: list[str]
    ) -> EvalResult:
        """
        Score model answers against expected answers.

        Args:
            batch: The eval batch that was given to the model
            model_answers: Model's responses (same order as batch.problems)

        Returns:
            EvalResult with accuracy and per-primitive breakdown
        """
        ...

    def update_state_from_eval(
        self,
        state: SkillState,
        result: EvalResult
    ) -> SkillState:
        """
        Update skill state based on eval results.

        Default policy:
        - Record eval stats (total_evals, total_samples_seen)
        - Add XP based on accuracy
        - Record result for rolling accuracy
        - Level up if accuracy threshold met

        Can be overridden for skill-specific leveling policies.

        Args:
            state: Current skill state
            result: Eval result to incorporate

        Returns:
            Updated skill state (may be same object, mutated)
        """
        # Update eval tracking
        state.total_evals += 1
        state.total_samples_seen += result.num_examples
        state.last_eval_accuracy = result.accuracy
        state.last_eval_timestamp = time.time()

        # XP gain: base XP * multiplier * accuracy
        base_xp = 10
        xp_gain = int(base_xp * self.config.xp_multiplier * result.accuracy)
        state.xp_total += xp_gain

        # Record for rolling accuracy
        passed = result.accuracy >= self.config.get_threshold(state.level)
        state.record_result(passed)

        # Store per-primitive accuracy
        state.primitive_accuracy = result.per_primitive_accuracy

        # Level up check: need rolling accuracy above threshold
        threshold = self.config.get_threshold(state.level)
        if passed and state.accuracy >= threshold:
            if state.level < self.config.max_level:
                state.record_level_up()

        return state

    def get_level_threshold(self, level: int) -> float:
        """Get accuracy threshold for a level."""
        return self.config.get_threshold(level)

    def can_advance(self, state: SkillState) -> bool:
        """Check if state meets criteria to advance to next level."""
        if state.level >= self.config.max_level:
            return False

        threshold = self.get_level_threshold(state.level)
        return state.accuracy >= threshold

    def __repr__(self) -> str:
        return f"<Skill:{self.id} (max_level={self.max_level})>"


# =============================================================================
# Concrete implementations
# =============================================================================

class GeneratorOnlySkill(Skill):
    """
    Skill that only has a generator (no passive for eval).

    Used when skill API provides training data but no local eval.
    Eval methods raise NotImplementedError.
    """

    def __init__(self, config: SkillConfig, generator: "GeneratorAdapter"):
        super().__init__(config)
        self.generator = generator

    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        return self.generator.generate_training_batch(level, count, seed)

    def generate_eval_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> EvalBatch:
        raise NotImplementedError(
            f"Skill '{self.id}' has no passive configured for eval. "
            f"Add a passive to configs/skills/{self.id}.yaml"
        )

    def score_eval(
        self,
        batch: EvalBatch,
        model_answers: list[str]
    ) -> EvalResult:
        raise NotImplementedError(
            f"Skill '{self.id}' has no passive configured for scoring."
        )


class PassiveOnlySkill(Skill):
    """
    Skill that only has a passive (no generator for training).

    Used for eval-only skills or when training data comes from
    elsewhere. Training methods raise NotImplementedError.
    """

    def __init__(self, config: SkillConfig, passive: "PassiveAdapter"):
        super().__init__(config)
        self.passive = passive

    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        raise NotImplementedError(
            f"Skill '{self.id}' has no generator configured for training. "
            f"Add an API config to configs/skills/{self.id}.yaml"
        )

    def generate_eval_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> EvalBatch:
        return self.passive.generate_eval_batch(
            skill_id=self.id,
            level=level,
            count=count,
            seed=seed,
        )

    def score_eval(
        self,
        batch: EvalBatch,
        model_answers: list[str]
    ) -> EvalResult:
        return self.passive.score_eval(batch, model_answers)


# Note: CompositeSkill is defined in composite.py to avoid circular imports
# It combines both GeneratorAdapter and PassiveAdapter
