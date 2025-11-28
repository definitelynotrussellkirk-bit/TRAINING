"""
CompositeSkill - combines Generator + Passive into a full Skill.

This is the standard skill implementation that most skills will use.
It combines:
- GeneratorAdapter for training data generation (via skill API)
- PassiveAdapter for eval generation and scoring (via local passive)
"""

from typing import Optional
import logging

from guild.skills.types import SkillConfig
from guild.skills.skill import Skill
from guild.skills.primitives import PrimitiveId, PrimitiveMeta
from guild.skills.eval_types import EvalBatch, EvalResult
from guild.skills.adapters.generator import GeneratorAdapter
from guild.skills.adapters.passive import PassiveAdapter

logger = logging.getLogger(__name__)


class CompositeSkill(Skill):
    """
    Skill implementation using Generator (for training) + Passive (for eval).

    This is the standard skill type. It:
    - Uses GeneratorAdapter to call skill API for training data
    - Uses PassiveAdapter to generate eval problems and score answers

    Most skills should use this implementation.

    Example:
        config = load_skill_config("binary")
        generator = GeneratorAdapter("binary", "http://localhost:8090")
        passive = PassiveAdapter(binary_passive)

        skill = CompositeSkill(config, generator, passive)

        # Generate training data
        training_batch = skill.generate_training_batch(level=5, count=100)

        # Generate and score eval
        eval_batch = skill.generate_eval_batch(level=5, count=5)
        result = skill.score_eval(eval_batch, model_answers)
    """

    def __init__(
        self,
        config: SkillConfig,
        generator: GeneratorAdapter,
        passive: PassiveAdapter,
        primitives: Optional[list[PrimitiveId]] = None,
    ):
        """
        Initialize composite skill.

        Args:
            config: SkillConfig loaded from YAML
            generator: GeneratorAdapter for training data
            passive: PassiveAdapter for eval
            primitives: Optional list of primitives this skill covers
        """
        super().__init__(config)
        self.generator = generator
        self.passive = passive
        self._primitives = primitives or []

    @property
    def primitives(self) -> list[PrimitiveId]:
        """List of primitives this skill covers."""
        return self._primitives

    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        """
        Generate training examples via skill API.

        Args:
            level: Skill level (1 to max_level)
            count: Number of examples to generate
            seed: Optional random seed for reproducibility

        Returns:
            List of training examples in inbox format
        """
        logger.debug(f"CompositeSkill({self.id}): generating {count} training samples at level {level}")
        return self.generator.generate_training_batch(level, count, seed)

    def generate_eval_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> EvalBatch:
        """
        Generate eval problems via passive.

        Args:
            level: Skill level (1 to max_level)
            count: Number of problems to generate
            seed: Optional random seed for reproducibility

        Returns:
            EvalBatch containing problems for evaluation
        """
        logger.debug(f"CompositeSkill({self.id}): generating {count} eval problems at level {level}")
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
        """
        Score model answers using passive.

        Args:
            batch: The eval batch that was given to the model
            model_answers: Model's responses (same order as batch.problems)

        Returns:
            EvalResult with accuracy and per-primitive breakdown
        """
        logger.debug(f"CompositeSkill({self.id}): scoring {len(model_answers)} answers")
        return self.passive.score_eval(batch, model_answers)

    def health(self) -> dict:
        """
        Check health of both generator and passive.

        Returns:
            Dict with health status for each component
        """
        return {
            "skill_id": self.id,
            "generator_healthy": self.generator.health(),
            "passive_id": self.passive.id,
            "passive_version": self.passive.version,
        }

    def __repr__(self) -> str:
        return (
            f"CompositeSkill({self.id!r}, "
            f"generator={self.generator.api_url!r}, "
            f"passive={self.passive.id!r})"
        )


class LocalSkill(Skill):
    """
    Skill that uses only local passive for both training and eval.

    Useful for skills that don't have an external API server.
    Training samples are generated by converting eval problems
    to training format.
    """

    def __init__(
        self,
        config: SkillConfig,
        passive: PassiveAdapter,
        primitives: Optional[list[PrimitiveId]] = None,
    ):
        """
        Initialize local-only skill.

        Args:
            config: SkillConfig loaded from YAML
            passive: PassiveAdapter for both training and eval
            primitives: Optional list of primitives
        """
        super().__init__(config)
        self.passive = passive
        self._primitives = primitives or []

    @property
    def primitives(self) -> list[PrimitiveId]:
        return self._primitives

    def generate_training_batch(
        self, *,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> list[dict]:
        """
        Generate training from passive problems.

        Converts eval problems to training format by using
        the problem prompt and expected answer.
        """
        batch = self.passive.generate_eval_batch(
            skill_id=self.id,
            level=level,
            count=count,
            seed=seed,
        )

        # Convert to training format
        training_samples = []
        for problem in batch.problems:
            training_samples.append({
                "messages": [
                    {"role": "user", "content": problem.prompt},
                    {"role": "assistant", "content": problem.expected},
                ],
                "metadata": {
                    "skill_id": self.id,
                    "level": level,
                    "primitive_id": problem.primitive_id,
                    **problem.metadata,
                },
            })

        return training_samples

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

    def __repr__(self) -> str:
        return f"LocalSkill({self.id!r}, passive={self.passive.id!r})"
