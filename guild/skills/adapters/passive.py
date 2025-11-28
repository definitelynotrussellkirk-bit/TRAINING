"""
PassiveAdapter - wraps PassiveModule for eval generation and scoring.

This adapter connects existing PassiveModule classes to the Skill interface,
allowing skills to generate eval problems and score model answers.
"""

from typing import Optional, Any
import logging

from guild.passives.base import PassiveModule
from guild.skills.eval_types import (
    EvalProblem,
    EvalBatch,
    EvalResult,
    EvalResultItem,
)

logger = logging.getLogger(__name__)


class PassiveAdapter:
    """
    Adapts PassiveModule to eval generation/scoring interface.

    Wraps an existing PassiveModule to provide:
    - generate_eval_batch(): Create EvalBatch from passive problems
    - score_eval(): Score model answers using passive.check_answer

    Example:
        from guild.passives.arithmetic import ArithmeticPassive

        passive = ArithmeticPassive()
        adapter = PassiveAdapter(passive)

        batch = adapter.generate_eval_batch("arithmetic", level=1, count=5)
        result = adapter.score_eval(batch, model_answers)
    """

    def __init__(self, passive: PassiveModule):
        """
        Initialize passive adapter.

        Args:
            passive: PassiveModule instance to wrap
        """
        self.passive = passive

    @property
    def id(self) -> str:
        """Get passive ID."""
        return self.passive.id

    @property
    def version(self) -> str:
        """Get passive version."""
        return self.passive.version

    def generate_eval_batch(
        self,
        skill_id: str,
        level: int,
        count: int,
        seed: Optional[int] = None
    ) -> EvalBatch:
        """
        Generate eval problems via passive.

        Calls passive.generate_problems() and wraps results in EvalBatch.

        Args:
            skill_id: Skill this batch is for
            level: Skill level (passed to batch metadata, not used by most passives)
            count: Number of problems to generate
            seed: Optional random seed for reproducibility

        Returns:
            EvalBatch containing problems for evaluation
        """
        logger.debug(f"Generating {count} eval problems from passive {self.passive.id}")

        # Generate raw problems from passive
        problems_raw = self.passive.generate_problems(count=count, seed=seed)

        # Convert to EvalProblem format
        problems = []
        for p in problems_raw:
            problems.append(EvalProblem(
                prompt=p["prompt"],
                expected=p["expected"],
                primitive_id=p.get("primitive_id") or p.get("type"),
                metadata={
                    k: v for k, v in p.items()
                    if k not in ("prompt", "expected", "type", "primitive_id")
                },
            ))

        return EvalBatch(
            skill_id=skill_id,
            level=level,
            problems=problems,
            metadata={
                "passive_id": self.passive.id,
                "passive_version": self.passive.version,
            },
        )

    def score_eval(
        self,
        batch: EvalBatch,
        model_answers: list[str]
    ) -> EvalResult:
        """
        Score model answers using passive.check_answer.

        Args:
            batch: The eval batch that was given to the model
            model_answers: Model's responses (same order as batch.problems)

        Returns:
            EvalResult with accuracy and per-primitive breakdown

        Raises:
            ValueError: If answer count doesn't match problem count
        """
        if len(model_answers) != len(batch.problems):
            raise ValueError(
                f"Answer count mismatch: {len(model_answers)} answers "
                f"for {len(batch.problems)} problems"
            )

        logger.debug(f"Scoring {len(model_answers)} answers with passive {self.passive.id}")

        # Score each problem
        items = []
        per_primitive: dict[str, list[bool]] = {}

        for problem, answer in zip(batch.problems, model_answers):
            is_correct = self.passive.check_answer(problem.expected, answer)

            items.append(EvalResultItem(
                problem=problem,
                model_answer=answer,
                is_correct=is_correct,
                primitive_id=problem.primitive_id,
                score=1.0 if is_correct else 0.0,
            ))

            # Track per-primitive results
            prim = problem.primitive_id or "unknown"
            per_primitive.setdefault(prim, []).append(is_correct)

        # Calculate accuracies
        total_correct = sum(1 for item in items if item.is_correct)
        accuracy = total_correct / len(items) if items else 0.0

        per_primitive_accuracy = {
            prim: sum(vals) / len(vals)
            for prim, vals in per_primitive.items()
        }

        logger.debug(f"Scored: {total_correct}/{len(items)} correct ({accuracy:.1%})")

        return EvalResult(
            accuracy=accuracy,
            per_primitive_accuracy=per_primitive_accuracy,
            num_examples=len(items),
            items=items,
            metadata={
                "skill_id": batch.skill_id,
                "level": batch.level,
                "passive_id": self.passive.id,
                "passive_version": self.passive.version,
            },
        )

    def __repr__(self) -> str:
        return f"PassiveAdapter({self.passive!r})"
