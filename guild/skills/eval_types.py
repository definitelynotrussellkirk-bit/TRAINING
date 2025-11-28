"""
Evaluation types for the Skill Engine.

These types represent evaluation problems, batches, and results.
They provide a standard interface for running evals across all skills.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvalProblem:
    """
    Single evaluation problem.

    This is what gets sent to the model for evaluation.

    Attributes:
        prompt: The question/task text for the model
        expected: The correct answer
        primitive_id: Which primitive this tests (for tracking)
        metadata: Additional problem info (difficulty, tags, etc.)
    """
    prompt: str
    expected: str
    primitive_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "prompt": self.prompt,
            "expected": self.expected,
            "primitive_id": self.primitive_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalProblem":
        """Deserialize from dict."""
        return cls(
            prompt=data["prompt"],
            expected=data["expected"],
            primitive_id=data.get("primitive_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalBatch:
    """
    Collection of problems for evaluation.

    A batch is what gets generated for a single eval run.

    Attributes:
        skill_id: Which skill this batch is for
        level: Skill level these problems target
        problems: List of eval problems
        metadata: Batch-level metadata (passive version, timestamp, etc.)
    """
    skill_id: str
    level: int
    problems: list[EvalProblem]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)

    @property
    def prompts(self) -> list[str]:
        """Get just the prompts for sending to model."""
        return [p.prompt for p in self.problems]

    @property
    def expected_answers(self) -> list[str]:
        """Get expected answers for scoring."""
        return [p.expected for p in self.problems]

    @property
    def primitive_ids(self) -> list[Optional[str]]:
        """Get primitive IDs for per-primitive tracking."""
        return [p.primitive_id for p in self.problems]

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "skill_id": self.skill_id,
            "level": self.level,
            "problems": [p.to_dict() for p in self.problems],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalBatch":
        """Deserialize from dict."""
        return cls(
            skill_id=data["skill_id"],
            level=data["level"],
            problems=[EvalProblem.from_dict(p) for p in data["problems"]],
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalResultItem:
    """
    Result for a single problem in an eval.

    Captures the problem, model's answer, and whether it was correct.
    """
    problem: EvalProblem
    model_answer: str
    is_correct: bool
    primitive_id: Optional[str] = None
    score: float = 0.0  # For partial credit scenarios

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "problem": self.problem.to_dict(),
            "model_answer": self.model_answer,
            "is_correct": self.is_correct,
            "primitive_id": self.primitive_id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResultItem":
        """Deserialize from dict."""
        return cls(
            problem=EvalProblem.from_dict(data["problem"]),
            model_answer=data["model_answer"],
            is_correct=data["is_correct"],
            primitive_id=data.get("primitive_id"),
            score=data.get("score", 1.0 if data["is_correct"] else 0.0),
        )


@dataclass
class EvalResult:
    """
    Aggregated evaluation results.

    Contains overall accuracy plus per-primitive breakdown.

    Attributes:
        accuracy: Overall accuracy (0.0 - 1.0)
        per_primitive_accuracy: Accuracy broken down by primitive
        num_examples: Total number of problems evaluated
        items: Individual result items (for detailed analysis)
        metadata: Additional info (skill_id, level, timestamp, etc.)
    """
    accuracy: float
    per_primitive_accuracy: dict[str, float]
    num_examples: int
    items: list[EvalResultItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate after initialization."""
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"Accuracy must be 0-1, got {self.accuracy}")

    @property
    def num_correct(self) -> int:
        """Number of correct answers."""
        return sum(1 for item in self.items if item.is_correct)

    @property
    def num_incorrect(self) -> int:
        """Number of incorrect answers."""
        return self.num_examples - self.num_correct

    @property
    def correct_rate_str(self) -> str:
        """Human-readable correct rate (e.g., '4/5')."""
        return f"{self.num_correct}/{self.num_examples}"

    @property
    def weakest_primitive(self) -> Optional[str]:
        """Primitive with lowest accuracy, or None if no primitives."""
        if not self.per_primitive_accuracy:
            return None
        return min(self.per_primitive_accuracy.keys(),
                   key=lambda k: self.per_primitive_accuracy[k])

    @property
    def strongest_primitive(self) -> Optional[str]:
        """Primitive with highest accuracy, or None if no primitives."""
        if not self.per_primitive_accuracy:
            return None
        return max(self.per_primitive_accuracy.keys(),
                   key=lambda k: self.per_primitive_accuracy[k])

    def passed(self, threshold: float) -> bool:
        """Check if accuracy meets or exceeds threshold."""
        return self.accuracy >= threshold

    def primitive_passed(self, primitive_id: str, threshold: float) -> bool:
        """Check if a specific primitive meets threshold."""
        acc = self.per_primitive_accuracy.get(primitive_id, 0.0)
        return acc >= threshold

    def get_failed_primitives(self, threshold: float) -> list[str]:
        """Get list of primitives below threshold."""
        return [
            prim for prim, acc in self.per_primitive_accuracy.items()
            if acc < threshold
        ]

    def get_passed_primitives(self, threshold: float) -> list[str]:
        """Get list of primitives at or above threshold."""
        return [
            prim for prim, acc in self.per_primitive_accuracy.items()
            if acc >= threshold
        ]

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "accuracy": self.accuracy,
            "per_primitive_accuracy": self.per_primitive_accuracy,
            "num_examples": self.num_examples,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        """Deserialize from dict."""
        return cls(
            accuracy=data["accuracy"],
            per_primitive_accuracy=data.get("per_primitive_accuracy", {}),
            num_examples=data["num_examples"],
            items=[EvalResultItem.from_dict(i) for i in data.get("items", [])],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_items(
        cls,
        items: list[EvalResultItem],
        metadata: Optional[dict] = None
    ) -> "EvalResult":
        """
        Construct EvalResult from list of items.

        Automatically computes accuracy and per-primitive breakdown.
        """
        if not items:
            return cls(
                accuracy=0.0,
                per_primitive_accuracy={},
                num_examples=0,
                items=[],
                metadata=metadata or {},
            )

        # Compute overall accuracy
        num_correct = sum(1 for item in items if item.is_correct)
        accuracy = num_correct / len(items)

        # Compute per-primitive accuracy
        primitive_results: dict[str, list[bool]] = {}
        for item in items:
            prim = item.primitive_id or "unknown"
            primitive_results.setdefault(prim, []).append(item.is_correct)

        per_primitive_accuracy = {
            prim: sum(results) / len(results)
            for prim, results in primitive_results.items()
        }

        return cls(
            accuracy=accuracy,
            per_primitive_accuracy=per_primitive_accuracy,
            num_examples=len(items),
            items=items,
            metadata=metadata or {},
        )


# =============================================================================
# Helper functions for creating eval results
# =============================================================================

def score_batch(
    batch: EvalBatch,
    model_answers: list[str],
    check_fn: callable,
) -> EvalResult:
    """
    Score a batch of eval problems.

    Args:
        batch: The eval batch with problems
        model_answers: Model's responses (same order as batch.problems)
        check_fn: Function (expected, got) -> bool for checking correctness

    Returns:
        EvalResult with accuracy and per-primitive breakdown

    Raises:
        ValueError: If len(model_answers) != len(batch.problems)
    """
    if len(model_answers) != len(batch.problems):
        raise ValueError(
            f"Answer count mismatch: {len(model_answers)} answers "
            f"for {len(batch.problems)} problems"
        )

    items = []
    for problem, answer in zip(batch.problems, model_answers):
        is_correct = check_fn(problem.expected, answer)
        items.append(EvalResultItem(
            problem=problem,
            model_answer=answer,
            is_correct=is_correct,
            primitive_id=problem.primitive_id,
            score=1.0 if is_correct else 0.0,
        ))

    return EvalResult.from_items(
        items,
        metadata={
            "skill_id": batch.skill_id,
            "level": batch.level,
            **batch.metadata,
        }
    )
