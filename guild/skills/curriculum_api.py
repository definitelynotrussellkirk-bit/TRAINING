"""
Curriculum API Extension
========================

Extends the Skill API contract with curriculum-specific endpoints
for evaluating model outputs, assessing difficulty, and progression.

Usage:
    from guild.skills.curriculum_api import CurriculumClient

    client = CurriculumClient("binary", "http://localhost:8090")

    # Evaluate model answers
    result = client.evaluate(
        level=5,
        problems=[
            {"prompt": "...", "expected": "42", "model_answer": "41"}
        ]
    )

    # Get difficulty metrics
    difficulty = client.difficulty(level=5)

    # Get progression recommendation
    next_level = client.suggest_next_level(
        current_level=5,
        recent_accuracy=[0.8, 0.9, 0.85]
    )

API Endpoints:
    POST /evaluate       - Score model answers against expected
    GET  /difficulty     - Get difficulty metrics for a level
    POST /next-level     - Suggest next level based on performance
    POST /generate-eval  - Generate fresh eval batch (not cached)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence
import requests
import logging

from guild.skills.contract import SkillClient, EvalRequirements

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class EvaluationProblem:
    """Single problem to evaluate."""
    prompt: str
    expected: str
    model_answer: str
    primitive_id: Optional[str] = None  # For per-primitive tracking
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result from evaluating a single problem."""
    is_correct: bool
    prompt: str
    expected: str
    model_answer: str
    score: float = 0.0  # 0.0-1.0 for partial credit
    primitive_id: Optional[str] = None
    feedback: Optional[str] = None  # Why wrong (if incorrect)


@dataclass
class BatchEvaluationResult:
    """Aggregated evaluation result."""
    accuracy: float  # 0.0-1.0
    correct_count: int
    total_count: int
    per_primitive_accuracy: dict[str, float] = field(default_factory=dict)
    results: list[EvaluationResult] = field(default_factory=list)
    passed: bool = False  # Did it meet pass_threshold?
    metadata: dict = field(default_factory=dict)


@dataclass
class DifficultyMetrics:
    """Difficulty metrics for a level."""
    level: int
    estimated_difficulty: float  # 0.0-1.0 (1.0 = hardest)
    complexity_score: float      # Computed from level params
    avg_tokens: int              # Average token count
    primitives: list[str]        # Primitives exercised at this level
    recommended_samples: int     # Recommended training samples
    metadata: dict = field(default_factory=dict)


@dataclass
class ProgressionRecommendation:
    """Recommendation for next level."""
    action: str                  # "advance", "stay", "retreat"
    current_level: int
    recommended_level: int
    confidence: float            # 0.0-1.0
    reason: str                  # Human-readable explanation
    weak_primitives: list[str]   # Primitives needing work
    metadata: dict = field(default_factory=dict)


# =============================================================================
# CURRICULUM API PROTOCOL
# =============================================================================

class CurriculumAPIContract(Protocol):
    """
    Protocol for curriculum-specific skill API endpoints.

    Extends base SkillAPIContract with:
        POST /evaluate       - Score model answers
        GET  /difficulty     - Get difficulty metrics
        POST /next-level     - Get progression recommendation
        POST /generate-eval  - Generate fresh eval problems
    """

    def evaluate(
        self,
        level: int,
        problems: list[EvaluationProblem]
    ) -> BatchEvaluationResult:
        """Score model answers against expected."""
        ...

    def difficulty(self, level: int) -> DifficultyMetrics:
        """Get difficulty metrics for a level."""
        ...

    def suggest_next_level(
        self,
        current_level: int,
        recent_accuracy: list[float]
    ) -> ProgressionRecommendation:
        """Get progression recommendation based on performance."""
        ...

    def generate_eval(
        self,
        level: int,
        count: int = 5
    ) -> list[dict]:
        """Generate fresh eval problems (not cached)."""
        ...


# =============================================================================
# CURRICULUM CLIENT
# =============================================================================

class CurriculumClient(SkillClient):
    """
    Extended skill client with curriculum endpoints.

    Inherits all base SkillClient methods (info, levels, sample, etc.)
    and adds curriculum-specific methods.

    Usage:
        client = CurriculumClient("binary", "http://localhost:8090")

        # Generate training data
        batch = client.sample(level=5, count=100)

        # Evaluate model answers
        result = client.evaluate(level=5, problems=[...])

        # Get difficulty info
        difficulty = client.difficulty(level=5)

        # Get progression recommendation
        next_level = client.suggest_next_level(
            current_level=5,
            recent_accuracy=[0.8, 0.9, 0.85]
        )
    """

    def evaluate(
        self,
        level: int,
        problems: list[EvaluationProblem] | list[dict]
    ) -> BatchEvaluationResult:
        """
        Evaluate model answers against expected answers.

        Args:
            level: Skill level
            problems: List of problems with prompt, expected, and model_answer

        Returns:
            BatchEvaluationResult with accuracy and per-problem results

        Request format:
            POST /evaluate
            {
                "level": 5,
                "problems": [
                    {"prompt": "...", "expected": "42", "model_answer": "41"},
                    ...
                ]
            }

        Response format:
            {
                "accuracy": 0.8,
                "correct_count": 4,
                "total_count": 5,
                "per_primitive_accuracy": {"add_binary": 1.0, "mul_binary": 0.5},
                "results": [
                    {"is_correct": true, "prompt": "...", "expected": "42", "model_answer": "42"},
                    ...
                ],
                "passed": true
            }
        """
        # Convert to dicts if needed
        problem_dicts = []
        for p in problems:
            if isinstance(p, EvaluationProblem):
                problem_dicts.append({
                    "prompt": p.prompt,
                    "expected": p.expected,
                    "model_answer": p.model_answer,
                    "primitive_id": p.primitive_id,
                    "metadata": p.metadata,
                })
            else:
                problem_dicts.append(p)

        try:
            r = requests.post(
                f"{self.api_url}/evaluate",
                json={"level": level, "problems": problem_dicts},
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()

            # Parse results
            results = []
            for res in data.get("results", []):
                results.append(EvaluationResult(
                    is_correct=res.get("is_correct", False),
                    prompt=res.get("prompt", ""),
                    expected=res.get("expected", ""),
                    model_answer=res.get("model_answer", ""),
                    score=res.get("score", 1.0 if res.get("is_correct") else 0.0),
                    primitive_id=res.get("primitive_id"),
                    feedback=res.get("feedback"),
                ))

            return BatchEvaluationResult(
                accuracy=data.get("accuracy", 0.0),
                correct_count=data.get("correct_count", 0),
                total_count=data.get("total_count", len(problems)),
                per_primitive_accuracy=data.get("per_primitive_accuracy", {}),
                results=results,
                passed=data.get("passed", False),
                metadata=data.get("metadata", {}),
            )

        except requests.RequestException as e:
            logger.warning(f"Curriculum API /evaluate failed: {e}")
            # Fall back to local evaluation
            return self._evaluate_local(level, problem_dicts)

    def _evaluate_local(
        self,
        level: int,
        problems: list[dict]
    ) -> BatchEvaluationResult:
        """Local fallback evaluation using exact string match."""
        results = []
        correct = 0

        for p in problems:
            expected = str(p.get("expected", "")).strip()
            model_answer = str(p.get("model_answer", "")).strip()
            is_correct = expected == model_answer

            if is_correct:
                correct += 1

            results.append(EvaluationResult(
                is_correct=is_correct,
                prompt=p.get("prompt", ""),
                expected=expected,
                model_answer=model_answer,
                score=1.0 if is_correct else 0.0,
                primitive_id=p.get("primitive_id"),
                feedback=None if is_correct else f"Expected '{expected}', got '{model_answer}'",
            ))

        total = len(problems)
        accuracy = correct / total if total > 0 else 0.0

        # Get pass threshold
        req = self.eval_requirements(level)
        passed = accuracy >= req.pass_threshold

        return BatchEvaluationResult(
            accuracy=accuracy,
            correct_count=correct,
            total_count=total,
            per_primitive_accuracy={},  # Can't compute without primitive info
            results=results,
            passed=passed,
            metadata={"fallback": True},
        )

    def difficulty(self, level: int) -> DifficultyMetrics:
        """
        Get difficulty metrics for a level.

        Args:
            level: Skill level to query

        Returns:
            DifficultyMetrics with complexity scores and recommendations

        Request format:
            GET /difficulty?level=5

        Response format:
            {
                "level": 5,
                "estimated_difficulty": 0.5,
                "complexity_score": 0.6,
                "avg_tokens": 150,
                "primitives": ["add_binary", "mul_binary"],
                "recommended_samples": 500
            }
        """
        try:
            r = requests.get(
                f"{self.api_url}/difficulty",
                params={"level": level},
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()

            return DifficultyMetrics(
                level=data.get("level", level),
                estimated_difficulty=data.get("estimated_difficulty", level / 30),
                complexity_score=data.get("complexity_score", 0.0),
                avg_tokens=data.get("avg_tokens", 100),
                primitives=data.get("primitives", []),
                recommended_samples=data.get("recommended_samples", 500),
                metadata=data.get("metadata", {}),
            )

        except requests.RequestException as e:
            logger.warning(f"Curriculum API /difficulty failed: {e}")
            # Fallback: estimate from level
            return self._difficulty_local(level)

    def _difficulty_local(self, level: int) -> DifficultyMetrics:
        """Local fallback for difficulty estimation."""
        # Get max level from info
        try:
            info = self.info()
            max_level = info.max_level
        except Exception:
            max_level = 30

        # Linear difficulty estimate
        estimated = level / max_level

        return DifficultyMetrics(
            level=level,
            estimated_difficulty=min(estimated, 1.0),
            complexity_score=level * 0.1,
            avg_tokens=50 + level * 10,
            primitives=[],
            recommended_samples=max(100, 500 - level * 10),
            metadata={"fallback": True},
        )

    def suggest_next_level(
        self,
        current_level: int,
        recent_accuracy: list[float],
        consecutive_passes_required: int = 3,
    ) -> ProgressionRecommendation:
        """
        Get progression recommendation based on recent performance.

        Args:
            current_level: Current skill level
            recent_accuracy: List of recent accuracy scores (newest last)
            consecutive_passes_required: How many passes needed to advance

        Returns:
            ProgressionRecommendation with action and reasoning

        Request format:
            POST /next-level
            {
                "current_level": 5,
                "recent_accuracy": [0.8, 0.9, 0.85],
                "consecutive_passes_required": 3
            }

        Response format:
            {
                "action": "advance",
                "current_level": 5,
                "recommended_level": 6,
                "confidence": 0.9,
                "reason": "3 consecutive passes above 80%",
                "weak_primitives": []
            }
        """
        try:
            r = requests.post(
                f"{self.api_url}/next-level",
                json={
                    "current_level": current_level,
                    "recent_accuracy": recent_accuracy,
                    "consecutive_passes_required": consecutive_passes_required,
                },
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()

            return ProgressionRecommendation(
                action=data.get("action", "stay"),
                current_level=data.get("current_level", current_level),
                recommended_level=data.get("recommended_level", current_level),
                confidence=data.get("confidence", 0.5),
                reason=data.get("reason", ""),
                weak_primitives=data.get("weak_primitives", []),
                metadata=data.get("metadata", {}),
            )

        except requests.RequestException as e:
            logger.warning(f"Curriculum API /next-level failed: {e}")
            return self._suggest_next_level_local(
                current_level, recent_accuracy, consecutive_passes_required
            )

    def _suggest_next_level_local(
        self,
        current_level: int,
        recent_accuracy: list[float],
        consecutive_passes_required: int = 3,
    ) -> ProgressionRecommendation:
        """Local fallback for progression recommendation."""
        # Get pass threshold
        req = self.eval_requirements(current_level)
        threshold = req.pass_threshold

        # Check recent accuracy
        if len(recent_accuracy) < consecutive_passes_required:
            return ProgressionRecommendation(
                action="stay",
                current_level=current_level,
                recommended_level=current_level,
                confidence=0.3,
                reason=f"Need {consecutive_passes_required} evals, have {len(recent_accuracy)}",
                weak_primitives=[],
                metadata={"fallback": True},
            )

        # Check for consecutive passes
        recent = recent_accuracy[-consecutive_passes_required:]
        passes = sum(1 for a in recent if a >= threshold)
        avg_accuracy = sum(recent) / len(recent)

        if passes == consecutive_passes_required:
            # All recent evals passed - advance
            return ProgressionRecommendation(
                action="advance",
                current_level=current_level,
                recommended_level=current_level + 1,
                confidence=min(avg_accuracy, 0.95),
                reason=f"{consecutive_passes_required} consecutive passes above {threshold:.0%}",
                weak_primitives=[],
                metadata={"fallback": True},
            )
        elif avg_accuracy < 0.5 and current_level > 1:
            # Struggling - retreat
            return ProgressionRecommendation(
                action="retreat",
                current_level=current_level,
                recommended_level=current_level - 1,
                confidence=0.7,
                reason=f"Average accuracy {avg_accuracy:.0%} below 50%",
                weak_primitives=[],
                metadata={"fallback": True},
            )
        else:
            # Stay at current level
            return ProgressionRecommendation(
                action="stay",
                current_level=current_level,
                recommended_level=current_level,
                confidence=0.6,
                reason=f"{passes}/{consecutive_passes_required} consecutive passes",
                weak_primitives=[],
                metadata={"fallback": True},
            )

    def generate_eval(self, level: int, count: int = 5) -> list[dict]:
        """
        Generate fresh eval problems (not from cache).

        Unlike get_eval() which may return cached problems,
        this always generates new problems.

        Args:
            level: Skill level
            count: Number of problems

        Returns:
            List of eval problems with prompt and expected fields

        Request format:
            POST /generate-eval
            {"level": 5, "count": 5}

        Response format:
            {
                "problems": [
                    {"prompt": "...", "expected": "..."},
                    ...
                ]
            }
        """
        try:
            r = requests.post(
                f"{self.api_url}/generate-eval",
                json={"level": level, "count": count},
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()
            return data.get("problems", [])

        except requests.RequestException:
            # Fallback to regular sample generation
            batch = self.sample(level=level, count=count)
            return [
                {
                    "prompt": s.messages[0]["content"] if s.messages else "",
                    "expected": s.messages[1]["content"] if len(s.messages) > 1 else "",
                    "metadata": s.metadata,
                }
                for s in batch.samples
            ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_curriculum_client(skill_id: str) -> CurriculumClient:
    """
    Get a CurriculumClient for a skill.

    Uses skill registry to find API URL.
    """
    # Default ports from CLAUDE.md
    skill_ports = {
        "sy": 8080,
        "syllo": 8080,
        "bin": 8090,
        "binary": 8090,
    }

    port = skill_ports.get(skill_id, 8080)
    api_url = f"http://localhost:{port}"

    return CurriculumClient(skill_id, api_url)


def evaluate_batch(
    skill_id: str,
    level: int,
    prompts: list[str],
    expected: list[str],
    model_answers: list[str],
) -> BatchEvaluationResult:
    """
    Convenience function to evaluate a batch of model answers.

    Args:
        skill_id: Skill ID (e.g., "binary", "sy")
        level: Skill level
        prompts: List of prompts
        expected: List of expected answers
        model_answers: List of model's answers

    Returns:
        BatchEvaluationResult
    """
    client = get_curriculum_client(skill_id)

    problems = [
        EvaluationProblem(prompt=p, expected=e, model_answer=m)
        for p, e, m in zip(prompts, expected, model_answers)
    ]

    return client.evaluate(level, problems)
