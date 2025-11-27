"""Combat evaluator - determines CombatResult from model outputs."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, List

from guild.quests.types import CombatResult, QuestInstance


logger = logging.getLogger(__name__)


class MatchQuality(Enum):
    """Quality of answer match."""
    EXACT = "exact"           # Perfect match
    NORMALIZED = "normalized" # Match after normalization
    PARTIAL = "partial"       # Partial/fuzzy match
    WRONG = "wrong"           # Incorrect answer
    INVALID = "invalid"       # Cannot evaluate (malformed, etc.)


@dataclass
class EvaluationResult:
    """Result of evaluating a model response."""

    combat_result: CombatResult
    match_quality: MatchQuality

    model_answer: str = ""
    expected_answer: str = ""

    confidence: float = 1.0
    reasoning: str = ""

    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the answer was correct (HIT or better)."""
        return self.combat_result in [
            CombatResult.CRITICAL_HIT,
            CombatResult.HIT,
            CombatResult.GLANCING,
        ]


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    - Strip whitespace
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    """
    if not answer:
        return ""

    # Strip and lowercase
    result = answer.strip().lower()

    # Remove common punctuation
    result = re.sub(r'[.,!?;:\'"]+', '', result)

    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result)

    return result.strip()


def extract_answer(response: str, markers: Optional[List[str]] = None) -> str:
    """
    Extract the final answer from a model response.

    Handles common patterns:
    - "The answer is X"
    - "Answer: X"
    - "Therefore, X"
    - Just X (last meaningful word/phrase)
    """
    if not response:
        return ""

    text = response.strip()

    # Check for explicit markers
    default_markers = [
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"answer[:\s]+(.+?)(?:\.|$)",
        r"therefore[,:\s]+(.+?)(?:\.|$)",
        r"solution[:\s]+(.+?)(?:\.|$)",
        r"result[:\s]+(.+?)(?:\.|$)",
    ]

    patterns = markers or default_markers

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fall back to last line/sentence
    lines = text.split('\n')
    last_meaningful = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            last_meaningful = stripped
            break

    return last_meaningful


class BaseEvaluator(ABC):
    """
    Base class for combat evaluators.

    Subclass to implement specific evaluation logic.
    """

    @abstractmethod
    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        """
        Evaluate a model response against a quest.

        Args:
            quest: The quest being attempted
            response: Model's response

        Returns:
            EvaluationResult with combat outcome
        """
        pass

    def result_from_match(
        self,
        match_quality: MatchQuality,
        model_answer: str,
        expected_answer: str,
    ) -> CombatResult:
        """Convert match quality to combat result."""
        return {
            MatchQuality.EXACT: CombatResult.CRITICAL_HIT,
            MatchQuality.NORMALIZED: CombatResult.HIT,
            MatchQuality.PARTIAL: CombatResult.GLANCING,
            MatchQuality.WRONG: CombatResult.MISS,
            MatchQuality.INVALID: CombatResult.CRITICAL_MISS,
        }.get(match_quality, CombatResult.MISS)


def get_expected_answer(quest: QuestInstance) -> str:
    """Extract expected answer from quest, handling different formats."""
    if quest.expected is None:
        return ""

    if isinstance(quest.expected, str):
        return quest.expected

    if isinstance(quest.expected, dict):
        # Try common keys
        for key in ["answer", "expected", "correct", "value"]:
            if key in quest.expected:
                return str(quest.expected[key])

    return str(quest.expected)


def get_skill_id(quest: QuestInstance) -> str:
    """Get primary skill ID from quest."""
    if hasattr(quest, 'skill_id'):
        return quest.skill_id
    if quest.skills:
        return quest.skills[0]
    return ""


class ExactMatchEvaluator(BaseEvaluator):
    """
    Evaluator using exact string matching.

    Best for quests with single, unambiguous answers.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        answer_markers: Optional[List[str]] = None,
    ):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.answer_markers = answer_markers

    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        expected = get_expected_answer(quest)
        model_answer = extract_answer(response, self.answer_markers)

        # Prepare for comparison
        expected_cmp = expected
        model_cmp = model_answer

        if self.strip_whitespace:
            expected_cmp = expected_cmp.strip()
            model_cmp = model_cmp.strip()

        if not self.case_sensitive:
            expected_cmp = expected_cmp.lower()
            model_cmp = model_cmp.lower()

        # Determine match quality
        if model_cmp == expected_cmp:
            match_quality = MatchQuality.EXACT
        elif normalize_answer(model_cmp) == normalize_answer(expected_cmp):
            match_quality = MatchQuality.NORMALIZED
        elif expected_cmp in model_cmp or model_cmp in expected_cmp:
            match_quality = MatchQuality.PARTIAL
        else:
            match_quality = MatchQuality.WRONG

        combat_result = self.result_from_match(
            match_quality, model_answer, expected
        )

        return EvaluationResult(
            combat_result=combat_result,
            match_quality=match_quality,
            model_answer=model_answer,
            expected_answer=expected,
            reasoning=f"Exact match: {match_quality.value}",
        )


class MultipleChoiceEvaluator(BaseEvaluator):
    """
    Evaluator for multiple-choice questions.

    Accepts answer letter (A, B, C, D) or full answer text.
    """

    def __init__(
        self,
        choices_key: str = "choices",
        answer_key: str = "correct_index",
    ):
        self.choices_key = choices_key
        self.answer_key = answer_key

    def _get_quest_data(self, quest: QuestInstance, key: str, default=None):
        """Get data from quest context, metadata, or expected."""
        if hasattr(quest, 'context') and isinstance(quest.context, dict):
            if key in quest.context:
                return quest.context[key]
        if hasattr(quest, 'metadata') and isinstance(quest.metadata, dict):
            if key in quest.metadata:
                return quest.metadata[key]
        if hasattr(quest, 'expected') and isinstance(quest.expected, dict):
            if key in quest.expected:
                return quest.expected[key]
        return default

    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        # Get choices and correct answer from quest
        choices = self._get_quest_data(quest, self.choices_key, [])
        correct_idx = self._get_quest_data(quest, self.answer_key, 0)

        if not choices or correct_idx >= len(choices):
            return EvaluationResult(
                combat_result=CombatResult.CRITICAL_MISS,
                match_quality=MatchQuality.INVALID,
                model_answer="",
                expected_answer="",
                reasoning="Invalid quest configuration",
            )

        expected = choices[correct_idx]
        expected_letter = chr(ord('A') + correct_idx)

        # Extract model's answer
        model_answer = extract_answer(response)

        # Check for letter answer (A, B, C, D, etc.)
        letter_match = re.search(r'\b([A-D])\b', model_answer.upper())
        if letter_match:
            selected_letter = letter_match.group(1)
            if selected_letter == expected_letter:
                return EvaluationResult(
                    combat_result=CombatResult.HIT,
                    match_quality=MatchQuality.NORMALIZED,
                    model_answer=selected_letter,
                    expected_answer=expected_letter,
                    reasoning=f"Correct choice: {expected_letter}",
                )

        # Check for text answer
        normalized_model = normalize_answer(model_answer)
        normalized_expected = normalize_answer(expected)

        if normalized_model == normalized_expected:
            return EvaluationResult(
                combat_result=CombatResult.CRITICAL_HIT,
                match_quality=MatchQuality.EXACT,
                model_answer=model_answer,
                expected_answer=expected,
                reasoning="Exact text match",
            )

        # Check partial
        if normalized_expected in normalized_model:
            return EvaluationResult(
                combat_result=CombatResult.GLANCING,
                match_quality=MatchQuality.PARTIAL,
                model_answer=model_answer,
                expected_answer=expected,
                reasoning="Partial text match",
            )

        return EvaluationResult(
            combat_result=CombatResult.MISS,
            match_quality=MatchQuality.WRONG,
            model_answer=model_answer,
            expected_answer=f"{expected_letter}: {expected}",
            reasoning="Incorrect answer",
        )


class NumericEvaluator(BaseEvaluator):
    """
    Evaluator for numeric answers.

    Supports tolerance for floating-point comparisons.
    """

    def __init__(
        self,
        absolute_tolerance: float = 0.001,
        relative_tolerance: float = 0.01,
        accept_scientific: bool = True,
    ):
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.accept_scientific = accept_scientific

    def _parse_number(self, s: str) -> Optional[float]:
        """Parse a number from string."""
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,\s]', '', s.strip())

            # Handle scientific notation
            if self.accept_scientific:
                match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', cleaned)
                if match:
                    return float(match.group())

            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        expected_str = get_expected_answer(quest)
        model_answer = extract_answer(response)

        expected = self._parse_number(expected_str)
        model = self._parse_number(model_answer)

        if expected is None:
            return EvaluationResult(
                combat_result=CombatResult.CRITICAL_MISS,
                match_quality=MatchQuality.INVALID,
                model_answer=model_answer,
                expected_answer=expected_str,
                reasoning="Cannot parse expected answer as number",
            )

        if model is None:
            return EvaluationResult(
                combat_result=CombatResult.MISS,
                match_quality=MatchQuality.WRONG,
                model_answer=model_answer,
                expected_answer=expected_str,
                reasoning="Cannot parse model answer as number",
            )

        # Check exact
        if model == expected:
            return EvaluationResult(
                combat_result=CombatResult.CRITICAL_HIT,
                match_quality=MatchQuality.EXACT,
                model_answer=str(model),
                expected_answer=str(expected),
                reasoning="Exact numeric match",
            )

        # Check within tolerance
        abs_diff = abs(model - expected)
        rel_diff = abs_diff / max(abs(expected), 1e-10)

        if abs_diff <= self.absolute_tolerance or rel_diff <= self.relative_tolerance:
            return EvaluationResult(
                combat_result=CombatResult.HIT,
                match_quality=MatchQuality.NORMALIZED,
                model_answer=str(model),
                expected_answer=str(expected),
                reasoning=f"Within tolerance (diff={abs_diff:.6f})",
                metrics={"difference": abs_diff, "relative_diff": rel_diff},
            )

        # Check order of magnitude (for glancing)
        if expected != 0:
            magnitude_diff = abs(
                len(str(int(abs(model)))) - len(str(int(abs(expected))))
            )
            if magnitude_diff <= 1:
                return EvaluationResult(
                    combat_result=CombatResult.GLANCING,
                    match_quality=MatchQuality.PARTIAL,
                    model_answer=str(model),
                    expected_answer=str(expected),
                    reasoning=f"Same order of magnitude (diff={abs_diff:.2f})",
                    metrics={"difference": abs_diff},
                )

        return EvaluationResult(
            combat_result=CombatResult.MISS,
            match_quality=MatchQuality.WRONG,
            model_answer=str(model),
            expected_answer=str(expected),
            reasoning=f"Incorrect (diff={abs_diff:.2f})",
            metrics={"difference": abs_diff},
        )


class CustomEvaluator(BaseEvaluator):
    """
    Evaluator using a custom evaluation function.

    Allows arbitrary evaluation logic.
    """

    def __init__(
        self,
        eval_func: Callable[[QuestInstance, str], EvaluationResult],
    ):
        self.eval_func = eval_func

    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        return self.eval_func(quest, response)


# Evaluator registry
EVALUATOR_REGISTRY: Dict[str, type] = {
    "exact": ExactMatchEvaluator,
    "multiple_choice": MultipleChoiceEvaluator,
    "numeric": NumericEvaluator,
}


def get_evaluator(
    evaluator_type: str,
    **kwargs,
) -> Optional[BaseEvaluator]:
    """Get an evaluator instance by type."""
    evaluator_class = EVALUATOR_REGISTRY.get(evaluator_type)
    if evaluator_class is None:
        logger.warning(f"Unknown evaluator type: {evaluator_type}")
        return None
    return evaluator_class(**kwargs)


def register_evaluator(evaluator_type: str, evaluator_class: type):
    """Register a custom evaluator class."""
    EVALUATOR_REGISTRY[evaluator_type] = evaluator_class


class CombatEvaluator:
    """
    Main combat evaluator that selects appropriate evaluator based on quest type.

    Example:
        evaluator = CombatEvaluator()
        result = evaluator.evaluate(quest, model_response)
    """

    def __init__(
        self,
        default_evaluator: Optional[BaseEvaluator] = None,
        evaluator_map: Optional[Dict[str, BaseEvaluator]] = None,
    ):
        self.default_evaluator = default_evaluator or ExactMatchEvaluator()
        self.evaluator_map = evaluator_map or {}

    def set_evaluator(self, quest_type: str, evaluator: BaseEvaluator):
        """Set evaluator for a quest type."""
        self.evaluator_map[quest_type] = evaluator

    def get_evaluator(self, quest: QuestInstance) -> BaseEvaluator:
        """Get appropriate evaluator for a quest."""
        # Check for explicit evaluator type in quest context/metadata
        evaluator_type = None
        if hasattr(quest, 'context') and isinstance(quest.context, dict):
            evaluator_type = quest.context.get("evaluator_type")
        if not evaluator_type and hasattr(quest, 'metadata') and isinstance(quest.metadata, dict):
            evaluator_type = quest.metadata.get("evaluator_type")

        if evaluator_type and evaluator_type in EVALUATOR_REGISTRY:
            return EVALUATOR_REGISTRY[evaluator_type]()

        # Check by skill
        skill_id = get_skill_id(quest)
        if skill_id in self.evaluator_map:
            return self.evaluator_map[skill_id]

        return self.default_evaluator

    def evaluate(
        self,
        quest: QuestInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate a model response."""
        evaluator = self.get_evaluator(quest)
        return evaluator.evaluate(quest, response)


# Global evaluator
_combat_evaluator: Optional[CombatEvaluator] = None


def init_combat_evaluator(
    default_evaluator: Optional[BaseEvaluator] = None,
) -> CombatEvaluator:
    """Initialize the global combat evaluator."""
    global _combat_evaluator
    _combat_evaluator = CombatEvaluator(default_evaluator)
    return _combat_evaluator


def get_combat_evaluator() -> CombatEvaluator:
    """Get the global combat evaluator."""
    global _combat_evaluator
    if _combat_evaluator is None:
        _combat_evaluator = CombatEvaluator()
    return _combat_evaluator


def reset_combat_evaluator():
    """Reset the global combat evaluator (for testing)."""
    global _combat_evaluator
    _combat_evaluator = None


def evaluate_combat(
    quest: QuestInstance,
    response: str,
) -> EvaluationResult:
    """Evaluate a quest response using the global evaluator."""
    return get_combat_evaluator().evaluate(quest, response)
