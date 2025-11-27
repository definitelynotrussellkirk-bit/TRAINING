"""Quest evaluation and result calculation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from guild.quests.types import (
    QuestInstance,
    QuestResult,
    QuestTemplate,
    CombatResult,
)
from guild.combat.types import CombatConfig


logger = logging.getLogger(__name__)


@dataclass
class EvaluationContext:
    """Context passed to evaluators."""
    hero_id: str
    response: str
    quest: QuestInstance
    template: Optional[QuestTemplate] = None
    response_metadata: dict = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return int(delta.total_seconds() * 1000)
        return 0


@dataclass
class EvaluationOutcome:
    """Outcome from an evaluator."""
    combat_result: CombatResult
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


class QuestEvaluator(ABC):
    """
    Abstract base class for quest evaluators.

    Evaluators determine the combat result (CRIT/HIT/MISS etc)
    based on the hero's response to a quest.

    Subclass this to create custom evaluation logic.
    """

    evaluator_id: str = "base"

    @abstractmethod
    def evaluate(
        self,
        context: EvaluationContext,
        params: Optional[dict] = None,
    ) -> EvaluationOutcome:
        """
        Evaluate a quest response.

        Args:
            context: Evaluation context with response and quest
            params: Optional evaluation parameters

        Returns:
            EvaluationOutcome with combat result and metrics
        """
        pass


class ExactMatchEvaluator(QuestEvaluator):
    """
    Evaluator that checks for exact match.

    Used when quest.expected contains an exact string to match.
    """

    evaluator_id: str = "exact_match"

    def evaluate(
        self,
        context: EvaluationContext,
        params: Optional[dict] = None,
    ) -> EvaluationOutcome:
        """Evaluate with exact string matching."""
        params = params or {}

        expected = context.quest.expected
        if expected is None:
            return EvaluationOutcome(
                combat_result=CombatResult.MISS,
                notes="No expected answer defined",
            )

        # Extract expected answer
        if isinstance(expected, dict):
            expected_answer = expected.get("answer", str(expected))
        else:
            expected_answer = str(expected)

        # Normalize for comparison
        response = context.response.strip()
        expected_answer = expected_answer.strip()

        case_sensitive = params.get("case_sensitive", False)
        if not case_sensitive:
            response = response.lower()
            expected_answer = expected_answer.lower()

        # Check match
        if response == expected_answer:
            return EvaluationOutcome(
                combat_result=CombatResult.HIT,
                metrics={"exact_match": 1.0},
                notes="Exact match",
            )
        else:
            return EvaluationOutcome(
                combat_result=CombatResult.MISS,
                metrics={"exact_match": 0.0},
                notes=f"Expected '{expected_answer}', got '{response[:50]}'",
            )


class ContainsEvaluator(QuestEvaluator):
    """
    Evaluator that checks if response contains expected.

    More lenient than exact match.
    """

    evaluator_id: str = "contains"

    def evaluate(
        self,
        context: EvaluationContext,
        params: Optional[dict] = None,
    ) -> EvaluationOutcome:
        """Evaluate with contains matching."""
        params = params or {}

        expected = context.quest.expected
        if expected is None:
            return EvaluationOutcome(
                combat_result=CombatResult.MISS,
                notes="No expected answer defined",
            )

        if isinstance(expected, dict):
            expected_answer = expected.get("answer", str(expected))
        else:
            expected_answer = str(expected)

        response = context.response.strip().lower()
        expected_answer = expected_answer.strip().lower()

        if expected_answer in response:
            return EvaluationOutcome(
                combat_result=CombatResult.HIT,
                metrics={"contains": 1.0},
                notes="Response contains expected",
            )
        else:
            return EvaluationOutcome(
                combat_result=CombatResult.MISS,
                metrics={"contains": 0.0},
                notes="Response does not contain expected",
            )


class CallbackEvaluator(QuestEvaluator):
    """
    Evaluator that uses a callback function.

    Allows registering arbitrary evaluation functions.
    """

    evaluator_id: str = "callback"

    def __init__(
        self,
        callback: Callable[[EvaluationContext, dict], EvaluationOutcome],
        evaluator_id: str = "callback",
    ):
        """
        Initialize with callback.

        Callback signature: (context, params) -> EvaluationOutcome
        """
        self.callback = callback
        self.evaluator_id = evaluator_id

    def evaluate(
        self,
        context: EvaluationContext,
        params: Optional[dict] = None,
    ) -> EvaluationOutcome:
        """Evaluate using callback."""
        params = params or {}
        return self.callback(context, params)


class QuestJudge:
    """
    Central evaluation manager.

    The QuestJudge:
    1. Manages registered evaluators
    2. Routes quests to appropriate evaluators
    3. Calculates XP and creates QuestResult objects

    Example:
        judge = QuestJudge()
        judge.register(MyCustomEvaluator())

        result = judge.evaluate(
            hero_id="hero_123",
            response="my answer",
            quest=quest_instance,
        )
    """

    def __init__(self, combat_config: Optional[CombatConfig] = None):
        self._evaluators: dict[str, QuestEvaluator] = {}
        self._default_evaluator = ExactMatchEvaluator()
        self._combat_config = combat_config or CombatConfig()

        # Register built-in evaluators
        self.register(self._default_evaluator)
        self.register(ContainsEvaluator())

    def register(self, evaluator: QuestEvaluator):
        """Register an evaluator."""
        self._evaluators[evaluator.evaluator_id] = evaluator
        logger.debug(f"Registered quest evaluator: {evaluator.evaluator_id}")

    def unregister(self, evaluator_id: str):
        """Remove a registered evaluator."""
        self._evaluators.pop(evaluator_id, None)

    def get_evaluator(self, evaluator_id: str) -> Optional[QuestEvaluator]:
        """Get an evaluator by ID."""
        return self._evaluators.get(evaluator_id)

    def list_evaluators(self) -> list[str]:
        """List registered evaluator IDs."""
        return list(self._evaluators.keys())

    def evaluate(
        self,
        hero_id: str,
        response: str,
        quest: QuestInstance,
        template: Optional[QuestTemplate] = None,
        response_metadata: Optional[dict] = None,
        params: Optional[dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> QuestResult:
        """
        Evaluate a quest response and create result.

        Args:
            hero_id: Hero identifier
            response: Hero's response to quest
            quest: The quest instance
            template: Optional template for evaluator lookup
            response_metadata: Optional metadata about response
            params: Optional evaluation parameters
            start_time: When response started
            end_time: When response completed

        Returns:
            QuestResult with combat result and XP
        """
        if end_time is None:
            end_time = datetime.now()

        # Create evaluation context
        context = EvaluationContext(
            hero_id=hero_id,
            response=response,
            quest=quest,
            template=template,
            response_metadata=response_metadata or {},
            start_time=start_time,
            end_time=end_time,
        )

        # Get evaluator
        evaluator_id = "exact_match"  # Default
        if template and template.evaluator_id:
            evaluator_id = template.evaluator_id

        evaluator = self._evaluators.get(evaluator_id)
        if evaluator is None:
            logger.warning(
                f"Evaluator '{evaluator_id}' not found, using default"
            )
            evaluator = self._default_evaluator

        # Merge params
        merged_params = {}
        if template and template.evaluator_params:
            merged_params.update(template.evaluator_params)
        if params:
            merged_params.update(params)

        # Evaluate
        outcome = evaluator.evaluate(context, merged_params)

        # Calculate XP
        xp_awarded = self._calculate_xp(
            combat_result=outcome.combat_result,
            quest=quest,
            template=template,
        )

        # Create result
        result = QuestResult(
            quest_id=quest.id,
            hero_id=hero_id,
            response=response,
            response_metadata=context.response_metadata,
            combat_result=outcome.combat_result,
            metrics=outcome.metrics,
            xp_awarded=xp_awarded,
            effects_triggered=[],  # Effects determined by caller
            attempted_at=end_time,
            duration_ms=context.duration_ms,
            evaluator_notes=outcome.notes,
        )

        logger.debug(
            f"Quest {quest.id} evaluated: {outcome.combat_result.value} "
            f"(XP: {result.total_xp})"
        )

        return result

    def _calculate_xp(
        self,
        combat_result: CombatResult,
        quest: QuestInstance,
        template: Optional[QuestTemplate],
    ) -> dict[str, int]:
        """Calculate XP rewards for each skill."""
        base_xp = self._combat_config.get_base_xp(combat_result)
        difficulty_mult = self._combat_config.get_difficulty_multiplier(
            quest.difficulty_level
        )

        # Get skill-specific base XP from template if available
        skill_base_xp = {}
        if template and template.base_xp:
            skill_base_xp = template.base_xp

        xp_awarded = {}
        for skill in quest.skills:
            # Use template's base_xp if available, otherwise use combat config
            skill_base = skill_base_xp.get(skill, base_xp)
            xp = int(skill_base * difficulty_mult)
            xp_awarded[skill] = xp

        return xp_awarded


# Global judge instance
_judge: Optional[QuestJudge] = None


def get_judge() -> QuestJudge:
    """Get the global quest judge, initializing if needed."""
    global _judge
    if _judge is None:
        _judge = QuestJudge()
    return _judge


def reset_judge():
    """Reset the global judge (for testing)."""
    global _judge
    _judge = None


# Convenience functions

def evaluate_quest(
    hero_id: str,
    response: str,
    quest: QuestInstance,
    template: Optional[QuestTemplate] = None,
    **kwargs,
) -> QuestResult:
    """Evaluate a quest response using the global judge."""
    return get_judge().evaluate(
        hero_id=hero_id,
        response=response,
        quest=quest,
        template=template,
        **kwargs,
    )


def register_evaluator(evaluator: QuestEvaluator):
    """Register an evaluator with the global judge."""
    get_judge().register(evaluator)
