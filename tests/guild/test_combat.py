"""Tests for combat evaluation, calculation, and stance management."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from guild.quests.types import CombatResult, QuestInstance, QuestDifficulty
from guild.combat.types import (
    CombatStance,
    CombatConfig,
    StanceConfig,
)
from guild.combat.evaluator import (
    MatchQuality,
    EvaluationResult,
    normalize_answer,
    extract_answer,
    ExactMatchEvaluator,
    MultipleChoiceEvaluator,
    NumericEvaluator,
    CombatEvaluator,
    init_combat_evaluator,
    get_combat_evaluator,
    reset_combat_evaluator,
    evaluate_combat,
    EVALUATOR_REGISTRY,
    get_evaluator,
)
from guild.combat.calculator import (
    XPBreakdown,
    CombatCalculator,
    CombatReporter,
    init_combat_calculator,
    get_combat_calculator,
    reset_combat_calculator,
    calculate_combat_xp,
)
from guild.combat.stance import (
    StanceSelection,
    StanceManager,
    ResponseFormatter,
    init_stance_manager,
    get_stance_manager,
    reset_stance_manager,
    select_stance,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_combat_evaluator()
    reset_combat_calculator()
    reset_stance_manager()
    yield
    reset_combat_evaluator()
    reset_combat_calculator()
    reset_stance_manager()


@pytest.fixture
def sample_quest():
    """Create a sample quest instance."""
    return QuestInstance(
        id="quest_001",
        template_id="test_template",
        skills=["logic"],
        prompt="What is 2 + 2?",
        expected={"answer": "4"},
        difficulty=QuestDifficulty.SILVER,
        difficulty_level=2,
    )


@pytest.fixture
def mc_quest():
    """Create a multiple-choice quest."""
    return QuestInstance(
        id="mc_001",
        template_id="mc_template",
        skills=["logic"],
        prompt="What color is the sky? A) Red B) Blue C) Green D) Yellow",
        expected={"answer": "Blue"},
        difficulty=QuestDifficulty.BRONZE,
        difficulty_level=1,
        context={
            "choices": ["Red", "Blue", "Green", "Yellow"],
            "correct_index": 1,
        },
    )


# =============================================================================
# Type Tests
# =============================================================================

class TestCombatTypes:
    """Tests for combat type definitions."""

    def test_combat_stance_enum(self):
        assert CombatStance.THOUGHTFUL.value == "thoughtful"
        assert CombatStance.QUICK_DRAW.value == "quick_draw"
        assert CombatStance.ALTERNATING.value == "alternating"

    def test_combat_config_base_xp(self):
        config = CombatConfig()

        assert config.get_base_xp(CombatResult.CRITICAL_HIT) == 15
        assert config.get_base_xp(CombatResult.HIT) == 10
        assert config.get_base_xp(CombatResult.GLANCING) == 5
        assert config.get_base_xp(CombatResult.MISS) == 2
        assert config.get_base_xp(CombatResult.CRITICAL_MISS) == 0

    def test_combat_config_difficulty_multiplier(self):
        config = CombatConfig()

        assert config.get_difficulty_multiplier(1) == 1.0
        assert config.get_difficulty_multiplier(5) == 1.5
        assert config.get_difficulty_multiplier(10) == 3.0

    def test_stance_config(self):
        config = StanceConfig()

        assert "💭" in config.thinking_emojis
        assert "🔚" in config.stop_emojis
        assert config.min_stop_count == 2
        assert config.max_stop_count == 4


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_normalize_answer_basic(self):
        assert normalize_answer("Hello") == "hello"
        assert normalize_answer("  WORLD  ") == "world"
        assert normalize_answer("Hello, World!") == "hello world"

    def test_normalize_answer_whitespace(self):
        assert normalize_answer("a   b   c") == "a b c"
        assert normalize_answer("\n\t test \n") == "test"

    def test_normalize_answer_empty(self):
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""

    def test_extract_answer_basic(self):
        assert extract_answer("The answer is 42") == "42"
        assert extract_answer("Answer: blue") == "blue"
        assert extract_answer("Therefore, red") == "red"

    def test_extract_answer_multiline(self):
        text = """Some reasoning here.
        More thinking.
        The answer is 7."""
        assert extract_answer(text) == "7"

    def test_extract_answer_fallback(self):
        text = "Just the answer"
        assert extract_answer(text) == "Just the answer"


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestExactMatchEvaluator:
    """Tests for exact match evaluation."""

    def test_exact_match(self, sample_quest):
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate(sample_quest, "The answer is 4")

        assert result.combat_result == CombatResult.CRITICAL_HIT
        assert result.match_quality == MatchQuality.EXACT
        assert result.success is True

    def test_normalized_match(self, sample_quest):
        # Create quest with punctuation in expected answer
        quest = QuestInstance(
            id="q_norm",
            template_id="t1",
            skills=["logic"],
            prompt="What is the answer?",
            expected={"answer": "4!"},  # Has punctuation
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
        )
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate(quest, "4")  # No punctuation

        assert result.combat_result == CombatResult.HIT
        assert result.match_quality == MatchQuality.NORMALIZED

    def test_partial_match(self, sample_quest):
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate(sample_quest, "The answer is 4 or maybe 5")

        assert result.combat_result == CombatResult.GLANCING
        assert result.match_quality == MatchQuality.PARTIAL

    def test_wrong_answer(self, sample_quest):
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate(sample_quest, "The answer is 5")

        assert result.combat_result == CombatResult.MISS
        assert result.match_quality == MatchQuality.WRONG
        assert result.success is False

    def test_case_insensitive(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            prompt="What color?",
            expected={"answer": "BLUE"},
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
        )

        evaluator = ExactMatchEvaluator(case_sensitive=False)
        result = evaluator.evaluate(quest, "blue")

        assert result.combat_result == CombatResult.CRITICAL_HIT


class TestMultipleChoiceEvaluator:
    """Tests for multiple choice evaluation."""

    def test_correct_letter(self, mc_quest):
        evaluator = MultipleChoiceEvaluator()
        result = evaluator.evaluate(mc_quest, "B")

        assert result.combat_result == CombatResult.HIT
        assert result.success is True

    def test_correct_text(self, mc_quest):
        evaluator = MultipleChoiceEvaluator()
        result = evaluator.evaluate(mc_quest, "The answer is Blue")

        assert result.combat_result == CombatResult.CRITICAL_HIT
        assert result.match_quality == MatchQuality.EXACT

    def test_wrong_letter(self, mc_quest):
        evaluator = MultipleChoiceEvaluator()
        result = evaluator.evaluate(mc_quest, "A")

        assert result.combat_result == CombatResult.MISS
        assert result.success is False


class TestNumericEvaluator:
    """Tests for numeric evaluation."""

    def test_exact_number(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["math"],
            prompt="What is 10/2?",
            expected={"answer": "5"},
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
        )

        evaluator = NumericEvaluator()
        result = evaluator.evaluate(quest, "5")

        assert result.combat_result == CombatResult.CRITICAL_HIT

    def test_within_tolerance(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["math"],
            prompt="What is pi?",
            expected={"answer": "3.14159"},
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
        )

        evaluator = NumericEvaluator(absolute_tolerance=0.01)
        result = evaluator.evaluate(quest, "3.14")

        assert result.combat_result == CombatResult.HIT

    def test_wrong_number(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["math"],
            prompt="What is 2+2?",
            expected={"answer": "4"},
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
        )

        evaluator = NumericEvaluator()
        result = evaluator.evaluate(quest, "100")

        assert result.combat_result == CombatResult.MISS


class TestCombatEvaluator:
    """Tests for the main combat evaluator."""

    def test_default_evaluator(self, sample_quest):
        evaluator = CombatEvaluator()
        result = evaluator.evaluate(sample_quest, "4")

        assert result.combat_result == CombatResult.CRITICAL_HIT

    def test_custom_evaluator_map(self, mc_quest):
        evaluator = CombatEvaluator()
        evaluator.set_evaluator("logic", MultipleChoiceEvaluator())

        result = evaluator.evaluate(mc_quest, "B")
        assert result.combat_result == CombatResult.HIT

    def test_evaluator_registry(self):
        assert "exact" in EVALUATOR_REGISTRY
        assert "multiple_choice" in EVALUATOR_REGISTRY
        assert "numeric" in EVALUATOR_REGISTRY

        exact = get_evaluator("exact")
        assert isinstance(exact, ExactMatchEvaluator)


class TestGlobalEvaluator:
    """Tests for global evaluator functions."""

    def test_init_and_get(self):
        init_combat_evaluator()
        evaluator = get_combat_evaluator()
        assert evaluator is not None

    def test_evaluate_combat_convenience(self, sample_quest):
        result = evaluate_combat(sample_quest, "4")
        assert result.combat_result == CombatResult.CRITICAL_HIT


# =============================================================================
# Calculator Tests
# =============================================================================

class TestCombatCalculator:
    """Tests for XP calculation."""

    def test_base_xp(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(CombatResult.CRITICAL_HIT)

        assert breakdown.base_xp == 15
        assert breakdown.total_xp == 15

    def test_difficulty_multiplier(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(CombatResult.HIT, difficulty=5)

        assert breakdown.base_xp == 10
        assert breakdown.multipliers["difficulty"] == 1.5
        assert breakdown.total_xp == 15  # 10 * 1.5

    def test_skill_multiplier(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.HIT,
            skill_multiplier=2.0,
        )

        assert breakdown.multipliers["skill"] == 2.0
        assert breakdown.total_xp == 20  # 10 * 2.0

    def test_streak_bonus(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.HIT,
            streak=10,
        )

        # 10 * 5% = 50% bonus
        assert breakdown.multipliers["streak"] == 1.5
        assert breakdown.total_xp == 15  # 10 * 1.5

    def test_streak_capped(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.HIT,
            streak=100,  # Very long streak
        )

        # Capped at 50%
        assert breakdown.multipliers["streak"] == 1.5

    def test_streak_no_bonus_on_miss(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.MISS,
            streak=10,
        )

        # No streak bonus for misses
        assert "streak" not in breakdown.multipliers

    def test_effect_multipliers(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.HIT,
            effect_multipliers=[1.1, 1.2],
        )

        # 10 * 1.1 * 1.2 = 13.2 -> 13
        assert breakdown.total_xp == 13

    def test_combined_bonuses(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(
            CombatResult.CRITICAL_HIT,  # 15 base
            difficulty=5,               # 1.5x
            skill_multiplier=1.5,       # 1.5x
            streak=4,                   # 1.2x (4 * 5% = 20%)
            effect_multipliers=[1.1],   # 1.1x
        )

        # 15 * 1.5 * 1.5 * 1.2 * 1.1 = 44.55 -> 45
        assert breakdown.total_xp == 45

    def test_from_quest(self, sample_quest):
        calc = CombatCalculator()
        breakdown = calc.calculate_from_quest(
            sample_quest,
            CombatResult.HIT,
        )

        # SILVER = 2, multiplier 1.1
        assert breakdown.total_xp == 11  # 10 * 1.1

    def test_xp_breakdown_dict(self):
        calc = CombatCalculator()
        breakdown = calc.calculate_xp(CombatResult.HIT, difficulty=3)

        d = breakdown.to_dict()
        assert "base_xp" in d
        assert "total_xp" in d
        assert "multipliers" in d


class TestCombatReporter:
    """Tests for combat reporting."""

    def test_record_and_summary(self, sample_quest):
        calc = CombatCalculator()
        reporter = CombatReporter(calc)

        # Record some combats
        for i in range(10):
            combat_result = CombatResult.HIT if i < 7 else CombatResult.MISS
            eval_result = EvaluationResult(
                combat_result=combat_result,
                match_quality=MatchQuality.EXACT if combat_result == CombatResult.HIT else MatchQuality.WRONG,
            )
            xp = calc.calculate_xp(combat_result)
            reporter.record(sample_quest, eval_result, xp)

        summary = reporter.get_summary()

        assert summary["total_combats"] == 10
        assert summary["wins"] == 7
        assert summary["losses"] == 3
        assert summary["win_rate"] == 0.7

    def test_skill_breakdown(self, sample_quest):
        calc = CombatCalculator()
        reporter = CombatReporter(calc)

        eval_result = EvaluationResult(
            combat_result=CombatResult.HIT,
            match_quality=MatchQuality.EXACT,
        )
        xp = calc.calculate_xp(CombatResult.HIT)
        reporter.record(sample_quest, eval_result, xp)

        breakdown = reporter.get_skill_breakdown()

        assert "logic" in breakdown
        assert breakdown["logic"]["attempts"] == 1


class TestGlobalCalculator:
    """Tests for global calculator functions."""

    def test_init_and_get(self):
        init_combat_calculator()
        calc = get_combat_calculator()
        assert calc is not None

    def test_calculate_combat_xp_convenience(self):
        breakdown = calculate_combat_xp(
            CombatResult.HIT,
            difficulty=3,
        )

        assert breakdown.total_xp > 0


# =============================================================================
# Stance Tests
# =============================================================================

class TestStanceManager:
    """Tests for stance management."""

    def test_thoughtful_stance(self):
        manager = StanceManager()
        selection = manager.select_stance(
            index=0,
            stance=CombatStance.THOUGHTFUL,
        )

        assert selection.stance == CombatStance.THOUGHTFUL
        assert selection.use_thinking is True
        assert selection.thinking_emoji != ""
        assert selection.stop_emojis != ""

    def test_quick_draw_stance(self):
        manager = StanceManager()
        selection = manager.select_stance(
            index=0,
            stance=CombatStance.QUICK_DRAW,
        )

        assert selection.stance == CombatStance.QUICK_DRAW
        assert selection.use_thinking is False
        assert selection.thinking_emoji == ""

    def test_alternating_stance(self):
        manager = StanceManager()

        # Even index = thinking
        sel0 = manager.select_stance(index=0, stance=CombatStance.ALTERNATING)
        assert sel0.use_thinking is True

        # Odd index = direct
        sel1 = manager.select_stance(index=1, stance=CombatStance.ALTERNATING)
        assert sel1.use_thinking is False

        # Back to thinking
        sel2 = manager.select_stance(index=2, stance=CombatStance.ALTERNATING)
        assert sel2.use_thinking is True

    def test_default_stance(self):
        manager = StanceManager()
        manager.set_default_stance(CombatStance.THOUGHTFUL)

        selection = manager.select_stance(index=0)
        assert selection.use_thinking is True

    def test_emoji_selection_deterministic(self):
        manager = StanceManager()

        sel1 = manager.select_stance(index=5, stance=CombatStance.THOUGHTFUL)
        sel2 = manager.select_stance(index=5, stance=CombatStance.THOUGHTFUL)

        assert sel1.thinking_emoji == sel2.thinking_emoji
        assert sel1.stop_emojis == sel2.stop_emojis


class TestResponseFormatter:
    """Tests for response formatting."""

    def test_format_system_prompt_thinking(self):
        formatter = ResponseFormatter()
        selection = StanceSelection(
            stance=CombatStance.THOUGHTFUL,
            use_thinking=True,
            thinking_emoji="💭",
            stop_emojis="🔚🔚",
        )

        prompt = formatter.format_system_prompt(
            "You are a helpful assistant.",
            selection,
        )

        assert "💭" in prompt
        assert "🔚🔚" in prompt
        assert "step by step" in prompt.lower()

    def test_format_system_prompt_direct(self):
        formatter = ResponseFormatter()
        selection = StanceSelection(
            stance=CombatStance.QUICK_DRAW,
            use_thinking=False,
        )

        prompt = formatter.format_system_prompt(
            "You are a helpful assistant.",
            selection,
        )

        assert "directly" in prompt.lower()

    def test_validate_thinking_response(self):
        formatter = ResponseFormatter()
        selection = StanceSelection(
            stance=CombatStance.THOUGHTFUL,
            use_thinking=True,
        )

        # Valid thinking response
        valid, reason = formatter.validate_response(
            "💭 Let me think... 🔚🔚 The answer is 4.",
            selection,
        )
        assert valid is True

        # Missing thinking emoji
        valid, reason = formatter.validate_response(
            "Let me think... 🔚🔚 The answer is 4.",
            selection,
        )
        assert valid is False

    def test_validate_direct_response(self):
        formatter = ResponseFormatter()
        selection = StanceSelection(
            stance=CombatStance.QUICK_DRAW,
            use_thinking=False,
        )

        # Valid direct response
        valid, reason = formatter.validate_response(
            "The answer is 4.",
            selection,
        )
        assert valid is True

        # Unexpected thinking emoji
        valid, reason = formatter.validate_response(
            "💭 The answer is 4.",
            selection,
        )
        assert valid is False

    def test_extract_thinking(self):
        formatter = ResponseFormatter()

        thinking, answer = formatter.extract_thinking(
            "💭 I need to calculate... 🔚🔚 42"
        )

        assert "calculate" in thinking
        assert answer == "42"

    def test_extract_thinking_no_marker(self):
        formatter = ResponseFormatter()

        thinking, answer = formatter.extract_thinking(
            "Just the answer"
        )

        assert thinking == ""
        assert answer == "Just the answer"


class TestGlobalStanceManager:
    """Tests for global stance manager functions."""

    def test_init_and_get(self):
        init_stance_manager()
        manager = get_stance_manager()
        assert manager is not None

    def test_select_stance_convenience(self):
        selection = select_stance(index=0)
        assert selection is not None
        assert selection.stance is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestCombatIntegration:
    """Integration tests for the combat module."""

    def test_full_combat_workflow(self, sample_quest):
        """Test complete combat workflow."""
        # 1. Select stance
        stance = select_stance(index=0)

        # 2. Evaluate response
        response = "4" if stance.use_thinking else "The answer is 4"
        if stance.use_thinking:
            response = f"{stance.thinking_emoji} Let me think... {stance.stop_emojis} 4"

        eval_result = evaluate_combat(sample_quest, response)

        # 3. Calculate XP
        xp = calculate_combat_xp(
            combat_result=eval_result.combat_result,
            difficulty=sample_quest.difficulty.value,
        )

        assert eval_result.success is True
        assert xp.total_xp > 0

    def test_alternating_stance_distribution(self):
        """Test that alternating gives 50/50 distribution."""
        thinking_count = 0
        direct_count = 0

        for i in range(100):
            selection = select_stance(index=i, stance=CombatStance.ALTERNATING)
            if selection.use_thinking:
                thinking_count += 1
            else:
                direct_count += 1

        assert thinking_count == 50
        assert direct_count == 50
