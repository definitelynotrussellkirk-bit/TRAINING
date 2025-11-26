# Guild Refactor - Tasks Phase 7-9

**Phases:** Combat Calculator, Incidents System, Runs System
**Prerequisites:** Phases 0-6 complete (guild-p6-complete tag)

---

# Phase 7: Combat Calculator

**Goal:** Calculate quest results (CRIT/HIT/MISS) and XP awards

---

### P7.1 - Create guild/combat/calculator.py

**Description:** Combat calculator determines quest outcomes and XP

**File:** `guild/combat/calculator.py`

```python
"""Combat calculator - determines quest outcomes and XP awards."""

from typing import Optional, Callable, Any
from datetime import datetime

from guild.combat.types import CombatConfig, CombatStance, StanceConfig
from guild.quests.types import (
    QuestInstance, QuestResult, QuestTemplate, CombatResult
)
from guild.types import generate_id


# Evaluator function signature
EvaluatorFn = Callable[[QuestInstance, str, dict], tuple[CombatResult, dict[str, float], str]]


class CombatCalculator:
    """
    Calculates quest outcomes based on responses.

    Evaluators are registered per-task type to determine if responses are correct.
    """

    def __init__(self, config: Optional[CombatConfig] = None):
        self.config = config or CombatConfig()
        self.stance_config = StanceConfig()
        self._evaluators: dict[str, EvaluatorFn] = {}
        self._current_stance = self.config.default_stance

        # Track consecutive results for debuff triggers
        self._consecutive_misses = 0
        self._consecutive_crit_misses = 0

    def register_evaluator(self, evaluator_id: str, fn: EvaluatorFn):
        """Register an evaluator function."""
        self._evaluators[evaluator_id] = fn

    def has_evaluator(self, evaluator_id: str) -> bool:
        """Check if an evaluator is registered."""
        return evaluator_id in self._evaluators

    def list_evaluators(self) -> list[str]:
        """List registered evaluator IDs."""
        return list(self._evaluators.keys())

    def evaluate(self,
                 quest: QuestInstance,
                 response: str,
                 response_metadata: Optional[dict] = None,
                 evaluator_id: Optional[str] = None,
                 hero_id: str = "") -> QuestResult:
        """
        Evaluate a quest response and calculate result.

        Args:
            quest: The quest instance
            response: Hero's response text
            response_metadata: Additional response info (timing, tokens, etc.)
            evaluator_id: Override evaluator (defaults to quest template's)
            hero_id: ID of the hero who attempted

        Returns:
            QuestResult with combat outcome and XP awards
        """
        response_metadata = response_metadata or {}

        # Determine evaluator
        if evaluator_id is None:
            evaluator_id = quest.metadata.get("evaluator_id", "default")

        evaluator = self._evaluators.get(evaluator_id)
        if evaluator is None:
            evaluator = self._evaluators.get("default")
        if evaluator is None:
            # Fallback: no evaluator, use expected comparison
            combat_result, metrics, notes = self._default_evaluate(quest, response)
        else:
            combat_result, metrics, notes = evaluator(quest, response, response_metadata)

        # Update consecutive tracking
        self._update_consecutive(combat_result)

        # Calculate XP
        xp_awarded = self._calculate_xp(quest, combat_result)

        # Build result
        return QuestResult(
            quest_id=quest.id,
            hero_id=hero_id,
            response=response,
            response_metadata=response_metadata,
            combat_result=combat_result,
            metrics=metrics,
            xp_awarded=xp_awarded,
            effects_triggered=self._check_triggers(combat_result),
            attempted_at=datetime.now(),
            duration_ms=response_metadata.get("duration_ms", 0),
            evaluator_notes=notes,
        )

    def _default_evaluate(self, quest: QuestInstance, response: str
                          ) -> tuple[CombatResult, dict[str, float], str]:
        """Default evaluator - exact match against expected."""
        if quest.expected is None:
            # No expected answer, assume HIT for valid response
            return CombatResult.HIT, {"has_response": 1.0}, "No expected answer defined"

        expected_answer = quest.expected.get("answer", "")

        # Normalize for comparison
        response_clean = response.strip().lower()
        expected_clean = str(expected_answer).strip().lower()

        if response_clean == expected_clean:
            return CombatResult.HIT, {"exact_match": 1.0}, "Exact match"

        # Check for partial match
        if expected_clean in response_clean or response_clean in expected_clean:
            return CombatResult.GLANCING, {"partial_match": 0.5}, "Partial match"

        return CombatResult.MISS, {"exact_match": 0.0}, "No match"

    def _calculate_xp(self, quest: QuestInstance, result: CombatResult) -> dict[str, int]:
        """Calculate XP awards for each skill."""
        base_xp = self.config.get_base_xp(result)
        difficulty_mult = self.config.get_difficulty_multiplier(quest.difficulty_level)

        xp_awarded = {}
        for skill_id in quest.skills:
            skill_xp = int(base_xp * difficulty_mult)
            xp_awarded[skill_id] = skill_xp

        return xp_awarded

    def _update_consecutive(self, result: CombatResult):
        """Update consecutive miss/crit_miss tracking."""
        if result == CombatResult.CRITICAL_MISS:
            self._consecutive_crit_misses += 1
            self._consecutive_misses += 1
        elif result == CombatResult.MISS:
            self._consecutive_misses += 1
            self._consecutive_crit_misses = 0
        else:
            self._consecutive_misses = 0
            self._consecutive_crit_misses = 0

    def _check_triggers(self, result: CombatResult) -> list[str]:
        """Check if any debuff triggers should fire."""
        triggered = []

        if self._consecutive_crit_misses >= self.config.crit_miss_debuff_threshold:
            triggered.append("curse_of_repetition")

        if self._consecutive_misses >= self.config.miss_debuff_threshold:
            triggered.append("confusion")

        return triggered

    # Stance management

    def set_stance(self, stance: CombatStance):
        """Set the current combat stance."""
        self._current_stance = stance

    def get_stance(self) -> CombatStance:
        """Get current stance."""
        return self._current_stance

    def should_use_thinking(self, index: int = 0) -> bool:
        """
        Determine if thinking mode should be used.
        For ALTERNATING stance, uses index for deterministic 50/50.
        """
        if self._current_stance == CombatStance.THOUGHTFUL:
            return True
        elif self._current_stance == CombatStance.QUICK_DRAW:
            return False
        else:  # ALTERNATING
            return index % 2 == 0

    def reset_consecutive(self):
        """Reset consecutive failure counters."""
        self._consecutive_misses = 0
        self._consecutive_crit_misses = 0


# Global calculator
_calculator: Optional[CombatCalculator] = None


def get_combat_calculator() -> CombatCalculator:
    """Get the global combat calculator."""
    global _calculator
    if _calculator is None:
        _calculator = CombatCalculator()
        _register_builtin_evaluators(_calculator)
    return _calculator


def reset_combat_calculator():
    """Reset the global calculator."""
    global _calculator
    _calculator = None


def _register_builtin_evaluators(calc: CombatCalculator):
    """Register built-in evaluators."""

    def exact_match_evaluator(quest: QuestInstance, response: str, metadata: dict
                              ) -> tuple[CombatResult, dict[str, float], str]:
        """Exact string match evaluator."""
        if quest.expected is None:
            return CombatResult.MISS, {}, "No expected answer"

        expected = str(quest.expected.get("answer", "")).strip()
        response = response.strip()

        if response == expected:
            return CombatResult.HIT, {"exact_match": 1.0}, "Exact match"
        elif response.lower() == expected.lower():
            return CombatResult.HIT, {"exact_match": 1.0, "case_diff": 1.0}, "Case-insensitive match"
        else:
            return CombatResult.MISS, {"exact_match": 0.0}, f"Expected: {expected}"

    def json_evaluator(quest: QuestInstance, response: str, metadata: dict
                       ) -> tuple[CombatResult, dict[str, float], str]:
        """JSON answer evaluator - extracts answer from JSON."""
        import json

        try:
            parsed = json.loads(response)
            answer = parsed.get("answer") or parsed.get("response")
        except json.JSONDecodeError:
            return CombatResult.CRITICAL_MISS, {"json_valid": 0.0}, "Invalid JSON"

        if quest.expected is None:
            return CombatResult.HIT, {"json_valid": 1.0}, "Valid JSON, no expected"

        expected = quest.expected.get("answer")

        if answer == expected:
            return CombatResult.HIT, {"json_valid": 1.0, "correct": 1.0}, "Correct"
        elif str(answer).lower() == str(expected).lower():
            return CombatResult.GLANCING, {"json_valid": 1.0, "correct": 0.5}, "Case mismatch"
        else:
            return CombatResult.MISS, {"json_valid": 1.0, "correct": 0.0}, f"Expected: {expected}"

    calc.register_evaluator("default", exact_match_evaluator)
    calc.register_evaluator("exact_match", exact_match_evaluator)
    calc.register_evaluator("json", json_evaluator)


def register_evaluator(evaluator_id: str, fn: EvaluatorFn):
    """Register an evaluator with the global calculator."""
    get_combat_calculator().register_evaluator(evaluator_id, fn)


def evaluate_quest(quest: QuestInstance, response: str,
                   hero_id: str = "", **kwargs) -> QuestResult:
    """Evaluate a quest using the global calculator."""
    return get_combat_calculator().evaluate(quest, response, hero_id=hero_id, **kwargs)
```

**Dependencies:** P1.3, P1.8

**Acceptance Criteria:**
- [ ] `from guild.combat.calculator import get_combat_calculator, evaluate_quest` works
- [ ] Evaluators can be registered and invoked
- [ ] XP calculation includes difficulty multiplier
- [ ] Consecutive miss tracking works
- [ ] Built-in evaluators work

**Effort:** L (45 min)

---

### P7.2 - Create guild/combat/evaluators.py

**Description:** Domain-specific evaluators (SYLLO, discrimination)

**File:** `guild/combat/evaluators.py`

```python
"""Domain-specific quest evaluators."""

from typing import Optional
import re
import json

from guild.quests.types import QuestInstance, CombatResult


def syllo_evaluator(quest: QuestInstance, response: str, metadata: dict
                    ) -> tuple[CombatResult, dict[str, float], str]:
    """
    SYLLO puzzle evaluator.

    Expects response to contain the hidden word, either:
    - As plain text (the word itself)
    - In JSON format {"answer": "word"}
    - With explanation followed by answer
    """
    if quest.expected is None:
        return CombatResult.MISS, {}, "No expected answer for SYLLO"

    expected_word = str(quest.expected.get("answer", "")).strip().lower()
    metrics = {}

    # Try to extract answer from various formats
    answer = _extract_syllo_answer(response)
    answer_lower = answer.lower() if answer else ""

    metrics["word_extracted"] = 1.0 if answer else 0.0

    # Check correctness
    if answer_lower == expected_word:
        metrics["correct"] = 1.0
        metrics["exact_match"] = 1.0

        # Check if response has good reasoning (bonus for CRIT)
        if _has_reasoning(response):
            return CombatResult.CRITICAL_HIT, metrics, "Correct with reasoning"
        else:
            return CombatResult.HIT, metrics, "Correct"

    # Check for close match (typos, etc.)
    if answer and _is_close_match(answer_lower, expected_word):
        metrics["correct"] = 0.5
        return CombatResult.GLANCING, metrics, f"Close match (expected: {expected_word})"

    # Wrong answer
    metrics["correct"] = 0.0
    return CombatResult.MISS, metrics, f"Expected: {expected_word}, got: {answer}"


def _extract_syllo_answer(response: str) -> Optional[str]:
    """Extract the answer word from various response formats."""
    response = response.strip()

    # Try JSON format
    try:
        data = json.loads(response)
        if "answer" in data:
            return str(data["answer"]).strip()
    except json.JSONDecodeError:
        pass

    # Try "Answer: word" format
    match = re.search(r'(?:answer|the word is|solution)[:\s]+([a-zA-Z]+)', response, re.I)
    if match:
        return match.group(1)

    # Try last word if response is short
    if len(response) < 50:
        words = response.split()
        if words:
            # Filter to alphabetic words
            alpha_words = [w for w in words if w.isalpha()]
            if alpha_words:
                return alpha_words[-1]

    # Try to find a single capitalized word
    caps = re.findall(r'\b[A-Z][a-z]+\b', response)
    if len(caps) == 1:
        return caps[0]

    return None


def _has_reasoning(response: str) -> bool:
    """Check if response contains reasoning."""
    reasoning_indicators = [
        "because", "since", "therefore", "thus",
        "the clue", "this means", "we can deduce",
        "syllable", "definition", "hint"
    ]
    response_lower = response.lower()
    return any(ind in response_lower for ind in reasoning_indicators)


def _is_close_match(answer: str, expected: str, max_distance: int = 1) -> bool:
    """Check if answer is close to expected (Levenshtein distance)."""
    if len(answer) != len(expected):
        return abs(len(answer) - len(expected)) <= 1 and (
            answer in expected or expected in answer
        )

    # Count character differences
    differences = sum(1 for a, e in zip(answer, expected) if a != e)
    return differences <= max_distance


def discrimination_evaluator(quest: QuestInstance, response: str, metadata: dict
                             ) -> tuple[CombatResult, dict[str, float], str]:
    """
    Discrimination task evaluator.

    For verification: expects "CORRECT" or "INCORRECT"
    For correction: expects "INCORRECT" followed by the correct answer
    """
    if quest.expected is None:
        return CombatResult.MISS, {}, "No expected answer"

    expected_judgment = quest.expected.get("judgment", "").upper()
    expected_correction = quest.expected.get("correction")

    response_upper = response.strip().upper()
    metrics = {}

    # Check judgment
    if expected_judgment == "CORRECT":
        if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
            metrics["judgment_correct"] = 1.0
            return CombatResult.HIT, metrics, "Correctly identified as CORRECT"
        else:
            metrics["judgment_correct"] = 0.0
            return CombatResult.MISS, metrics, "Should have said CORRECT"

    elif expected_judgment == "INCORRECT":
        if "INCORRECT" in response_upper:
            metrics["judgment_correct"] = 1.0

            # Check correction if expected
            if expected_correction:
                if expected_correction.lower() in response.lower():
                    metrics["correction_correct"] = 1.0
                    return CombatResult.CRITICAL_HIT, metrics, "Correct judgment and correction"
                else:
                    metrics["correction_correct"] = 0.0
                    return CombatResult.HIT, metrics, "Correct judgment, wrong correction"
            else:
                return CombatResult.HIT, metrics, "Correctly identified as INCORRECT"
        else:
            metrics["judgment_correct"] = 0.0
            return CombatResult.MISS, metrics, "Should have said INCORRECT"

    return CombatResult.MISS, {}, f"Unknown expected judgment: {expected_judgment}"


def format_compliance_evaluator(quest: QuestInstance, response: str, metadata: dict
                                ) -> tuple[CombatResult, dict[str, float], str]:
    """
    Evaluator for format compliance tasks.
    Checks if response matches expected format patterns.
    """
    metrics = {}

    expected_format = quest.expected.get("format") if quest.expected else None
    if not expected_format:
        return CombatResult.HIT, {"has_response": 1.0}, "No format specified"

    # Check format type
    if expected_format == "json":
        try:
            json.loads(response)
            metrics["json_valid"] = 1.0
            return CombatResult.HIT, metrics, "Valid JSON"
        except json.JSONDecodeError:
            metrics["json_valid"] = 0.0
            return CombatResult.MISS, metrics, "Invalid JSON"

    elif expected_format == "numbered_list":
        if re.search(r'^\d+\.', response, re.MULTILINE):
            return CombatResult.HIT, {"format_match": 1.0}, "Has numbered list"
        return CombatResult.MISS, {"format_match": 0.0}, "Missing numbered list"

    elif expected_format == "bullet_list":
        if re.search(r'^[-*•]', response, re.MULTILINE):
            return CombatResult.HIT, {"format_match": 1.0}, "Has bullet list"
        return CombatResult.MISS, {"format_match": 0.0}, "Missing bullet list"

    # Custom regex pattern
    elif expected_format.startswith("regex:"):
        pattern = expected_format[6:]
        if re.search(pattern, response):
            return CombatResult.HIT, {"format_match": 1.0}, "Matches pattern"
        return CombatResult.MISS, {"format_match": 0.0}, "Does not match pattern"

    return CombatResult.HIT, {}, f"Unknown format: {expected_format}"


# Registration helper
def register_domain_evaluators(calculator: "CombatCalculator"):
    """Register all domain-specific evaluators."""
    calculator.register_evaluator("syllo_evaluator", syllo_evaluator)
    calculator.register_evaluator("syllo", syllo_evaluator)
    calculator.register_evaluator("discrimination_evaluator", discrimination_evaluator)
    calculator.register_evaluator("discrimination", discrimination_evaluator)
    calculator.register_evaluator("format_compliance", format_compliance_evaluator)
```

**Dependencies:** P7.1

**Acceptance Criteria:**
- [ ] `from guild.combat.evaluators import syllo_evaluator` works
- [ ] SYLLO evaluator extracts answers from various formats
- [ ] Discrimination evaluator handles verify/correct modes
- [ ] Close match detection works

**Effort:** M (35 min)

---

### P7.3 - Update guild/combat/__init__.py

**Description:** Export combat components

**File:** `guild/combat/__init__.py`

```python
"""Combat system - evaluation and XP calculation."""

from guild.combat.types import (
    CombatConfig,
    CombatStance,
    StanceConfig,
)
from guild.combat.calculator import (
    CombatCalculator,
    get_combat_calculator,
    reset_combat_calculator,
    register_evaluator,
    evaluate_quest,
)
from guild.combat.evaluators import (
    syllo_evaluator,
    discrimination_evaluator,
    format_compliance_evaluator,
    register_domain_evaluators,
)

__all__ = [
    # Types
    "CombatConfig",
    "CombatStance",
    "StanceConfig",
    # Calculator
    "CombatCalculator",
    "get_combat_calculator",
    "reset_combat_calculator",
    "register_evaluator",
    "evaluate_quest",
    # Evaluators
    "syllo_evaluator",
    "discrimination_evaluator",
    "format_compliance_evaluator",
    "register_domain_evaluators",
]
```

**Dependencies:** P7.1, P7.2

**Acceptance Criteria:**
- [ ] `from guild.combat import evaluate_quest, syllo_evaluator` works

**Effort:** S (5 min)

---

### P7.4 - Create tests/guild/test_combat.py

**Description:** Tests for combat calculator

**File:** `tests/guild/test_combat.py`

```python
"""Tests for combat calculator."""

import pytest

from guild.combat.calculator import (
    CombatCalculator, get_combat_calculator, reset_combat_calculator
)
from guild.combat.evaluators import (
    syllo_evaluator, discrimination_evaluator
)
from guild.combat.types import CombatConfig, CombatStance
from guild.quests.types import QuestInstance, QuestDifficulty, CombatResult


class TestCombatCalculator:
    @pytest.fixture
    def calculator(self):
        reset_combat_calculator()
        return get_combat_calculator()

    @pytest.fixture
    def sample_quest(self):
        return QuestInstance(
            id="test_quest",
            template_id="test",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.SILVER,
            difficulty_level=3,
            prompt="Test prompt",
            expected={"answer": "correct_answer"},
        )

    def test_evaluate_exact_match(self, calculator, sample_quest):
        result = calculator.evaluate(
            sample_quest,
            response="correct_answer",
            hero_id="hero1"
        )

        assert result.combat_result == CombatResult.HIT
        assert result.metrics["exact_match"] == 1.0
        assert "logic_weaving" in result.xp_awarded

    def test_evaluate_miss(self, calculator, sample_quest):
        result = calculator.evaluate(
            sample_quest,
            response="wrong_answer",
            hero_id="hero1"
        )

        assert result.combat_result == CombatResult.MISS
        assert result.metrics.get("exact_match", 0) == 0.0

    def test_xp_difficulty_multiplier(self, calculator):
        # Low difficulty
        low_quest = QuestInstance(
            id="low",
            template_id="test",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Easy",
            expected={"answer": "a"},
        )

        # High difficulty
        high_quest = QuestInstance(
            id="high",
            template_id="test",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.DRAGON,
            difficulty_level=10,
            prompt="Hard",
            expected={"answer": "a"},
        )

        low_result = calculator.evaluate(low_quest, "a", hero_id="h")
        high_result = calculator.evaluate(high_quest, "a", hero_id="h")

        # Higher difficulty should give more XP
        assert high_result.xp_awarded["logic_weaving"] > low_result.xp_awarded["logic_weaving"]

    def test_consecutive_miss_tracking(self, calculator, sample_quest):
        # Multiple misses
        for _ in range(5):
            result = calculator.evaluate(sample_quest, "wrong", hero_id="h")

        # Should trigger confusion effect
        assert "confusion" in result.effects_triggered

    def test_crit_miss_tracking(self, calculator):
        quest = QuestInstance(
            id="json_quest",
            template_id="test",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Return JSON",
            expected={"answer": "test"},
            metadata={"evaluator_id": "json"},
        )

        # Multiple critical misses (invalid JSON)
        for _ in range(3):
            result = calculator.evaluate(quest, "not json", hero_id="h")

        assert result.combat_result == CombatResult.CRITICAL_MISS
        assert "curse_of_repetition" in result.effects_triggered

    def test_stance_alternating(self, calculator):
        calculator.set_stance(CombatStance.ALTERNATING)

        assert calculator.should_use_thinking(0) is True
        assert calculator.should_use_thinking(1) is False
        assert calculator.should_use_thinking(2) is True

    def test_stance_thoughtful(self, calculator):
        calculator.set_stance(CombatStance.THOUGHTFUL)

        assert calculator.should_use_thinking(0) is True
        assert calculator.should_use_thinking(1) is True

    def test_stance_quick_draw(self, calculator):
        calculator.set_stance(CombatStance.QUICK_DRAW)

        assert calculator.should_use_thinking(0) is False
        assert calculator.should_use_thinking(1) is False


class TestSylloEvaluator:
    @pytest.fixture
    def syllo_quest(self):
        return QuestInstance(
            id="syllo_test",
            template_id="syllo",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Find the hidden word",
            expected={"answer": "elephant"},
        )

    def test_exact_match(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest, "elephant", {}
        )
        assert result == CombatResult.HIT
        assert metrics["correct"] == 1.0

    def test_json_format(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest, '{"answer": "elephant"}', {}
        )
        assert result == CombatResult.HIT

    def test_answer_prefix(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest, "The answer is: elephant", {}
        )
        assert result == CombatResult.HIT

    def test_with_reasoning(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest,
            "Because the clue says 'large mammal' and the syllable is 'el', the answer is elephant",
            {}
        )
        assert result == CombatResult.CRITICAL_HIT

    def test_close_match(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest, "elefant", {}  # Typo
        )
        assert result == CombatResult.GLANCING

    def test_wrong_answer(self, syllo_quest):
        result, metrics, notes = syllo_evaluator(
            syllo_quest, "giraffe", {}
        )
        assert result == CombatResult.MISS


class TestDiscriminationEvaluator:
    def test_correct_identification(self):
        quest = QuestInstance(
            id="disc_test",
            template_id="disc",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Is this correct?",
            expected={"judgment": "CORRECT"},
        )

        result, metrics, notes = discrimination_evaluator(
            quest, "CORRECT", {}
        )
        assert result == CombatResult.HIT

    def test_incorrect_identification(self):
        quest = QuestInstance(
            id="disc_test",
            template_id="disc",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Is this correct?",
            expected={"judgment": "INCORRECT", "correction": "elephant"},
        )

        result, metrics, notes = discrimination_evaluator(
            quest, "INCORRECT. The answer should be elephant.", {}
        )
        assert result == CombatResult.CRITICAL_HIT

    def test_wrong_judgment(self):
        quest = QuestInstance(
            id="disc_test",
            template_id="disc",
            skills=[],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Is this correct?",
            expected={"judgment": "CORRECT"},
        )

        result, metrics, notes = discrimination_evaluator(
            quest, "INCORRECT", {}
        )
        assert result == CombatResult.MISS
```

**Dependencies:** P7.1, P7.2

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_combat.py -v` passes all tests

**Effort:** M (35 min)

---

### P7.5 - Commit Phase 7

**Description:** Commit combat calculator

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 7 - Combat Calculator

- guild/combat/calculator.py: CombatCalculator with evaluator registration
- guild/combat/evaluators.py: SYLLO, discrimination, format evaluators
- tests/guild/test_combat.py: Combat system tests

XP calculation, stance management, consecutive miss tracking"
git tag guild-p7-complete
```

**Dependencies:** P7.1-P7.4

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p7-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 8: Incidents System

**Goal:** Structured error tracking and incident management

---

### P8.1 - Create guild/incidents/tracker.py

**Description:** Incident detection and tracking system

**File:** `guild/incidents/tracker.py`

```python
"""Incident tracker - detects and logs system problems."""

from typing import Optional, Callable, Any
from datetime import datetime
from pathlib import Path
import json

from guild.incidents.types import (
    Incident, IncidentCategory, IncidentStatus, IncidentRule
)
from guild.types import Severity, generate_id


# Detector function signature
DetectorFn = Callable[[dict], Optional[dict]]  # Returns incident data or None


class IncidentTracker:
    """
    Detects and tracks system incidents (errors, anomalies).
    """

    def __init__(self, persist_dir: Optional[Path | str] = None):
        self._incidents: dict[str, Incident] = {}
        self._rules: dict[str, IncidentRule] = {}
        self._detectors: dict[str, DetectorFn] = {}

        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def register_rule(self, rule: IncidentRule):
        """Register an incident detection rule."""
        self._rules[rule.id] = rule

    def register_detector(self, detector_type: str, fn: DetectorFn):
        """Register a detector function."""
        self._detectors[detector_type] = fn

    def check(self, context: dict, step: int = 0,
              run_id: Optional[str] = None) -> list[Incident]:
        """
        Check all rules against current context.
        Returns list of new incidents detected.
        """
        new_incidents = []

        for rule_id, rule in self._rules.items():
            detector = self._detectors.get(rule.detector_type)
            if detector is None:
                continue

            try:
                result = detector(context)
                if result:
                    # Detector triggered
                    incident = self._create_incident(
                        rule=rule,
                        step=step,
                        run_id=run_id,
                        context={**context, **result}
                    )
                    self._incidents[incident.id] = incident
                    new_incidents.append(incident)
                    self._persist_incident(incident)

            except Exception as e:
                # Detector error - create meta-incident
                error_incident = Incident(
                    id=generate_id("inc"),
                    category=IncidentCategory.LOGIC,
                    severity=Severity.LOW,
                    title=f"Detector Error: {rule.detector_type}",
                    description=str(e),
                    detected_at_step=step,
                    run_id=run_id,
                    context={"rule_id": rule_id, "error": str(e)},
                )
                self._incidents[error_incident.id] = error_incident

        return new_incidents

    def _create_incident(self, rule: IncidentRule, step: int,
                         run_id: Optional[str], context: dict) -> Incident:
        """Create an incident from a triggered rule."""
        # Format title/description with context
        title = rule.title_template.format(**context) if rule.title_template else rule.name
        description = rule.description_template.format(**context) if rule.description_template else ""
        rpg_name = rule.rpg_name_template.format(**context) if rule.rpg_name_template else None

        return Incident(
            id=generate_id("inc"),
            category=rule.category,
            severity=rule.severity,
            title=title,
            description=description,
            detected_at_step=step,
            run_id=run_id,
            context=context,
            rpg_name=rpg_name,
        )

    def report(self, category: IncidentCategory, severity: Severity,
               title: str, description: str = "",
               step: int = 0, run_id: Optional[str] = None,
               context: Optional[dict] = None) -> Incident:
        """Manually report an incident."""
        incident = Incident(
            id=generate_id("inc"),
            category=category,
            severity=severity,
            title=title,
            description=description,
            detected_at_step=step,
            run_id=run_id,
            context=context or {},
        )
        self._incidents[incident.id] = incident
        self._persist_incident(incident)
        return incident

    def resolve(self, incident_id: str, resolution: str):
        """Mark an incident as resolved."""
        if incident_id in self._incidents:
            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.RESOLVED
            incident.resolution = resolution
            incident.resolved_at = datetime.now()
            self._persist_incident(incident)

    def get(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self._incidents.get(incident_id)

    def list(self, status: Optional[IncidentStatus] = None,
             category: Optional[IncidentCategory] = None,
             severity: Optional[Severity] = None,
             run_id: Optional[str] = None) -> list[Incident]:
        """List incidents with optional filters."""
        incidents = list(self._incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]
        if category:
            incidents = [i for i in incidents if i.category == category]
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        if run_id:
            incidents = [i for i in incidents if i.run_id == run_id]

        return sorted(incidents, key=lambda i: i.detected_at_time, reverse=True)

    def open_incidents(self) -> list[Incident]:
        """Get all open (unresolved) incidents."""
        return self.list(status=IncidentStatus.OPEN)

    def critical_incidents(self) -> list[Incident]:
        """Get all critical severity incidents."""
        return self.list(severity=Severity.CRITICAL)

    # Persistence

    def _persist_incident(self, incident: Incident):
        """Persist an incident to disk."""
        if not self._persist_dir:
            return

        path = self._persist_dir / f"{incident.id}.json"
        path.write_text(json.dumps(incident.to_dict(), indent=2))

    def load_incidents(self):
        """Load incidents from disk."""
        if not self._persist_dir:
            return

        for path in self._persist_dir.glob("inc_*.json"):
            try:
                data = json.loads(path.read_text())
                incident = Incident.from_dict(data)
                self._incidents[incident.id] = incident
            except Exception as e:
                print(f"Warning: Failed to load incident {path}: {e}")

    # Stats

    def stats(self) -> dict:
        """Get incident statistics."""
        all_incidents = list(self._incidents.values())

        by_status = {}
        for status in IncidentStatus:
            by_status[status.value] = sum(1 for i in all_incidents if i.status == status)

        by_severity = {}
        for sev in Severity:
            by_severity[sev.value] = sum(1 for i in all_incidents if i.severity == sev)

        by_category = {}
        for cat in IncidentCategory:
            by_category[cat.value] = sum(1 for i in all_incidents if i.category == cat)

        return {
            "total": len(all_incidents),
            "by_status": by_status,
            "by_severity": by_severity,
            "by_category": by_category,
            "open_count": by_status.get("open", 0),
            "critical_count": by_severity.get("critical", 0),
        }


# Global tracker
_tracker: Optional[IncidentTracker] = None


def get_incident_tracker(persist_dir: Optional[Path | str] = None) -> IncidentTracker:
    """Get the global incident tracker."""
    global _tracker
    if _tracker is None:
        _tracker = IncidentTracker(persist_dir)
        _register_builtin_detectors(_tracker)
    return _tracker


def reset_incident_tracker():
    """Reset the global tracker."""
    global _tracker
    _tracker = None


def _register_builtin_detectors(tracker: IncidentTracker):
    """Register built-in detectors."""

    def nan_detector(context: dict) -> Optional[dict]:
        """Detect NaN values in metrics."""
        import math
        for key, value in context.items():
            if isinstance(value, float) and math.isnan(value):
                return {"metric": key, "value": "NaN"}
        return None

    def oom_detector(context: dict) -> Optional[dict]:
        """Detect OOM indicators."""
        error = context.get("error", "")
        if "CUDA out of memory" in error or "OOM" in error:
            return {"error_type": "OOM", "message": error[:200]}
        return None

    def high_loss_detector(context: dict) -> Optional[dict]:
        """Detect abnormally high loss."""
        loss = context.get("loss")
        if loss is not None and loss > 100:
            return {"loss": loss, "threshold": 100}
        return None

    tracker.register_detector("nan", nan_detector)
    tracker.register_detector("oom", oom_detector)
    tracker.register_detector("high_loss", high_loss_detector)


def report_incident(category: IncidentCategory, severity: Severity,
                    title: str, **kwargs) -> Incident:
    """Report an incident using the global tracker."""
    return get_incident_tracker().report(category, severity, title, **kwargs)
```

**Dependencies:** P1.7

**Acceptance Criteria:**
- [ ] `from guild.incidents.tracker import get_incident_tracker, report_incident` works
- [ ] Detectors can be registered and triggered
- [ ] Incidents persist to disk
- [ ] Filtering works

**Effort:** L (40 min)

---

### P8.2 - Create configs/incidents/rules.yaml

**Description:** Incident detection rules

**File:** `configs/incidents/rules.yaml`

```yaml
# Incident detection rules

rules:
  - id: detect_nan_loss
    name: NaN Loss Detection
    category: training
    severity: critical
    detector_type: nan
    detector_config:
      metrics:
        - loss
        - val_loss
    title_template: "Reality Tear: NaN {metric}"
    description_template: "Training metric {metric} became NaN"
    rpg_name_template: "Reality Tear"

  - id: detect_oom
    name: OOM Detection
    category: infra
    severity: high
    detector_type: oom
    detector_config: {}
    title_template: "Exhaustion: Out of Memory"
    description_template: "{message}"
    rpg_name_template: "Hero Exhaustion"

  - id: detect_high_loss
    name: High Loss Detection
    category: training
    severity: medium
    detector_type: high_loss
    detector_config:
      threshold: 100
    title_template: "Training Instability: Loss = {loss:.2f}"
    description_template: "Training loss exceeded threshold of {threshold}"
    rpg_name_template: "Training Turbulence"

  - id: detect_overfitting
    name: Overfitting Detection
    category: training
    severity: medium
    detector_type: metric_gap
    detector_config:
      metric_a: train_accuracy
      metric_b: val_accuracy
      max_gap: 0.3
    title_template: "Tunnel Vision: Gap = {gap:.2%}"
    description_template: "Train/val accuracy gap ({gap:.2%}) exceeds threshold"
    rpg_name_template: "Tunnel Vision"

  - id: detect_data_corruption
    name: Data Corruption Detection
    category: data
    severity: high
    detector_type: pattern_match
    detector_config:
      patterns:
        - "\\x00"  # Null bytes
        - "(?:user|assistant)\\s*:\\s*$"  # Empty turns
    title_template: "Cursed Scroll Detected"
    description_template: "Data file contains corruption pattern"
    rpg_name_template: "Cursed Scroll"

  - id: detect_repetition
    name: Repetition Detection
    category: training
    severity: medium
    detector_type: pattern_match
    detector_config:
      patterns:
        - "(\\b\\w+\\b)(?:\\s+\\1){4,}"  # Word repeated 5+ times
    title_template: "Curse of Repetition"
    description_template: "Output contains degenerate repetition"
    rpg_name_template: "Curse of Repetition"
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File is valid YAML
- [ ] Contains at least 5 rules

**Effort:** M (20 min)

---

### P8.3 - Update guild/incidents/__init__.py

**Description:** Export incidents components

**File:** `guild/incidents/__init__.py`

```python
"""Incidents system - error tracking and management."""

from guild.incidents.types import (
    Incident,
    IncidentCategory,
    IncidentStatus,
    IncidentRule,
)
from guild.incidents.tracker import (
    IncidentTracker,
    get_incident_tracker,
    reset_incident_tracker,
    report_incident,
)

__all__ = [
    # Types
    "Incident",
    "IncidentCategory",
    "IncidentStatus",
    "IncidentRule",
    # Tracker
    "IncidentTracker",
    "get_incident_tracker",
    "reset_incident_tracker",
    "report_incident",
]
```

**Dependencies:** P8.1

**Acceptance Criteria:**
- [ ] `from guild.incidents import report_incident, IncidentTracker` works

**Effort:** S (5 min)

---

### P8.4 - Create tests/guild/test_incidents.py

**Description:** Tests for incident tracker

**File:** `tests/guild/test_incidents.py`

```python
"""Tests for incident tracker."""

import pytest
import tempfile
from pathlib import Path
import math

from guild.incidents.tracker import (
    IncidentTracker, get_incident_tracker, reset_incident_tracker
)
from guild.incidents.types import (
    Incident, IncidentCategory, IncidentStatus, IncidentRule
)
from guild.types import Severity


class TestIncidentTracker:
    @pytest.fixture
    def tracker(self):
        reset_incident_tracker()
        return IncidentTracker()

    def test_report_incident(self, tracker):
        incident = tracker.report(
            category=IncidentCategory.TRAINING,
            severity=Severity.HIGH,
            title="Test Incident",
            description="Something went wrong",
            step=100,
        )

        assert incident.id.startswith("inc_")
        assert incident.category == IncidentCategory.TRAINING
        assert incident.severity == Severity.HIGH
        assert incident.status == IncidentStatus.OPEN

    def test_resolve_incident(self, tracker):
        incident = tracker.report(
            category=IncidentCategory.DATA,
            severity=Severity.LOW,
            title="Minor Issue",
        )

        tracker.resolve(incident.id, "Fixed the thing")

        resolved = tracker.get(incident.id)
        assert resolved.status == IncidentStatus.RESOLVED
        assert resolved.resolution == "Fixed the thing"
        assert resolved.resolved_at is not None

    def test_list_incidents(self, tracker):
        # Create various incidents
        tracker.report(IncidentCategory.TRAINING, Severity.HIGH, "Training 1")
        tracker.report(IncidentCategory.TRAINING, Severity.LOW, "Training 2")
        tracker.report(IncidentCategory.DATA, Severity.HIGH, "Data 1")

        all_incidents = tracker.list()
        assert len(all_incidents) == 3

        training = tracker.list(category=IncidentCategory.TRAINING)
        assert len(training) == 2

        high_sev = tracker.list(severity=Severity.HIGH)
        assert len(high_sev) == 2

    def test_nan_detector(self, tracker):
        result = tracker.check({
            "loss": float('nan'),
            "accuracy": 0.85,
        }, step=500)

        assert len(result) == 1
        assert "NaN" in result[0].title or "nan" in result[0].context.get("value", "")

    def test_oom_detector(self, tracker):
        result = tracker.check({
            "error": "CUDA out of memory. Tried to allocate 2.00 GiB",
        }, step=100)

        assert len(result) == 1
        assert "OOM" in result[0].context.get("error_type", "")

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tracker with persistence
            tracker1 = IncidentTracker(persist_dir=tmpdir)
            tracker1.report(
                IncidentCategory.TRAINING,
                Severity.HIGH,
                "Persistent Incident"
            )

            # Create new tracker, load from disk
            tracker2 = IncidentTracker(persist_dir=tmpdir)
            tracker2.load_incidents()

            assert len(tracker2.list()) == 1
            assert tracker2.list()[0].title == "Persistent Incident"

    def test_stats(self, tracker):
        tracker.report(IncidentCategory.TRAINING, Severity.CRITICAL, "Crit 1")
        tracker.report(IncidentCategory.TRAINING, Severity.HIGH, "High 1")
        incident = tracker.report(IncidentCategory.DATA, Severity.LOW, "Low 1")
        tracker.resolve(incident.id, "Fixed")

        stats = tracker.stats()
        assert stats["total"] == 3
        assert stats["critical_count"] == 1
        assert stats["by_status"]["open"] == 2
        assert stats["by_status"]["resolved"] == 1

    def test_custom_detector(self, tracker):
        def custom_detector(context):
            if context.get("custom_metric", 0) > 100:
                return {"value": context["custom_metric"]}
            return None

        tracker.register_detector("custom", custom_detector)

        rule = IncidentRule(
            id="custom_rule",
            name="Custom Rule",
            category=IncidentCategory.LOGIC,
            severity=Severity.MEDIUM,
            detector_type="custom",
            title_template="Custom: {value}",
            description_template="Custom metric exceeded",
        )
        tracker.register_rule(rule)

        result = tracker.check({"custom_metric": 150}, step=0)
        assert len(result) == 1
        assert "150" in result[0].title


class TestGlobalTracker:
    def test_global_tracker(self):
        reset_incident_tracker()

        tracker1 = get_incident_tracker()
        tracker2 = get_incident_tracker()

        assert tracker1 is tracker2

    def test_builtin_detectors_registered(self):
        reset_incident_tracker()
        tracker = get_incident_tracker()

        # Should have nan and oom detectors
        assert "nan" in tracker._detectors
        assert "oom" in tracker._detectors
```

**Dependencies:** P8.1

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_incidents.py -v` passes all tests

**Effort:** M (30 min)

---

### P8.5 - Commit Phase 8

**Description:** Commit incidents system

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 8 - Incidents System

- guild/incidents/tracker.py: IncidentTracker with detector registration
- configs/incidents/rules.yaml: Detection rules
- tests/guild/test_incidents.py: Incident tests

Built-in detectors: NaN, OOM, high loss. Persistence support."
git tag guild-p8-complete
```

**Dependencies:** P8.1-P8.4

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p8-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 9: Runs System

**Goal:** Unified run management for training, evaluation, and generation

---

### P9.1 - Create guild/runs/runner.py

**Description:** Run manager coordinates quest execution

**File:** `guild/runs/runner.py`

```python
"""Run manager - coordinates quest execution."""

from typing import Optional, Iterator, Callable
from datetime import datetime
from pathlib import Path
import json

from guild.runs.types import RunConfig, RunState, RunType
from guild.types import Status, generate_id
from guild.quests.types import QuestInstance, QuestResult
from guild.quests.board import QuestBoard
from guild.progression.engine import ProgressionEngine
from guild.combat.calculator import CombatCalculator
from guild.incidents.tracker import IncidentTracker


# Hook signatures
PreQuestHook = Callable[[QuestInstance, RunState], Optional[QuestInstance]]
PostQuestHook = Callable[[QuestResult, RunState], None]
CheckpointHook = Callable[[RunState, int], None]


class RunManager:
    """
    Manages run lifecycle: setup, execution, checkpointing, completion.
    """

    def __init__(self,
                 quest_board: Optional[QuestBoard] = None,
                 progression: Optional[ProgressionEngine] = None,
                 combat: Optional[CombatCalculator] = None,
                 incidents: Optional[IncidentTracker] = None,
                 persist_dir: Optional[Path | str] = None):

        self.board = quest_board
        self.progression = progression
        self.combat = combat
        self.incidents = incidents

        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._active_runs: dict[str, RunState] = {}

        # Hooks
        self._pre_quest_hooks: list[PreQuestHook] = []
        self._post_quest_hooks: list[PostQuestHook] = []
        self._checkpoint_hooks: list[CheckpointHook] = []

    # Run lifecycle

    def create_run(self, config: RunConfig) -> RunState:
        """Create a new run from config."""
        run_id = config.id or generate_id("run")
        state = RunState(
            run_id=run_id,
            config=config,
            status=Status.PENDING,
        )
        self._active_runs[run_id] = state
        return state

    def start_run(self, run_id: str) -> RunState:
        """Start a pending run."""
        state = self._active_runs.get(run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")

        state.status = Status.ACTIVE
        state.started_at = datetime.now()

        if self.progression:
            self.progression.hero.current_run_id = run_id

        self._persist_run(state)
        return state

    def pause_run(self, run_id: str) -> RunState:
        """Pause an active run."""
        state = self._active_runs.get(run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")

        state.status = Status.PAUSED
        state.paused_at = datetime.now()
        self._persist_run(state)
        return state

    def resume_run(self, run_id: str) -> RunState:
        """Resume a paused run."""
        state = self._active_runs.get(run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")

        state.status = Status.ACTIVE
        state.paused_at = None
        self._persist_run(state)
        return state

    def complete_run(self, run_id: str) -> RunState:
        """Mark a run as completed."""
        state = self._active_runs.get(run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")

        state.status = Status.COMPLETED
        state.completed_at = datetime.now()
        self._persist_run(state)
        return state

    def fail_run(self, run_id: str, error: str) -> RunState:
        """Mark a run as failed."""
        state = self._active_runs.get(run_id)
        if state is None:
            raise ValueError(f"Unknown run: {run_id}")

        state.status = Status.FAILED
        state.completed_at = datetime.now()
        state.metrics["error"] = error
        self._persist_run(state)
        return state

    # Quest execution

    def execute_quest(self, quest: QuestInstance, response: str,
                      run_id: Optional[str] = None,
                      response_metadata: Optional[dict] = None) -> QuestResult:
        """
        Execute a quest and process the result.

        Args:
            quest: The quest to execute
            response: Hero's response
            run_id: Optional run ID for context
            response_metadata: Additional response info

        Returns:
            QuestResult after evaluation and XP award
        """
        state = self._active_runs.get(run_id) if run_id else None

        # Pre-quest hooks
        for hook in self._pre_quest_hooks:
            modified = hook(quest, state)
            if modified:
                quest = modified

        # Evaluate
        hero_id = self.progression.hero.hero_id if self.progression else ""
        result = self.combat.evaluate(
            quest, response,
            hero_id=hero_id,
            response_metadata=response_metadata
        ) if self.combat else QuestResult(
            quest_id=quest.id,
            hero_id=hero_id,
            response=response,
            combat_result=CombatResult.HIT,
        )

        # Award XP
        if self.progression:
            self.progression.award_xp(result)

        # Complete quest on board
        if self.board:
            self.board.complete(result)

        # Update run state
        if state:
            state.quests_completed += 1
            if result.success:
                state.quests_succeeded += 1
            state.current_step += 1

            # Check for checkpoint
            if state.current_step % state.config.checkpoint_every_steps == 0:
                self._checkpoint(state)

        # Post-quest hooks
        for hook in self._post_quest_hooks:
            hook(result, state)

        # Check incidents
        if self.incidents and state:
            context = {
                "step": state.current_step,
                "result": result.combat_result.value,
                **result.metrics,
            }
            new_incidents = self.incidents.check(context, state.current_step, run_id)
            state.incident_ids.extend(i.id for i in new_incidents)

        return result

    def _checkpoint(self, state: RunState):
        """Create a checkpoint for a run."""
        state.last_checkpoint_step = state.current_step

        # Run checkpoint hooks
        for hook in self._checkpoint_hooks:
            hook(state, state.current_step)

        self._persist_run(state)

    # Hook registration

    def add_pre_quest_hook(self, hook: PreQuestHook):
        """Add a pre-quest hook."""
        self._pre_quest_hooks.append(hook)

    def add_post_quest_hook(self, hook: PostQuestHook):
        """Add a post-quest hook."""
        self._post_quest_hooks.append(hook)

    def add_checkpoint_hook(self, hook: CheckpointHook):
        """Add a checkpoint hook."""
        self._checkpoint_hooks.append(hook)

    # Persistence

    def _persist_run(self, state: RunState):
        """Persist run state to disk."""
        if not self._persist_dir:
            return

        path = self._persist_dir / f"{state.run_id}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2))

    def load_run(self, run_id: str) -> Optional[RunState]:
        """Load a run from disk."""
        if not self._persist_dir:
            return None

        path = self._persist_dir / f"{run_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        state = RunState.from_dict(data)
        self._active_runs[run_id] = state
        return state

    # Queries

    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get a run by ID."""
        return self._active_runs.get(run_id)

    def list_runs(self, status: Optional[Status] = None,
                  run_type: Optional[RunType] = None) -> list[RunState]:
        """List runs with optional filters."""
        runs = list(self._active_runs.values())

        if status:
            runs = [r for r in runs if r.status == status]
        if run_type:
            runs = [r for r in runs if r.config.type == run_type]

        return runs

    def active_runs(self) -> list[RunState]:
        """Get all active runs."""
        return self.list_runs(status=Status.ACTIVE)


# Import for type checking
from guild.quests.types import CombatResult


# Global manager
_manager: Optional[RunManager] = None


def get_run_manager(**kwargs) -> RunManager:
    """Get the global run manager."""
    global _manager
    if _manager is None:
        _manager = RunManager(**kwargs)
    return _manager


def reset_run_manager():
    """Reset the global manager."""
    global _manager
    _manager = None
```

**Dependencies:** P1.6, P5.3, P6.1, P7.1, P8.1

**Acceptance Criteria:**
- [ ] `from guild.runs.runner import RunManager, get_run_manager` works
- [ ] Run lifecycle (create/start/pause/complete) works
- [ ] Quest execution integrates with all systems
- [ ] Hooks can be registered

**Effort:** L (50 min)

---

### P9.2 - Update guild/runs/__init__.py

**Description:** Export runs components

**File:** `guild/runs/__init__.py`

```python
"""Runs system - unified execution management."""

from guild.runs.types import (
    RunConfig,
    RunState,
    RunType,
)
from guild.runs.runner import (
    RunManager,
    get_run_manager,
    reset_run_manager,
)

__all__ = [
    # Types
    "RunConfig",
    "RunState",
    "RunType",
    # Manager
    "RunManager",
    "get_run_manager",
    "reset_run_manager",
]
```

**Dependencies:** P9.1

**Acceptance Criteria:**
- [ ] `from guild.runs import RunManager, RunConfig` works

**Effort:** S (5 min)

---

### P9.3 - Create tests/guild/test_runs.py

**Description:** Tests for run manager

**File:** `tests/guild/test_runs.py`

```python
"""Tests for run manager."""

import pytest
import tempfile
from pathlib import Path

from guild.runs.runner import RunManager, reset_run_manager
from guild.runs.types import RunConfig, RunState, RunType
from guild.types import Status
from guild.quests.types import QuestInstance, QuestDifficulty, CombatResult
from guild.quests.board import QuestBoard
from guild.combat.calculator import CombatCalculator
from guild.progression.engine import create_progression_engine
from guild.progression.types import HeroIdentity


class TestRunManager:
    @pytest.fixture
    def manager(self):
        reset_run_manager()

        identity = HeroIdentity(
            id="test_hero", name="Test",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )
        progression = create_progression_engine("test_hero", identity)

        return RunManager(
            quest_board=QuestBoard(),
            progression=progression,
            combat=CombatCalculator(),
        )

    @pytest.fixture
    def sample_config(self):
        return RunConfig(
            id="test_run",
            type=RunType.TRAINING,
            name="Test Training Run",
            max_quests=100,
            checkpoint_every_steps=10,
        )

    def test_create_run(self, manager, sample_config):
        state = manager.create_run(sample_config)

        assert state.run_id == "test_run"
        assert state.status == Status.PENDING
        assert state.config.type == RunType.TRAINING

    def test_run_lifecycle(self, manager, sample_config):
        state = manager.create_run(sample_config)

        # Start
        state = manager.start_run("test_run")
        assert state.status == Status.ACTIVE
        assert state.started_at is not None

        # Pause
        state = manager.pause_run("test_run")
        assert state.status == Status.PAUSED
        assert state.paused_at is not None

        # Resume
        state = manager.resume_run("test_run")
        assert state.status == Status.ACTIVE
        assert state.paused_at is None

        # Complete
        state = manager.complete_run("test_run")
        assert state.status == Status.COMPLETED
        assert state.completed_at is not None

    def test_execute_quest(self, manager, sample_config):
        manager.create_run(sample_config)
        manager.start_run("test_run")

        quest = QuestInstance(
            id="quest_1",
            template_id="test",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Test",
            expected={"answer": "correct"},
        )

        result = manager.execute_quest(
            quest,
            response="correct",
            run_id="test_run"
        )

        assert result.combat_result == CombatResult.HIT
        assert result.success is True

        state = manager.get_run("test_run")
        assert state.quests_completed == 1
        assert state.quests_succeeded == 1

    def test_hooks(self, manager, sample_config):
        pre_quest_called = []
        post_quest_called = []

        def pre_hook(quest, state):
            pre_quest_called.append(quest.id)
            return quest

        def post_hook(result, state):
            post_quest_called.append(result.quest_id)

        manager.add_pre_quest_hook(pre_hook)
        manager.add_post_quest_hook(post_hook)

        manager.create_run(sample_config)
        manager.start_run("test_run")

        quest = QuestInstance(
            id="hook_test",
            template_id="test",
            skills=[],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Test",
        )

        manager.execute_quest(quest, "response", run_id="test_run")

        assert "hook_test" in pre_quest_called
        assert "hook_test" in post_quest_called

    def test_checkpoint(self, manager):
        config = RunConfig(
            id="checkpoint_test",
            type=RunType.TRAINING,
            checkpoint_every_steps=2,
        )

        checkpoint_steps = []

        def checkpoint_hook(state, step):
            checkpoint_steps.append(step)

        manager.add_checkpoint_hook(checkpoint_hook)

        manager.create_run(config)
        manager.start_run("checkpoint_test")

        quest = QuestInstance(
            id="q", template_id="t", skills=[],
            difficulty=QuestDifficulty.BRONZE, difficulty_level=1,
            prompt="Test",
        )

        # Execute 4 quests
        for _ in range(4):
            manager.execute_quest(quest, "r", run_id="checkpoint_test")

        # Should checkpoint at steps 2 and 4
        assert 2 in checkpoint_steps
        assert 4 in checkpoint_steps

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = RunManager(persist_dir=tmpdir)

            config = RunConfig(id="persist_run", type=RunType.TRAINING)
            manager1.create_run(config)
            manager1.start_run("persist_run")

            # New manager, load from disk
            manager2 = RunManager(persist_dir=tmpdir)
            state = manager2.load_run("persist_run")

            assert state is not None
            assert state.run_id == "persist_run"
            assert state.status == Status.ACTIVE

    def test_list_runs(self, manager, sample_config):
        # Create multiple runs
        config1 = RunConfig(id="run1", type=RunType.TRAINING)
        config2 = RunConfig(id="run2", type=RunType.EVALUATION)
        config3 = RunConfig(id="run3", type=RunType.TRAINING)

        manager.create_run(config1)
        manager.create_run(config2)
        manager.create_run(config3)

        manager.start_run("run1")
        manager.start_run("run2")

        active = manager.active_runs()
        assert len(active) == 2

        training = manager.list_runs(run_type=RunType.TRAINING)
        assert len(training) == 2
```

**Dependencies:** P9.1

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_runs.py -v` passes all tests

**Effort:** M (35 min)

---

### P9.4 - Commit Phase 9

**Description:** Commit runs system

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 9 - Runs System

- guild/runs/runner.py: RunManager with lifecycle and hooks
- tests/guild/test_runs.py: Run management tests

Integrates quest board, progression, combat, incidents.
Hook support for pre/post quest and checkpoints."
git tag guild-p9-complete
```

**Dependencies:** P9.1-P9.3

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p9-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Checkpoint: Validate Phase 7-9

```bash
# All guild tests pass
pytest tests/guild/ -v

# Verify combat system
python -c "
from guild.combat import get_combat_calculator, evaluate_quest, CombatStance
from guild.quests.types import QuestInstance, QuestDifficulty

calc = get_combat_calculator()
print(f'Evaluators: {calc.list_evaluators()}')
print(f'Stance: {calc.get_stance()}')
"

# Verify incidents
python -c "
from guild.incidents import get_incident_tracker, report_incident, IncidentCategory
from guild.types import Severity

tracker = get_incident_tracker()
inc = tracker.report(IncidentCategory.TRAINING, Severity.LOW, 'Test')
print(f'Incident created: {inc.id}')
print(f'Stats: {tracker.stats()}')
"

# Verify runs
python -c "
from guild.runs import RunManager, RunConfig, RunType

manager = RunManager()
config = RunConfig(id='test', type=RunType.TRAINING)
state = manager.create_run(config)
print(f'Run created: {state.run_id}')
"
```

**Decision Point:**
- All tests pass → Continue to Phase 10
- Issues found → Fix before proceeding

---

**Total Tasks in Phases 7-9:** 14 tasks
**Estimated Time:** 1-2 weeks
