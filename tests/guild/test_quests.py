"""Tests for quest loading, registry, forge, and evaluation."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import os
from datetime import datetime

from guild.quests.types import (
    QuestDifficulty,
    CombatResult,
    QuestTemplate,
    QuestInstance,
    QuestResult,
)
from guild.quests.loader import (
    load_quest_template,
    discover_quest_templates,
    load_all_quest_templates,
    QuestLoader,
    _dict_to_quest_template,
)
from guild.quests.registry import (
    QuestRegistry,
    init_quest_registry,
    get_quest_registry,
    reset_quest_registry,
    get_quest,
    list_quests,
)
from guild.quests.forge import (
    QuestGenerator,
    StaticGenerator,
    CallbackGenerator,
    QuestForge,
    get_forge,
    reset_forge,
    create_quest,
)
from guild.quests.evaluator import (
    EvaluationContext,
    EvaluationOutcome,
    ExactMatchEvaluator,
    ContainsEvaluator,
    QuestJudge,
    get_judge,
    reset_judge,
    evaluate_quest,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test quests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        quests_dir = config_dir / "quests"
        reasoning_dir = quests_dir / "reasoning"
        reasoning_dir.mkdir(parents=True)

        # Create test quest template
        logic_quest = """
id: test_logic_quest
name: Test Logic Quest
description: A test quest for logic

skills:
  - logic_weaving

regions:
  - test_realm

difficulty_level: 3
difficulty: silver

generator_id: static
generator_params:
  prompt: "What is 2 + 2?"
  expected: "4"

evaluator_id: exact_match
evaluator_params:
  case_sensitive: false

base_xp:
  logic_weaving: 15

tags:
  - test
  - logic

enabled: true
"""
        (reasoning_dir / "test_logic_quest.yaml").write_text(logic_quest)

        # Create another quest at root level
        math_quest = """
id: test_math_quest
name: Test Math Quest
description: A simple math quest
skills:
  - math_prowess
difficulty_level: 1
generator_id: static
evaluator_id: contains
enabled: true
"""
        (quests_dir / "test_math_quest.yaml").write_text(math_quest)

        yield config_dir


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_quest_registry()
    reset_forge()
    reset_judge()
    yield
    reset_quest_registry()
    reset_forge()
    reset_judge()


# =============================================================================
# Type Tests
# =============================================================================

class TestQuestTypes:
    """Tests for quest type definitions."""

    def test_difficulty_from_level(self):
        assert QuestDifficulty.from_level(1) == QuestDifficulty.BRONZE
        assert QuestDifficulty.from_level(3) == QuestDifficulty.SILVER
        assert QuestDifficulty.from_level(5) == QuestDifficulty.GOLD
        assert QuestDifficulty.from_level(7) == QuestDifficulty.PLATINUM
        assert QuestDifficulty.from_level(10) == QuestDifficulty.DRAGON

    def test_combat_result_enum(self):
        assert CombatResult.CRITICAL_HIT.value == "crit"
        assert CombatResult.HIT.value == "hit"
        assert CombatResult.MISS.value == "miss"

    def test_quest_template_creation(self):
        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="static",
            evaluator_id="exact_match",
        )
        assert template.id == "test"
        assert template.enabled is True

    def test_quest_instance_create(self):
        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="static",
            evaluator_id="exact_match",
        )

        instance = QuestInstance.create(
            template=template,
            prompt="Test prompt",
            expected={"answer": "test"},
        )

        assert instance.template_id == "test"
        assert instance.prompt == "Test prompt"
        assert instance.skills == ["logic"]
        assert instance.id.startswith("quest_")

    def test_quest_instance_serialization(self):
        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=2,
            generator_id="static",
            evaluator_id="exact_match",
        )

        instance = QuestInstance.create(
            template=template,
            prompt="Test prompt",
            expected={"answer": "42"},
        )

        # Serialize
        data = instance.to_dict()
        assert data["template_id"] == "test"
        assert data["difficulty"] == 1  # BRONZE value

        # Deserialize
        restored = QuestInstance.from_dict(data)
        assert restored.template_id == "test"
        assert restored.prompt == "Test prompt"
        assert restored.difficulty == QuestDifficulty.BRONZE

    def test_quest_result_success_property(self):
        result_hit = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.HIT,
        )
        assert result_hit.success is True

        result_crit = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.CRITICAL_HIT,
        )
        assert result_crit.success is True

        result_miss = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.MISS,
        )
        assert result_miss.success is False

    def test_quest_result_total_xp(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"logic": 10, "math": 5},
        )
        assert result.total_xp == 15


# =============================================================================
# Loader Tests
# =============================================================================

class TestQuestLoader:
    """Tests for quest template loading."""

    def test_load_quest_template(self, temp_config_dir):
        template = load_quest_template("test_logic_quest", "reasoning", temp_config_dir)

        assert template.id == "test_logic_quest"
        assert template.name == "Test Logic Quest"
        assert template.difficulty == QuestDifficulty.SILVER
        assert template.difficulty_level == 3
        assert "logic_weaving" in template.skills
        assert template.generator_id == "static"

    def test_load_quest_not_found(self, temp_config_dir):
        with pytest.raises(FileNotFoundError):
            load_quest_template("nonexistent", None, temp_config_dir)

    def test_discover_quest_templates(self, temp_config_dir):
        templates = discover_quest_templates(temp_config_dir)

        # Should find both quests
        ids = [t[0] for t in templates]
        assert "test_logic_quest" in ids
        assert "test_math_quest" in ids

    def test_load_all_quest_templates(self, temp_config_dir):
        templates = load_all_quest_templates(temp_config_dir)

        assert len(templates) == 2
        assert "test_logic_quest" in templates
        assert "test_math_quest" in templates

    def test_quest_loader_caching(self, temp_config_dir):
        loader = QuestLoader(temp_config_dir)

        # First load
        quest1 = loader.load("test_logic_quest", "reasoning")

        # Second load should be cached
        quest2 = loader.load("test_logic_quest", "reasoning")
        assert quest1 is quest2

        # Clear cache
        loader.clear_cache()
        quest3 = loader.load("test_logic_quest", "reasoning")
        assert quest1 is not quest3

    def test_dict_to_quest_template_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            _dict_to_quest_template({})

    def test_dict_to_quest_template_missing_skills(self):
        with pytest.raises(ValueError, match="at least one skill"):
            _dict_to_quest_template({"id": "test", "skills": []})

    def test_dict_to_quest_template_auto_difficulty(self):
        template = _dict_to_quest_template({
            "id": "test",
            "skills": ["logic"],
            "difficulty_level": 5,
        })
        assert template.difficulty == QuestDifficulty.GOLD


# =============================================================================
# Registry Tests
# =============================================================================

class TestQuestRegistry:
    """Tests for quest registry."""

    def test_registry_get(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        quest = registry.get("test_logic_quest")
        assert quest.id == "test_logic_quest"

    def test_registry_get_unknown(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        with pytest.raises(KeyError, match="Unknown quest"):
            registry.get("nonexistent")

    def test_registry_exists(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        assert registry.exists("test_logic_quest") is True
        assert registry.exists("nonexistent") is False
        assert "test_logic_quest" in registry

    def test_registry_list_ids(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        ids = registry.list_ids()
        assert "test_logic_quest" in ids
        assert "test_math_quest" in ids

    def test_registry_by_skill(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        logic_quests = registry.by_skill("logic_weaving")
        assert len(logic_quests) == 1
        assert logic_quests[0].id == "test_logic_quest"

    def test_registry_by_difficulty_level(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        level_3 = registry.by_difficulty_level(3)
        assert len(level_3) == 1
        assert level_3[0].difficulty_level == 3

        level_3_tolerance = registry.by_difficulty_level(3, tolerance=2)
        assert len(level_3_tolerance) == 2  # Level 1 and 3

    def test_registry_by_tag(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        test_quests = registry.by_tag("test")
        assert len(test_quests) == 1

    def test_registry_search(self, temp_config_dir):
        registry = QuestRegistry(temp_config_dir)

        results = registry.search(skill="logic_weaving", difficulty_level=3)
        assert len(results) == 1
        assert results[0].id == "test_logic_quest"


class TestGlobalQuestRegistry:
    """Tests for global quest registry functions."""

    def test_init_and_get_registry(self, temp_config_dir):
        init_quest_registry(temp_config_dir)

        registry = get_quest_registry()
        assert "test_logic_quest" in registry

    def test_get_quest_convenience(self, temp_config_dir):
        init_quest_registry(temp_config_dir)

        quest = get_quest("test_logic_quest")
        assert quest.id == "test_logic_quest"

    def test_list_quests_convenience(self, temp_config_dir):
        init_quest_registry(temp_config_dir)

        ids = list_quests()
        assert "test_logic_quest" in ids


# =============================================================================
# Forge Tests
# =============================================================================

class TestQuestForge:
    """Tests for quest generation."""

    def test_static_generator(self):
        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="static",
            evaluator_id="exact_match",
            generator_params={"prompt": "Test prompt?", "expected": "answer"},
        )

        generator = StaticGenerator()
        instance = generator.generate(template)

        assert instance.prompt == "Test prompt?"
        assert instance.expected == "answer"
        assert instance.template_id == "test"

    def test_callback_generator(self):
        def my_generator(template, params):
            return (
                f"Solve this {template.name} puzzle",
                {"answer": "42"},
                {"source": "callback"},
            )

        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="callback",
            evaluator_id="exact_match",
        )

        generator = CallbackGenerator(my_generator, "custom")
        instance = generator.generate(template)

        assert "Solve this Test Quest puzzle" in instance.prompt
        assert instance.expected == {"answer": "42"}
        assert instance.context["source"] == "callback"

    def test_quest_forge_create(self, temp_config_dir):
        init_quest_registry(temp_config_dir)
        forge = QuestForge()

        template = get_quest("test_logic_quest")
        instance = forge.create(template)

        assert instance.template_id == "test_logic_quest"
        assert instance.prompt == "What is 2 + 2?"

    def test_quest_forge_create_from_id(self, temp_config_dir):
        init_quest_registry(temp_config_dir)
        forge = QuestForge()

        instance = forge.create_from_id("test_logic_quest")

        assert instance.template_id == "test_logic_quest"

    def test_quest_forge_custom_generator(self):
        forge = QuestForge()

        class CustomGenerator(QuestGenerator):
            generator_id = "custom"

            def generate(self, template, params=None):
                return QuestInstance.create(
                    template=template,
                    prompt="Custom prompt",
                    expected={"answer": "custom"},
                )

        forge.register(CustomGenerator())
        assert "custom" in forge.list_generators()

    def test_quest_forge_create_batch(self, temp_config_dir):
        init_quest_registry(temp_config_dir)
        forge = QuestForge()

        template = get_quest("test_logic_quest")
        instances = forge.create_batch(template, count=3)

        assert len(instances) == 3
        # Each should have unique ID
        ids = [i.id for i in instances]
        assert len(set(ids)) == 3


class TestGlobalForge:
    """Tests for global forge functions."""

    def test_get_forge_auto_init(self):
        forge = get_forge()
        assert forge is not None
        assert "static" in forge.list_generators()

    def test_create_quest_convenience(self, temp_config_dir):
        init_quest_registry(temp_config_dir)

        template = get_quest("test_logic_quest")
        instance = create_quest(template)

        assert instance.template_id == "test_logic_quest"


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestExactMatchEvaluator:
    """Tests for exact match evaluation."""

    def test_exact_match_success(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is 2+2?",
            expected={"answer": "4"},
        )

        context = EvaluationContext(
            hero_id="h1",
            response="4",
            quest=quest,
        )

        evaluator = ExactMatchEvaluator()
        outcome = evaluator.evaluate(context)

        assert outcome.combat_result == CombatResult.HIT
        assert outcome.metrics["exact_match"] == 1.0

    def test_exact_match_failure(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is 2+2?",
            expected={"answer": "4"},
        )

        context = EvaluationContext(
            hero_id="h1",
            response="5",
            quest=quest,
        )

        evaluator = ExactMatchEvaluator()
        outcome = evaluator.evaluate(context)

        assert outcome.combat_result == CombatResult.MISS
        assert outcome.metrics["exact_match"] == 0.0

    def test_exact_match_case_insensitive(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What color?",
            expected="Blue",
        )

        context = EvaluationContext(
            hero_id="h1",
            response="blue",
            quest=quest,
        )

        evaluator = ExactMatchEvaluator()
        outcome = evaluator.evaluate(context, {"case_sensitive": False})

        assert outcome.combat_result == CombatResult.HIT


class TestContainsEvaluator:
    """Tests for contains evaluation."""

    def test_contains_success(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is the answer?",
            expected="42",
        )

        context = EvaluationContext(
            hero_id="h1",
            response="The answer is 42.",
            quest=quest,
        )

        evaluator = ContainsEvaluator()
        outcome = evaluator.evaluate(context)

        assert outcome.combat_result == CombatResult.HIT

    def test_contains_failure(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is the answer?",
            expected="42",
        )

        context = EvaluationContext(
            hero_id="h1",
            response="I don't know",
            quest=quest,
        )

        evaluator = ContainsEvaluator()
        outcome = evaluator.evaluate(context)

        assert outcome.combat_result == CombatResult.MISS


class TestQuestJudge:
    """Tests for quest judgment."""

    def test_judge_evaluate(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is 2+2?",
            expected={"answer": "4"},
        )

        judge = QuestJudge()
        result = judge.evaluate(
            hero_id="hero_123",
            response="4",
            quest=quest,
        )

        assert result.combat_result == CombatResult.HIT
        assert result.hero_id == "hero_123"
        assert result.quest_id == "q1"
        assert "logic" in result.xp_awarded

    def test_judge_xp_calculation(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic", "math"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is 2+2?",
            expected="4",
        )

        judge = QuestJudge()
        result = judge.evaluate(
            hero_id="hero_123",
            response="4",
            quest=quest,
        )

        # Both skills should get XP
        assert "logic" in result.xp_awarded
        assert "math" in result.xp_awarded
        assert result.total_xp > 0

    def test_judge_with_template_base_xp(self):
        template = QuestTemplate(
            id="test",
            name="Test Quest",
            description="A test",
            skills=["logic"],
            regions=["realm"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="static",
            evaluator_id="exact_match",
            base_xp={"logic": 25},
        )

        quest = QuestInstance.create(
            template=template,
            prompt="Test?",
            expected="answer",
        )

        judge = QuestJudge()
        result = judge.evaluate(
            hero_id="h1",
            response="answer",
            quest=quest,
            template=template,
        )

        # Should use template's base_xp
        assert result.xp_awarded["logic"] >= 25


class TestGlobalJudge:
    """Tests for global judge functions."""

    def test_get_judge_auto_init(self):
        judge = get_judge()
        assert judge is not None
        assert "exact_match" in judge.list_evaluators()

    def test_evaluate_quest_convenience(self):
        quest = QuestInstance(
            id="q1",
            template_id="t1",
            skills=["logic"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="What is 2+2?",
            expected="4",
        )

        result = evaluate_quest(
            hero_id="h1",
            response="4",
            quest=quest,
        )

        assert result.combat_result == CombatResult.HIT


# =============================================================================
# Integration Tests
# =============================================================================

class TestQuestsIntegration:
    """Integration tests for the quest module."""

    def test_full_workflow(self, temp_config_dir):
        """Test complete workflow: load, create, evaluate."""
        # Initialize
        init_quest_registry(temp_config_dir)

        # Get template
        template = get_quest("test_logic_quest")
        assert template.name == "Test Logic Quest"

        # Generate instance
        instance = create_quest(template)
        assert instance.prompt == "What is 2 + 2?"

        # Evaluate response
        result = evaluate_quest(
            hero_id="hero_123",
            response="4",
            quest=instance,
            template=template,
        )

        assert result.success is True
        assert result.combat_result == CombatResult.HIT
        assert result.total_xp > 0

    def test_wrong_answer_flow(self, temp_config_dir):
        """Test flow with incorrect answer."""
        init_quest_registry(temp_config_dir)

        template = get_quest("test_logic_quest")
        instance = create_quest(template)

        result = evaluate_quest(
            hero_id="hero_123",
            response="5",  # Wrong answer
            quest=instance,
            template=template,
        )

        assert result.success is False
        assert result.combat_result == CombatResult.MISS


# =============================================================================
# Test with Real Configs (optional)
# =============================================================================

class TestRealQuestConfigs:
    """Tests with real config files (skipped if not present)."""

    def test_load_real_quests(self):
        """Test loading real quest configs."""
        real_config_dir = project_root / "configs"

        if not (real_config_dir / "quests").exists():
            pytest.skip("No real quest configs found")

        templates = discover_quest_templates(real_config_dir)

        if not templates:
            pytest.skip("No quest configs in configs/quests/")

        # Load first quest
        quest_id, category = templates[0]
        quest = load_quest_template(quest_id, category, real_config_dir)
        assert quest.id is not None
        assert quest.skills
