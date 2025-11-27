"""Tests for progression system - XP, effects, and hero management."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import json
from datetime import datetime

from guild.types import Severity
from guild.quests.types import QuestResult, CombatResult, QuestInstance, QuestDifficulty
from guild.progression.types import (
    EffectType,
    StatusEffect,
    EffectDefinition,
    HeroIdentity,
    HeroState,
)
from guild.progression.xp import (
    LevelConfig,
    XPModifiers,
    XPCalculator,
    get_calculator,
    reset_calculator,
)
from guild.progression.effects import (
    EffectRegistry,
    EffectEvaluator,
    EffectManager,
    get_effect_manager,
    reset_effect_manager,
)
from guild.progression.hero_manager import (
    HeroManager,
    init_hero_manager,
    get_hero_manager,
    reset_hero_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with effects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        prog_dir = config_dir / "progression"
        prog_dir.mkdir(parents=True)

        effects_yaml = """
effects:
  test_buff:
    id: test_buff
    name: Test Buff
    description: A test buff effect
    type: buff
    severity: low
    default_duration_steps: 100
    effects:
      xp_multiplier: 1.5

  test_debuff:
    id: test_debuff
    name: Test Debuff
    description: A test debuff effect
    type: debuff
    severity: medium
    default_duration_steps: 50
    effects:
      xp_multiplier: 0.75

  permanent_debuff:
    id: permanent_debuff
    name: Permanent Debuff
    description: A permanent debuff
    type: debuff
    severity: high
    default_duration_steps: null
    effects:
      training_blocked: true

rules:
  - id: test_momentum_rule
    effect_id: test_buff
    trigger_type: consecutive_successes
    trigger_config:
      count: 3
      result: hit_or_crit
    cooldown_steps: 50

  - id: test_confusion_rule
    effect_id: test_debuff
    trigger_type: consecutive_failures
    trigger_config:
      count: 3
      result: miss
    cooldown_steps: 50
"""
        (prog_dir / "effects.yaml").write_text(effects_yaml)
        yield config_dir


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_calculator()
    reset_effect_manager()
    reset_hero_manager()
    yield
    reset_calculator()
    reset_effect_manager()
    reset_hero_manager()


@pytest.fixture
def sample_hero_identity():
    """Create a sample hero identity."""
    return HeroIdentity(
        id="test_hero",
        name="Test Hero",
        architecture="qwen",
        generation="3",
        size="0.6B",
        variant="base",
    )


@pytest.fixture
def sample_quest_result():
    """Create a sample quest result."""
    return QuestResult(
        quest_id="q1",
        hero_id="test_hero",
        response="answer",
        combat_result=CombatResult.HIT,
        xp_awarded={"logic": 10, "math": 5},
    )


# =============================================================================
# XP System Tests
# =============================================================================

class TestLevelConfig:
    """Tests for level configuration."""

    def test_xp_for_level(self):
        config = LevelConfig(base_xp=100, growth_rate=1.5)

        assert config.xp_for_level(1) == 0  # Level 1 requires 0 XP
        assert config.xp_for_level(2) == 100
        assert config.xp_for_level(3) == 150  # 100 * 1.5

    def test_total_xp_for_level(self):
        config = LevelConfig(base_xp=100, growth_rate=1.5)

        assert config.total_xp_for_level(1) == 0
        assert config.total_xp_for_level(2) == 100
        assert config.total_xp_for_level(3) == 250  # 0 + 100 + 150

    def test_level_from_xp(self):
        config = LevelConfig(base_xp=100, growth_rate=1.5)

        assert config.level_from_xp(0) == 1
        assert config.level_from_xp(50) == 1
        assert config.level_from_xp(100) == 2
        assert config.level_from_xp(250) == 3


class TestXPModifiers:
    """Tests for XP modifiers."""

    def test_total_multiplier(self):
        mods = XPModifiers(
            skill_multiplier=1.5,
            difficulty_multiplier=1.2,
            effect_multiplier=0.8,
            streak_multiplier=1.1,
        )

        expected = 1.5 * 1.2 * 0.8 * 1.1
        assert mods.total_multiplier == pytest.approx(expected)

    def test_default_multipliers(self):
        mods = XPModifiers()
        assert mods.total_multiplier == 1.0


class TestXPCalculator:
    """Tests for XP calculation."""

    def test_calculate_base_xp(self):
        calc = XPCalculator()

        assert calc.calculate_base_xp(CombatResult.CRITICAL_HIT) == 15
        assert calc.calculate_base_xp(CombatResult.HIT) == 10
        assert calc.calculate_base_xp(CombatResult.MISS) == 2

    def test_calculate_streak_mult(self):
        calc = XPCalculator()

        assert calc.calculate_streak_mult(0) == 1.0
        assert calc.calculate_streak_mult(1) == 1.05  # 5% bonus
        assert calc.calculate_streak_mult(5) == 1.25  # 25% bonus
        assert calc.calculate_streak_mult(20) == 1.5  # Capped at 50%

    def test_calculate_xp(self, sample_quest_result):
        calc = XPCalculator()

        xp = calc.calculate_xp(sample_quest_result)

        assert "logic" in xp
        assert "math" in xp
        assert xp["logic"] == 10
        assert xp["math"] == 5

    def test_calculate_xp_with_multipliers(self, sample_quest_result):
        calc = XPCalculator()

        xp = calc.calculate_xp(
            sample_quest_result,
            skill_multiplier=2.0,
            effect_multipliers=[1.5],
            streak=5,
        )

        # Base 10 * 2.0 * 1.5 * 1.25 = 37.5 -> 37
        assert xp["logic"] == 37

    def test_check_level_up(self):
        calc = XPCalculator()

        level, levels_gained = calc.check_level_up(1, 0, 250)

        assert level == 3
        assert levels_gained == [2, 3]

    def test_calculate_level_progress(self):
        calc = XPCalculator()

        # At level 2, need 150 XP to reach level 3
        # With 175 XP total (100 for L2, 75 toward L3)
        progress = calc.calculate_level_progress(2, 175)

        assert progress == pytest.approx(0.5, rel=0.1)


class TestGlobalXPCalculator:
    """Tests for global XP calculator."""

    def test_get_calculator(self):
        calc = get_calculator()
        assert calc is not None

        calc2 = get_calculator()
        assert calc is calc2  # Same instance


# =============================================================================
# Effects System Tests
# =============================================================================

class TestStatusEffect:
    """Tests for status effect types."""

    def test_effect_not_expired(self):
        effect = StatusEffect(
            id="test",
            name="Test",
            description="Test effect",
            type=EffectType.BUFF,
            severity=Severity.LOW,
            applied_at_step=100,
            duration_steps=50,
        )

        assert effect.is_expired(125) is False
        assert effect.is_expired(150) is True

    def test_permanent_effect(self):
        effect = StatusEffect(
            id="test",
            name="Test",
            description="Test effect",
            type=EffectType.DEBUFF,
            severity=Severity.HIGH,
            applied_at_step=100,
            duration_steps=None,
        )

        assert effect.is_expired(1000000) is False

    def test_effect_serialization(self):
        effect = StatusEffect(
            id="test",
            name="Test",
            description="Test effect",
            type=EffectType.BUFF,
            severity=Severity.LOW,
            applied_at_step=100,
            effects={"xp_multiplier": 1.5},
        )

        data = effect.to_dict()
        restored = StatusEffect.from_dict(data)

        assert restored.id == effect.id
        assert restored.type == effect.type
        assert restored.effects["xp_multiplier"] == 1.5


class TestEffectDefinition:
    """Tests for effect definitions."""

    def test_create_instance(self):
        definition = EffectDefinition(
            id="test",
            name="Test",
            description="Test effect",
            type=EffectType.BUFF,
            severity=Severity.LOW,
            default_duration_steps=100,
            effects={"xp_multiplier": 1.5},
        )

        instance = definition.create_instance(step=500)

        assert instance.id == "test"
        assert instance.applied_at_step == 500
        assert instance.duration_steps == 100
        assert instance.effects["xp_multiplier"] == 1.5


class TestEffectRegistry:
    """Tests for effect registry."""

    def test_load_effects(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)

        effect = registry.get_effect("test_buff")
        assert effect is not None
        assert effect.name == "Test Buff"
        assert effect.type == EffectType.BUFF

    def test_get_rules(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)

        rules = registry.get_rules()
        assert len(rules) == 2

    def test_create_effect(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)

        effect = registry.create_effect("test_buff", step=100)
        assert effect is not None
        assert effect.id == "test_buff"
        assert effect.applied_at_step == 100


class TestEffectEvaluator:
    """Tests for effect rule evaluation."""

    def test_evaluate_consecutive_success(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)
        evaluator = EffectEvaluator(registry)

        hero = HeroState(
            hero_id="test",
            identity=HeroIdentity(
                id="test",
                name="Test",
                architecture="qwen",
                generation="3",
                size="0.6B",
                variant="base",
            ),
            current_step=100,
        )

        # 3 consecutive successes should trigger buff
        recent = [CombatResult.HIT, CombatResult.HIT, CombatResult.HIT]
        effects = evaluator.evaluate_rules(hero, recent_results=recent)

        assert len(effects) == 1
        assert effects[0].id == "test_buff"

    def test_evaluate_consecutive_failure(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)
        evaluator = EffectEvaluator(registry)

        hero = HeroState(
            hero_id="test",
            identity=HeroIdentity(
                id="test",
                name="Test",
                architecture="qwen",
                generation="3",
                size="0.6B",
                variant="base",
            ),
            current_step=100,
        )

        # 3 consecutive misses should trigger debuff
        recent = [CombatResult.MISS, CombatResult.MISS, CombatResult.MISS]
        effects = evaluator.evaluate_rules(hero, recent_results=recent)

        assert len(effects) == 1
        assert effects[0].id == "test_debuff"

    def test_get_xp_multiplier(self, temp_config_dir):
        registry = EffectRegistry(temp_config_dir)
        evaluator = EffectEvaluator(registry)

        hero = HeroState(
            hero_id="test",
            identity=HeroIdentity(
                id="test",
                name="Test",
                architecture="qwen",
                generation="3",
                size="0.6B",
                variant="base",
            ),
        )

        # No effects
        assert evaluator.get_xp_multiplier(hero) == 1.0

        # Add buff effect
        effect = registry.create_effect("test_buff", step=100)
        hero.add_effect(effect)

        assert evaluator.get_xp_multiplier(hero) == 1.5


class TestEffectManager:
    """Tests for effect management."""

    def test_apply_effect(self, temp_config_dir):
        manager = EffectManager(temp_config_dir)

        hero = HeroState(
            hero_id="test",
            identity=HeroIdentity(
                id="test",
                name="Test",
                architecture="qwen",
                generation="3",
                size="0.6B",
                variant="base",
            ),
            current_step=100,
        )

        effect = manager.apply_effect(hero, "test_buff")

        assert effect is not None
        assert len(hero.active_effects) == 1
        assert hero.active_effects[0].id == "test_buff"

    def test_has_blocking_effect(self, temp_config_dir):
        manager = EffectManager(temp_config_dir)

        hero = HeroState(
            hero_id="test",
            identity=HeroIdentity(
                id="test",
                name="Test",
                architecture="qwen",
                generation="3",
                size="0.6B",
                variant="base",
            ),
            current_step=100,
        )

        assert manager.has_blocking_effect(hero) is False

        manager.apply_effect(hero, "permanent_debuff")

        assert manager.has_blocking_effect(hero) is True


# =============================================================================
# Hero Manager Tests
# =============================================================================

class TestHeroState:
    """Tests for hero state."""

    def test_health_property(self, sample_hero_identity):
        hero = HeroState(
            hero_id="test",
            identity=sample_hero_identity,
        )

        assert hero.health == "healthy"

    def test_get_skill(self, sample_hero_identity):
        hero = HeroState(
            hero_id="test",
            identity=sample_hero_identity,
        )

        skill = hero.get_skill("logic")

        assert skill.skill_id == "logic"
        assert skill.level == 1

        # Second call returns same instance
        skill2 = hero.get_skill("logic")
        assert skill is skill2

    def test_add_remove_effect(self, sample_hero_identity):
        hero = HeroState(
            hero_id="test",
            identity=sample_hero_identity,
        )

        effect = StatusEffect(
            id="test",
            name="Test",
            description="Test",
            type=EffectType.BUFF,
            severity=Severity.LOW,
            applied_at_step=100,
        )

        hero.add_effect(effect)
        assert len(hero.active_effects) == 1

        hero.remove_effect("test")
        assert len(hero.active_effects) == 0

    def test_serialization(self, sample_hero_identity):
        hero = HeroState(
            hero_id="test",
            identity=sample_hero_identity,
            total_xp=500,
            total_quests=10,
        )
        hero.get_skill("logic").xp_total = 100

        data = hero.to_dict()
        restored = HeroState.from_dict(data)

        assert restored.hero_id == "test"
        assert restored.total_xp == 500
        assert restored.skills["logic"].xp_total == 100


class TestHeroManager:
    """Tests for hero management."""

    def test_create_hero(self, temp_state_dir, sample_hero_identity):
        manager = HeroManager(temp_state_dir)

        hero = manager.create_hero("test_hero", sample_hero_identity)

        assert hero.hero_id == "test_hero"
        assert hero.identity.name == "Test Hero"

    def test_persistence(self, temp_state_dir, sample_hero_identity):
        # Create and save
        manager1 = HeroManager(temp_state_dir)
        manager1.create_hero("test_hero", sample_hero_identity)

        # Load in new manager
        manager2 = HeroManager(temp_state_dir)
        hero = manager2.get_hero()

        assert hero is not None
        assert hero.hero_id == "test_hero"

    def test_record_result(
        self, temp_state_dir, temp_config_dir, sample_hero_identity, sample_quest_result
    ):
        from guild.progression.effects import init_effect_manager
        init_effect_manager(temp_config_dir)

        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        summary = manager.record_result(sample_quest_result)

        assert "xp_gained" in summary
        assert "logic" in summary["xp_gained"]

        hero = manager.get_hero()
        assert hero.total_quests == 1
        assert hero.total_xp > 0

    def test_add_xp(self, temp_state_dir, sample_hero_identity):
        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        result = manager.add_xp("logic", 500, source="test")

        assert result["xp_added"] == 500
        assert result["new_total"] == 500

        hero = manager.get_hero()
        skill = hero.get_skill("logic")
        assert skill.xp_total == 500

    def test_get_status(self, temp_state_dir, sample_hero_identity):
        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        status = manager.get_status()

        assert status["loaded"] is True
        assert status["hero_id"] == "test_hero"
        assert status["health"] == "healthy"

    def test_win_streak(
        self, temp_state_dir, temp_config_dir, sample_hero_identity
    ):
        from guild.progression.effects import init_effect_manager
        init_effect_manager(temp_config_dir)

        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        # Record wins
        for i in range(5):
            result = QuestResult(
                quest_id=f"q{i}",
                hero_id="test_hero",
                response="answer",
                combat_result=CombatResult.HIT,
                xp_awarded={"logic": 10},
            )
            summary = manager.record_result(result)

        assert summary["streak"] == 5

        # Record a miss
        miss_result = QuestResult(
            quest_id="q_miss",
            hero_id="test_hero",
            response="wrong",
            combat_result=CombatResult.MISS,
            xp_awarded={"logic": 2},
        )
        summary = manager.record_result(miss_result)

        assert summary["streak"] == 0


class TestGlobalHeroManager:
    """Tests for global hero manager."""

    def test_init_and_get(self, temp_state_dir):
        init_hero_manager(temp_state_dir)

        manager = get_hero_manager()
        assert manager is not None

    def test_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_hero_manager()


# =============================================================================
# Integration Tests
# =============================================================================

class TestProgressionIntegration:
    """Integration tests for the progression system."""

    def test_full_workflow(
        self, temp_state_dir, temp_config_dir, sample_hero_identity
    ):
        """Test complete progression workflow."""
        from guild.progression.effects import init_effect_manager
        init_effect_manager(temp_config_dir)

        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        # Record multiple results
        for i in range(10):
            result = QuestResult(
                quest_id=f"q{i}",
                hero_id="test_hero",
                response="answer",
                combat_result=CombatResult.HIT,
                xp_awarded={"logic": 15},
            )
            manager.record_result(result)

        # Check progress
        hero = manager.get_hero()
        assert hero.total_quests == 10
        assert hero.total_xp > 0

        skill_progress = manager.get_skill_progress("logic")
        assert skill_progress["level"] >= 1
        assert skill_progress["xp_total"] > 0

    def test_effect_triggers_on_streak(
        self, temp_state_dir, temp_config_dir, sample_hero_identity
    ):
        """Test that effects trigger on win streaks."""
        from guild.progression.effects import init_effect_manager
        init_effect_manager(temp_config_dir)

        manager = HeroManager(temp_state_dir)
        manager.create_hero("test_hero", sample_hero_identity)

        # Record 3 consecutive successes
        effects_triggered = []
        for i in range(3):
            result = QuestResult(
                quest_id=f"q{i}",
                hero_id="test_hero",
                response="answer",
                combat_result=CombatResult.HIT,
                xp_awarded={"logic": 10},
            )
            summary = manager.record_result(result)
            effects_triggered.extend(summary.get("effects_triggered", []))

        # Should have triggered the buff
        assert "test_buff" in effects_triggered


# =============================================================================
# Test with Real Configs (optional)
# =============================================================================

class TestRealProgressionConfigs:
    """Tests with real config files (skipped if not present)."""

    def test_load_real_effects(self):
        """Test loading real effects configs."""
        real_config_dir = project_root / "configs"

        if not (real_config_dir / "progression" / "effects.yaml").exists():
            pytest.skip("No real effects config found")

        registry = EffectRegistry(real_config_dir)
        effects = registry.get_all_effects()

        assert len(effects) > 0
