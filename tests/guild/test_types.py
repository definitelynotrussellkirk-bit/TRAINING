"""Tests for guild type definitions."""

import pytest
import json
from datetime import datetime

from guild.types import Severity, Status, generate_id
from guild.skills.types import SkillConfig, SkillState, SkillCategory
from guild.quests.types import (
    QuestTemplate, QuestInstance, QuestResult,
    QuestDifficulty, CombatResult
)
from guild.facilities.types import Facility, FacilityType
from guild.progression.types import (
    StatusEffect, EffectType, HeroState, HeroIdentity
)
from guild.runs.types import RunConfig, RunState, RunType
from guild.incidents.types import Incident, IncidentCategory, IncidentStatus
from guild.combat.types import CombatConfig, CombatStance


class TestBasicTypes:
    def test_severity_enum(self):
        assert Severity.LOW.value == "low"
        assert Severity.CRITICAL.value == "critical"

    def test_status_enum(self):
        assert Status.PENDING.value == "pending"
        assert Status.COMPLETED.value == "completed"

    def test_generate_id(self):
        id1 = generate_id("test")
        id2 = generate_id("test")
        assert id1.startswith("test_")
        assert id1 != id2
        assert len(id1) == 13  # "test_" + 8 chars

    def test_generate_id_no_prefix(self):
        id1 = generate_id()
        assert len(id1) == 8
        assert "_" not in id1


class TestSkillTypes:
    def test_skill_config_creation(self):
        skill = SkillConfig(
            id="logic_weaving",
            name="Logic Weaving",
            description="Deductive reasoning",
            category=SkillCategory.REASONING,
            accuracy_thresholds={1: 0.6, 2: 0.65, 3: 0.7}
        )
        assert skill.id == "logic_weaving"
        assert skill.get_threshold(1) == 0.6
        assert skill.get_threshold(3) == 0.7

    def test_skill_config_default_threshold(self):
        skill = SkillConfig(
            id="test",
            name="Test",
            description="Test",
            category=SkillCategory.REASONING,
        )
        # Default threshold formula: 0.6 + (level-1) * 0.03
        assert skill.get_threshold(1) == pytest.approx(0.6)
        assert skill.get_threshold(5) == pytest.approx(0.72)

    def test_skill_state_accuracy(self):
        state = SkillState(skill_id="test")
        assert state.accuracy == 0.0

        state.record_result(True)
        state.record_result(True)
        state.record_result(False)
        assert state.accuracy == pytest.approx(2/3)

    def test_skill_state_window_limit(self):
        state = SkillState(skill_id="test", window_size=5)

        # Record more than window size
        for _ in range(10):
            state.record_result(True)

        assert len(state.recent_results) == 5

    def test_skill_state_level_up(self):
        state = SkillState(skill_id="test", xp_total=1000)
        state.record_level_up()

        assert state.level == 2
        assert state.xp_marks[2] == 1000
        assert state.eligible_for_trial is False

    def test_skill_state_serialization(self):
        state = SkillState(skill_id="test", level=3, xp_total=1500.0)
        state.record_result(True)
        state.record_result(False)

        data = state.to_dict()
        json_str = json.dumps(data)  # Should not raise

        loaded = SkillState.from_dict(json.loads(json_str))
        assert loaded.skill_id == "test"
        assert loaded.level == 3
        assert loaded.xp_total == 1500.0
        assert len(loaded.recent_results) == 2


class TestQuestTypes:
    def test_quest_difficulty_from_level(self):
        assert QuestDifficulty.from_level(1) == QuestDifficulty.BRONZE
        assert QuestDifficulty.from_level(3) == QuestDifficulty.SILVER
        assert QuestDifficulty.from_level(5) == QuestDifficulty.GOLD
        assert QuestDifficulty.from_level(7) == QuestDifficulty.PLATINUM
        assert QuestDifficulty.from_level(10) == QuestDifficulty.DRAGON

    def test_quest_instance_creation(self):
        template = QuestTemplate(
            id="syllo_basic",
            name="Basic SYLLO",
            description="Easy puzzle",
            skills=["logic_weaving"],
            regions=["novice_valley"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="syllo_gen",
            evaluator_id="syllo_eval"
        )

        instance = QuestInstance.create(
            template=template,
            prompt="Solve this puzzle..."
        )

        assert instance.template_id == "syllo_basic"
        assert instance.skills == ["logic_weaving"]
        assert instance.difficulty == QuestDifficulty.BRONZE

    def test_quest_result_success(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"logic_weaving": 10}
        )
        assert result.success is True
        assert result.total_xp == 10

    def test_quest_result_failure(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="wrong",
            combat_result=CombatResult.MISS,
            xp_awarded={"logic_weaving": 2}
        )
        assert result.success is False

    def test_quest_instance_serialization(self):
        template = QuestTemplate(
            id="test",
            name="Test",
            description="Test",
            skills=["logic"],
            regions=["test"],
            difficulty=QuestDifficulty.SILVER,
            difficulty_level=3,
            generator_id="gen",
            evaluator_id="eval"
        )

        instance = QuestInstance.create(template, prompt="Test prompt")
        data = instance.to_dict()
        loaded = QuestInstance.from_dict(data)

        assert loaded.id == instance.id
        assert loaded.template_id == "test"
        assert loaded.difficulty == QuestDifficulty.SILVER
        assert loaded.difficulty_level == 3


class TestFacilityTypes:
    def test_facility_get_path(self):
        facility = Facility(
            id="arena",
            name="Arena",
            type=FacilityType.BATTLEFIELD,
            base_path="/tmp/test",
            paths={"checkpoints": "models/"}
        )

        assert facility.get_path("checkpoints") == "/tmp/test/models/"
        assert facility.get_path("checkpoints", "step-1000") == "/tmp/test/models/step-1000"
        assert facility.get_path("other") == "/tmp/test/other"

    def test_facility_serialization(self):
        facility = Facility(
            id="test",
            name="Test",
            type=FacilityType.HUB,
            base_path="/test",
            paths={"logs": "logs/"},
        )

        data = facility.to_dict()
        loaded = Facility.from_dict(data)

        assert loaded.id == "test"
        assert loaded.type == FacilityType.HUB
        assert loaded.paths["logs"] == "logs/"


class TestProgressionTypes:
    def test_status_effect_expiry(self):
        effect = StatusEffect(
            id="confusion",
            name="Confusion",
            description="Confused",
            type=EffectType.DEBUFF,
            severity=Severity.MEDIUM,
            applied_at_step=100,
            duration_steps=50
        )

        assert effect.is_expired(100) is False
        assert effect.is_expired(149) is False
        assert effect.is_expired(150) is True

    def test_status_effect_permanent(self):
        effect = StatusEffect(
            id="permanent",
            name="Permanent",
            description="No duration",
            type=EffectType.DEBUFF,
            severity=Severity.LOW,
            applied_at_step=0,
            duration_steps=None
        )

        assert effect.is_expired(1000000) is False

    def test_hero_identity(self):
        identity = HeroIdentity(
            id="hero1",
            name="Test Hero",
            architecture="qwen",
            generation="3",
            size="0.6B",
            variant="base"
        )

        data = identity.to_dict()
        loaded = HeroIdentity.from_dict(data)

        assert loaded.id == "hero1"
        assert loaded.architecture == "qwen"

    def test_hero_state_health_healthy(self):
        identity = HeroIdentity(
            id="hero1", name="Test",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )
        state = HeroState(hero_id="hero1", identity=identity)
        assert state.health == "healthy"

    def test_hero_state_health_wounded(self):
        identity = HeroIdentity(
            id="hero1", name="Test",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )
        state = HeroState(hero_id="hero1", identity=identity)

        effect = StatusEffect(
            id="nan_dragon",
            name="NaN Dragon",
            description="Training collapsed",
            type=EffectType.DEBUFF,
            severity=Severity.CRITICAL,
            applied_at_step=0
        )
        state.add_effect(effect)
        assert state.health == "wounded"

    def test_hero_state_get_skill(self):
        identity = HeroIdentity(
            id="hero1", name="Test",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )
        state = HeroState(hero_id="hero1", identity=identity)

        # Get skill creates it if not exists
        skill = state.get_skill("logic")
        assert skill.skill_id == "logic"
        assert skill.level == 1

        # Get same skill returns same object
        skill2 = state.get_skill("logic")
        assert skill is skill2

    def test_hero_state_serialization(self):
        identity = HeroIdentity(
            id="hero1", name="Test",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )
        state = HeroState(hero_id="hero1", identity=identity)
        state.get_skill("logic").record_result(True)
        state.total_quests = 100

        data = state.to_dict()
        json_str = json.dumps(data)

        loaded = HeroState.from_dict(json.loads(json_str))
        assert loaded.hero_id == "hero1"
        assert loaded.total_quests == 100
        assert "logic" in loaded.skills


class TestRunTypes:
    def test_run_config_creation(self):
        config = RunConfig(
            id="run1",
            type=RunType.TRAINING,
            name="Test Run",
            max_steps=1000
        )
        assert config.type == RunType.TRAINING
        assert config.max_steps == 1000

    def test_run_state_success_rate(self):
        config = RunConfig(id="r1", type=RunType.TRAINING)
        state = RunState(run_id="r1", config=config)

        state.quests_completed = 100
        state.quests_succeeded = 75
        assert state.success_rate == 0.75

    def test_run_state_success_rate_zero(self):
        config = RunConfig(id="r1", type=RunType.TRAINING)
        state = RunState(run_id="r1", config=config)
        assert state.success_rate == 0.0

    def test_run_state_serialization(self):
        config = RunConfig(id="r1", type=RunType.EVALUATION, name="Eval")
        state = RunState(run_id="r1", config=config)
        state.quests_completed = 50

        data = state.to_dict()
        loaded = RunState.from_dict(data)

        assert loaded.run_id == "r1"
        assert loaded.config.type == RunType.EVALUATION
        assert loaded.quests_completed == 50


class TestIncidentTypes:
    def test_incident_creation(self):
        incident = Incident(
            id="inc1",
            category=IncidentCategory.TRAINING,
            severity=Severity.CRITICAL,
            title="NaN Loss Detected",
            description="Loss became NaN at step 1000",
            detected_at_step=1000
        )
        assert incident.category == IncidentCategory.TRAINING
        assert incident.severity == Severity.CRITICAL
        assert incident.status == IncidentStatus.OPEN

    def test_incident_serialization(self):
        incident = Incident(
            id="inc1",
            category=IncidentCategory.DATA,
            severity=Severity.MEDIUM,
            title="Data Issue",
            description="Found corrupt data",
            detected_at_step=500
        )

        data = incident.to_dict()
        loaded = Incident.from_dict(data)

        assert loaded.id == "inc1"
        assert loaded.category == IncidentCategory.DATA


class TestCombatTypes:
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

    def test_combat_stance_enum(self):
        assert CombatStance.THOUGHTFUL.value == "thoughtful"
        assert CombatStance.QUICK_DRAW.value == "quick_draw"
        assert CombatStance.ALTERNATING.value == "alternating"
