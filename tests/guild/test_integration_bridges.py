"""
Tests for the integration layer bridges.

Tests cover:
- HeroSkillState and HeroStateManager
- SkillEngineBridge
- SchedulerBridge
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import test targets
from guild.integration.hero_state import (
    HeroSkillState,
    HeroStateManager,
)
from guild.integration.scheduler_bridge import (
    QuestType,
    QuestRecommendation,
    SkillMetrics,
    SchedulerBridge,
)
from guild.integration.skill_engine_bridge import (
    TrainingRequest,
    TrainingBatch,
    EvalSyncResult,
)


class TestHeroSkillState:
    """Tests for HeroSkillState dataclass."""

    def test_create_state(self):
        """HeroSkillState can be created with required fields."""
        state = HeroSkillState(
            hero_id="dio",
            skill_id="bin",
        )
        assert state.hero_id == "dio"
        assert state.skill_id == "bin"
        assert state.level == 1
        assert state.xp_total == 0.0

    def test_accuracy_property(self):
        """accuracy property calculates rolling average."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        state.recent_results = [True, True, False, True]
        assert state.accuracy == 0.75

    def test_empty_accuracy(self):
        """Empty results returns 0.0."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        assert state.accuracy == 0.0

    def test_record_result(self):
        """record_result adds to recent_results."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        state.record_result(True)
        state.record_result(False)
        assert state.recent_results == [True, False]

    def test_record_result_window_size(self):
        """record_result trims to window_size."""
        state = HeroSkillState(hero_id="dio", skill_id="bin", window_size=3)
        for _ in range(5):
            state.record_result(True)
        assert len(state.recent_results) == 3

    def test_record_level_up(self):
        """record_level_up increments level."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        state.xp_total = 1000.0
        state.record_level_up()
        assert state.level == 2
        assert state.xp_marks[2] == 1000.0
        assert state.eligible_for_trial is False

    def test_record_eval(self):
        """record_eval tracks eval results."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        state.record_eval(
            accuracy=0.85,
            per_primitive={"add": 0.9, "sub": 0.8},
            samples=100,
        )
        assert state.total_evals == 1
        assert state.total_samples_seen == 100
        assert state.last_eval_accuracy == 0.85
        assert state.primitive_accuracy["add"] == 0.9

    def test_record_eval_ema(self):
        """record_eval uses EMA for primitive accuracy."""
        state = HeroSkillState(hero_id="dio", skill_id="bin")
        # First eval
        state.record_eval(accuracy=0.8, per_primitive={"add": 1.0})
        # Second eval - should blend with EMA (0.3 * new + 0.7 * old)
        state.record_eval(accuracy=0.9, per_primitive={"add": 0.5})
        # Expected: 0.3 * 0.5 + 0.7 * 1.0 = 0.85
        assert abs(state.primitive_accuracy["add"] - 0.85) < 0.001

    def test_to_dict(self):
        """to_dict serializes state."""
        state = HeroSkillState(hero_id="dio", skill_id="bin", level=5)
        d = state.to_dict()
        assert d["hero_id"] == "dio"
        assert d["skill_id"] == "bin"
        assert d["level"] == 5

    def test_from_dict(self):
        """from_dict deserializes state."""
        data = {
            "hero_id": "dio",
            "skill_id": "bin",
            "level": 5,
            "xp_total": 500.0,
            "primitive_accuracy": {"add": 0.9},
        }
        state = HeroSkillState.from_dict(data)
        assert state.hero_id == "dio"
        assert state.level == 5
        assert state.primitive_accuracy["add"] == 0.9


class TestHeroStateManager:
    """Tests for HeroStateManager."""

    @pytest.fixture
    def temp_manager(self, tmp_path):
        """Create manager with temp directory."""
        return HeroStateManager(base_dir=tmp_path)

    def test_get_creates_default(self, temp_manager):
        """get creates default state if not exists."""
        state = temp_manager.get("dio", "bin")
        assert state.hero_id == "dio"
        assert state.skill_id == "bin"
        assert state.level == 1

    def test_set_and_get(self, temp_manager):
        """set persists state."""
        state = HeroSkillState(hero_id="dio", skill_id="bin", level=10)
        temp_manager.set("dio", "bin", state)

        # Clear cache to force reload
        temp_manager.clear_cache()

        loaded = temp_manager.get("dio", "bin")
        assert loaded.level == 10

    def test_update(self, temp_manager):
        """update modifies specific fields."""
        temp_manager.get("dio", "bin")  # Create default
        updated = temp_manager.update("dio", "bin", level=5, xp_total=1000.0)
        assert updated.level == 5
        assert updated.xp_total == 1000.0

    def test_list_heroes(self, temp_manager):
        """list_heroes returns heroes with state files."""
        temp_manager.get("dio", "bin")
        temp_manager.set("dio", "bin", HeroSkillState(hero_id="dio", skill_id="bin"))
        temp_manager.get("atlas", "sy")
        temp_manager.set("atlas", "sy", HeroSkillState(hero_id="atlas", skill_id="sy"))

        heroes = temp_manager.list_heroes()
        assert "dio" in heroes
        assert "atlas" in heroes

    def test_list_skills(self, temp_manager):
        """list_skills returns skills for a hero."""
        state1 = HeroSkillState(hero_id="dio", skill_id="bin")
        state2 = HeroSkillState(hero_id="dio", skill_id="sy")
        temp_manager.set("dio", "bin", state1)
        temp_manager.set("dio", "sy", state2)

        skills = temp_manager.list_skills("dio")
        assert "bin" in skills
        assert "sy" in skills

    def test_get_all(self, temp_manager):
        """get_all returns all skills for a hero."""
        temp_manager.set("dio", "bin", HeroSkillState(hero_id="dio", skill_id="bin", level=3))
        temp_manager.set("dio", "sy", HeroSkillState(hero_id="dio", skill_id="sy", level=5))

        all_states = temp_manager.get_all("dio")
        assert "bin" in all_states
        assert "sy" in all_states
        assert all_states["bin"].level == 3


class TestQuestTypes:
    """Tests for QuestType enum."""

    def test_quest_types(self):
        """QuestType has expected values."""
        assert QuestType.TRAINING.value == "training"
        assert QuestType.EVAL.value == "eval"
        assert QuestType.SPARRING.value == "sparring"
        assert QuestType.MIXED.value == "mixed"


class TestQuestRecommendation:
    """Tests for QuestRecommendation dataclass."""

    def test_create_recommendation(self):
        """QuestRecommendation can be created."""
        rec = QuestRecommendation(
            skill_id="bin",
            level=3,
            quest_type=QuestType.TRAINING,
            priority=0.8,
            reason="Low accuracy",
        )
        assert rec.skill_id == "bin"
        assert rec.level == 3
        assert rec.priority == 0.8

    def test_to_dict(self):
        """to_dict serializes recommendation."""
        rec = QuestRecommendation(
            skill_id="bin",
            level=3,
            quest_type=QuestType.EVAL,
            priority=0.9,
            reason="Time since eval",
            target_primitives=["add", "sub"],
        )
        d = rec.to_dict()
        assert d["skill_id"] == "bin"
        assert d["quest_type"] == "eval"
        assert d["target_primitives"] == ["add", "sub"]


class TestSkillMetrics:
    """Tests for SkillMetrics dataclass."""

    def test_create_metrics(self):
        """SkillMetrics can be created."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.85,
            last_eval_time=datetime.now(),
            total_evals=5,
            weak_primitives=["sub"],
            time_since_eval=timedelta(hours=2),
        )
        assert metrics.skill_id == "bin"
        assert metrics.accuracy == 0.85
        assert metrics.weak_primitives == ["sub"]


class TestTrainingTypes:
    """Tests for training-related types."""

    def test_training_request(self):
        """TrainingRequest stores request parameters."""
        req = TrainingRequest(
            skill_id="bin",
            level=3,
            count=100,
            difficulty="hard",
        )
        assert req.skill_id == "bin"
        assert req.count == 100

    def test_training_batch(self):
        """TrainingBatch stores examples."""
        batch = TrainingBatch(
            skill_id="bin",
            level=3,
            examples=[
                {"prompt": "1+1", "answer": "2"},
                {"prompt": "2+2", "answer": "4"},
            ],
        )
        assert batch.count == 2

    def test_eval_sync_result(self):
        """EvalSyncResult tracks sync status."""
        result = EvalSyncResult(
            skill_id="bin",
            accuracy=0.9,
            level=3,
            level_changed=True,
            new_level=4,
            primitives_updated=["add", "sub"],
        )
        assert result.level_changed
        assert result.new_level == 4


class TestSchedulerBridgeLogic:
    """Tests for SchedulerBridge calculation logic."""

    def test_priority_base(self):
        """Base priority is 0.5."""
        # Create metrics with good accuracy, recent eval, no weak primitives
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,  # Above threshold
            last_eval_time=datetime.now() - timedelta(hours=1),
            total_evals=5,
            weak_primitives=[],
            time_since_eval=timedelta(hours=1),
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        priority = bridge._calculate_priority(metrics)
        assert 0.4 <= priority <= 0.6

    def test_priority_low_accuracy(self):
        """Low accuracy increases priority."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.50,  # Below 0.80 threshold
            last_eval_time=datetime.now(),
            total_evals=5,
            weak_primitives=[],
            time_since_eval=timedelta(hours=1),
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        priority = bridge._calculate_priority(metrics)
        assert priority > 0.6  # Should be elevated

    def test_priority_weak_primitives(self):
        """Weak primitives increase priority."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,
            last_eval_time=datetime.now(),
            total_evals=5,
            weak_primitives=["add", "sub", "mul"],  # 3 weak primitives
            time_since_eval=timedelta(hours=1),
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        priority = bridge._calculate_priority(metrics)
        assert priority > 0.5

    def test_decide_quest_type_no_eval(self):
        """No eval time → EVAL quest."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,
            last_eval_time=None,
            total_evals=0,
            weak_primitives=[],
            time_since_eval=None,
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        quest_type = bridge._decide_quest_type(metrics)
        assert quest_type == QuestType.EVAL

    def test_decide_quest_type_overdue_eval(self):
        """Overdue eval → EVAL quest."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,
            last_eval_time=datetime.now() - timedelta(hours=10),
            total_evals=5,
            weak_primitives=[],
            time_since_eval=timedelta(hours=10),  # > 4 hour threshold
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        quest_type = bridge._decide_quest_type(metrics)
        assert quest_type == QuestType.EVAL

    def test_decide_quest_type_weak_primitives(self):
        """Weak primitives + recent eval → SPARRING quest."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,
            last_eval_time=datetime.now() - timedelta(hours=1),
            total_evals=5,
            weak_primitives=["add", "sub"],
            time_since_eval=timedelta(hours=1),  # Recent
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        quest_type = bridge._decide_quest_type(metrics)
        assert quest_type == QuestType.SPARRING

    def test_decide_quest_type_default_training(self):
        """No issues → TRAINING quest."""
        metrics = SkillMetrics(
            skill_id="bin",
            level=3,
            accuracy=0.90,
            last_eval_time=datetime.now() - timedelta(hours=1),
            total_evals=5,
            weak_primitives=[],
            time_since_eval=timedelta(hours=1),
        )
        bridge = SchedulerBridge.__new__(SchedulerBridge)
        quest_type = bridge._decide_quest_type(metrics)
        assert quest_type == QuestType.TRAINING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
