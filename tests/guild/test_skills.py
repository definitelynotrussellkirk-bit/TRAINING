"""Tests for skill loading, registry, and state management."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import json
import os

from guild.skills.types import (
    SkillCategory,
    MetricDefinition,
    SkillConfig,
    SkillState,
)
from guild.skills.loader import (
    load_skill_config,
    discover_skills,
    load_all_skills,
    SkillLoader,
    _dict_to_skill_config,
)
from guild.skills.registry import (
    SkillRegistry,
    init_registry,
    get_registry,
    reset_registry,
    get_skill,
    list_skills,
)
from guild.skills.state_manager import (
    AccuracyRecord,
    ProgressionRecord,
    SkillStateManager,
    init_state_manager,
    get_state_manager,
    reset_state_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir(parents=True)

        # Create test skill configs
        logic_skill = """
id: test_logic
name: Test Logic Skill
description: A test skill for logic
category: reasoning

tags:
  - test
  - reasoning

metrics:
  - accuracy
  - precision
primary_metric: accuracy

accuracy_thresholds:
  1: 0.60
  2: 0.70
  3: 0.80

xp_multiplier: 1.5

rpg_name: Logic Master
rpg_description: Master of logical deduction
"""
        (skills_dir / "test_logic.yaml").write_text(logic_skill)

        math_skill = """
id: test_math
name: Test Math Skill
description: A test skill for math
category: math

tags:
  - test
  - math
  - calculation

metrics:
  - accuracy
primary_metric: accuracy

accuracy_thresholds:
  1: 0.50
  2: 0.60

xp_multiplier: 1.0
"""
        (skills_dir / "test_math.yaml").write_text(math_skill)

        yield config_dir


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_registry()
    reset_state_manager()
    yield
    reset_registry()
    reset_state_manager()


# =============================================================================
# Type Tests
# =============================================================================

class TestSkillTypes:
    """Tests for skill type definitions."""

    def test_skill_category_enum(self):
        assert SkillCategory.REASONING.value == "reasoning"
        assert SkillCategory.MATH.value == "math"
        assert SkillCategory.CODE.value == "code"

    def test_metric_definition(self):
        metric = MetricDefinition(
            id="accuracy",
            name="Accuracy",
            description="Model accuracy",
            higher_is_better=True,
            range_min=0.0,
            range_max=1.0,
        )
        assert metric.id == "accuracy"
        assert metric.higher_is_better is True

    def test_skill_config_get_threshold(self):
        config = SkillConfig(
            id="test",
            name="Test",
            description="Test skill",
            category=SkillCategory.REASONING,
            accuracy_thresholds={1: 0.60, 2: 0.70, 3: 0.80},
        )

        assert config.get_threshold(1) == 0.60
        assert config.get_threshold(2) == 0.70
        assert config.get_threshold(3) == 0.80
        # Beyond defined thresholds
        assert config.get_threshold(5) == 0.80  # Max defined

    def test_skill_config_default_threshold(self):
        config = SkillConfig(
            id="test",
            name="Test",
            description="Test skill",
            category=SkillCategory.REASONING,
            accuracy_thresholds={},
        )
        # Uses default formula
        assert config.get_threshold(1) == 0.60
        assert config.get_threshold(2) == pytest.approx(0.63)

    def test_skill_state_serialization(self):
        state = SkillState(skill_id="test")
        state.level = 3
        state.xp_total = 150.0
        state.record_result(True)
        state.record_result(False)

        # Serialize
        data = state.to_dict()
        assert data["skill_id"] == "test"
        assert data["level"] == 3
        assert data["xp_total"] == 150.0
        assert data["recent_results"] == [True, False]

        # Deserialize
        restored = SkillState.from_dict(data)
        assert restored.skill_id == "test"
        assert restored.level == 3
        assert restored.accuracy == 0.5  # 1/2

    def test_skill_state_rolling_accuracy(self):
        state = SkillState(skill_id="test", window_size=3)

        state.record_result(True)
        assert state.accuracy == 1.0

        state.record_result(False)
        assert state.accuracy == 0.5

        state.record_result(True)
        assert state.accuracy == pytest.approx(2/3)

        # Window should trim
        state.record_result(True)
        assert len(state.recent_results) == 3
        assert state.accuracy == pytest.approx(2/3)  # [False, True, True]


# =============================================================================
# Loader Tests
# =============================================================================

class TestSkillLoader:
    """Tests for skill loading from YAML."""

    def test_load_skill_config(self, temp_config_dir):
        config = load_skill_config("test_logic", temp_config_dir)

        assert config.id == "test_logic"
        assert config.name == "Test Logic Skill"
        assert config.category == SkillCategory.REASONING
        assert "test" in config.tags
        assert config.xp_multiplier == 1.5
        assert config.get_threshold(1) == 0.60

    def test_load_skill_not_found(self, temp_config_dir):
        with pytest.raises(FileNotFoundError):
            load_skill_config("nonexistent", temp_config_dir)

    def test_discover_skills(self, temp_config_dir):
        skill_ids = discover_skills(temp_config_dir)

        assert "test_logic" in skill_ids
        assert "test_math" in skill_ids
        assert len(skill_ids) == 2

    def test_load_all_skills(self, temp_config_dir):
        skills = load_all_skills(temp_config_dir)

        assert len(skills) == 2
        assert "test_logic" in skills
        assert "test_math" in skills
        assert skills["test_logic"].category == SkillCategory.REASONING
        assert skills["test_math"].category == SkillCategory.MATH

    def test_skill_loader_caching(self, temp_config_dir):
        loader = SkillLoader(temp_config_dir)

        # First load
        skill1 = loader.load("test_logic")

        # Second load should be cached
        skill2 = loader.load("test_logic")
        assert skill1 is skill2  # Same instance

        # Clear cache
        loader.clear_cache()
        skill3 = loader.load("test_logic")
        assert skill1 is not skill3  # New instance

    def test_skill_loader_exists(self, temp_config_dir):
        loader = SkillLoader(temp_config_dir)

        assert loader.exists("test_logic") is True
        assert loader.exists("nonexistent") is False

    def test_dict_to_skill_config_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            _dict_to_skill_config({"name": "Test"})

    def test_dict_to_skill_config_invalid_category(self):
        with pytest.raises(ValueError, match="Invalid skill category"):
            _dict_to_skill_config({
                "id": "test",
                "category": "invalid_category",
            })


# =============================================================================
# Registry Tests
# =============================================================================

class TestSkillRegistry:
    """Tests for skill registry."""

    def test_registry_get(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        skill = registry.get("test_logic")
        assert skill.id == "test_logic"

    def test_registry_get_unknown(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        with pytest.raises(KeyError, match="Unknown skill"):
            registry.get("nonexistent")

    def test_registry_get_or_none(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        skill = registry.get_or_none("test_logic")
        assert skill is not None

        skill = registry.get_or_none("nonexistent")
        assert skill is None

    def test_registry_exists(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        assert registry.exists("test_logic") is True
        assert registry.exists("nonexistent") is False
        assert "test_logic" in registry
        assert "nonexistent" not in registry

    def test_registry_list_ids(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        ids = registry.list_ids()
        assert "test_logic" in ids
        assert "test_math" in ids

    def test_registry_all(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        all_skills = registry.all()
        assert len(all_skills) == 2
        assert "test_logic" in all_skills

    def test_registry_iteration(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        skills = list(registry)
        assert len(skills) == 2

    def test_registry_len(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        assert len(registry) == 2

    def test_registry_by_category(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        reasoning_skills = registry.by_category(SkillCategory.REASONING)
        assert len(reasoning_skills) == 1
        assert reasoning_skills[0].id == "test_logic"

        math_skills = registry.by_category(SkillCategory.MATH)
        assert len(math_skills) == 1
        assert math_skills[0].id == "test_math"

    def test_registry_by_tag(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        test_skills = registry.by_tag("test")
        assert len(test_skills) == 2

        math_skills = registry.by_tag("math")
        assert len(math_skills) == 1
        assert math_skills[0].id == "test_math"

    def test_registry_search(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        # By category
        results = registry.search(category=SkillCategory.REASONING)
        assert len(results) == 1

        # By tags
        results = registry.search(tags=["test", "reasoning"])
        assert len(results) == 1
        assert results[0].id == "test_logic"

        # By name
        results = registry.search(name_contains="math")
        assert len(results) == 1
        assert results[0].id == "test_math"

    def test_registry_refresh(self, temp_config_dir):
        registry = SkillRegistry(temp_config_dir)

        # Load all
        registry.all()

        # Refresh clears cache
        registry.refresh()
        assert registry._loaded is False


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_init_and_get_registry(self, temp_config_dir):
        init_registry(temp_config_dir)

        registry = get_registry()
        assert registry is not None
        assert "test_logic" in registry

    def test_get_skill_convenience(self, temp_config_dir):
        init_registry(temp_config_dir)

        skill = get_skill("test_logic")
        assert skill.id == "test_logic"

    def test_list_skills_convenience(self, temp_config_dir):
        init_registry(temp_config_dir)

        ids = list_skills()
        assert "test_logic" in ids


# =============================================================================
# State Manager Tests
# =============================================================================

class TestSkillStateManager:
    """Tests for skill state management."""

    def test_create_manager(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        assert manager.state_file.parent.exists()

    def test_get_state_creates_new(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        state = manager.get_state("new_skill")
        assert state.skill_id == "new_skill"
        assert state.level == 1

    def test_record_accuracy(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.record_accuracy("test_skill", 0.85, step=100)

        history = manager.get_accuracy_history("test_skill")
        assert len(history) == 1
        assert history[0].accuracy == 0.85
        assert history[0].step == 100

    def test_record_accuracy_with_metadata(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.record_accuracy(
            "test_skill",
            0.75,
            step=200,
            metadata={"problems": 50, "correct": 37}
        )

        history = manager.get_accuracy_history("test_skill")
        assert history[0].metadata["problems"] == 50

    def test_history_limit(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir, history_limit=5)

        for i in range(10):
            manager.record_accuracy("test_skill", 0.5 + i * 0.05, step=i * 100)

        history = manager.get_accuracy_history("test_skill")
        assert len(history) == 5
        assert history[0].step == 500  # Oldest remaining

    def test_record_progression(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.record_progression("test_skill", from_level=1, to_level=2)

        history = manager.get_progression_history("test_skill")
        assert len(history) == 1
        assert history[0].from_level == 1
        assert history[0].to_level == 2

    def test_set_level(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.set_level("test_skill", 5)

        state = manager.get_state("test_skill")
        assert state.level == 5

        # Should have recorded progression
        history = manager.get_progression_history("test_skill")
        assert len(history) == 1

    def test_persistence(self, temp_state_dir):
        # First manager
        manager1 = SkillStateManager(temp_state_dir)
        manager1.record_accuracy("test_skill", 0.80, step=100)
        manager1.set_level("test_skill", 3)

        # New manager loads from disk
        manager2 = SkillStateManager(temp_state_dir)

        state = manager2.get_state("test_skill")
        assert state.level == 3

        history = manager2.get_accuracy_history("test_skill")
        assert len(history) == 1
        assert history[0].accuracy == 0.80

    def test_get_status(self, temp_state_dir, temp_config_dir):
        init_registry(temp_config_dir)
        manager = SkillStateManager(temp_state_dir)

        manager.record_accuracy("test_logic", 0.75, step=100)
        manager.record_accuracy("test_logic", 0.80, step=200)
        manager.record_accuracy("test_logic", 0.85, step=300)

        status = manager.get_status("test_logic")

        assert status["skill_id"] == "test_logic"
        assert status["level"] == 1
        assert status["total_evals"] == 3
        assert status["recent_accuracy"] is not None

    def test_get_all_status(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.record_accuracy("skill_a", 0.70, step=100)
        manager.record_accuracy("skill_b", 0.80, step=100)

        all_status = manager.get_all_status()

        assert "skill_a" in all_status
        assert "skill_b" in all_status

    def test_check_progression(self, temp_state_dir, temp_config_dir):
        init_registry(temp_config_dir)
        manager = SkillStateManager(temp_state_dir)

        # Not enough evals
        should, reason = manager.check_progression("test_logic", min_evals=3)
        assert should is False
        assert "Need 3 evals" in reason

        # Add evals below threshold
        manager.record_accuracy("test_logic", 0.50, step=100)
        manager.record_accuracy("test_logic", 0.55, step=200)
        manager.record_accuracy("test_logic", 0.52, step=300)

        should, reason = manager.check_progression("test_logic", min_evals=3)
        assert should is False
        assert "< threshold" in reason

    def test_progress_if_ready(self, temp_state_dir, temp_config_dir):
        init_registry(temp_config_dir)
        manager = SkillStateManager(temp_state_dir)

        # Add evals above threshold (0.60 for level 1)
        manager.record_accuracy("test_logic", 0.65, step=100)
        manager.record_accuracy("test_logic", 0.70, step=200)
        manager.record_accuracy("test_logic", 0.75, step=300)

        progressed, new_level = manager.progress_if_ready("test_logic", min_evals=3)

        assert progressed is True
        assert new_level == 2

        state = manager.get_state("test_logic")
        assert state.level == 2

    def test_reset_skill(self, temp_state_dir):
        manager = SkillStateManager(temp_state_dir)

        manager.set_level("test_skill", 5)
        manager.record_accuracy("test_skill", 0.90, step=100)

        manager.reset_skill("test_skill")

        state = manager.get_state("test_skill")
        assert state.level == 1

        history = manager.get_accuracy_history("test_skill")
        assert len(history) == 0


class TestGlobalStateManager:
    """Tests for global state manager functions."""

    def test_init_and_get_state_manager(self, temp_state_dir):
        init_state_manager(temp_state_dir)

        manager = get_state_manager()
        assert manager is not None

    def test_get_state_manager_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_state_manager()


# =============================================================================
# Integration Tests
# =============================================================================

class TestSkillsIntegration:
    """Integration tests for the skills module."""

    def test_full_workflow(self, temp_config_dir, temp_state_dir):
        """Test complete workflow: load, record, progress."""
        # Initialize
        init_registry(temp_config_dir)
        init_state_manager(temp_state_dir)

        manager = get_state_manager()

        # Get skill config
        skill = get_skill("test_logic")
        assert skill.category == SkillCategory.REASONING

        # Record evaluations
        for i in range(5):
            manager.record_accuracy("test_logic", 0.65 + i * 0.02, step=(i + 1) * 100)

        # Check status
        status = manager.get_status("test_logic")
        assert status["total_evals"] == 5

        # Progress should trigger (avg > 0.60 threshold)
        progressed, new_level = manager.progress_if_ready("test_logic", min_evals=3)
        assert progressed is True
        assert new_level == 2

    def test_multi_skill_tracking(self, temp_config_dir, temp_state_dir):
        """Test tracking multiple skills."""
        init_registry(temp_config_dir)
        init_state_manager(temp_state_dir)

        manager = get_state_manager()

        # Track both skills
        manager.record_accuracy("test_logic", 0.70, step=100)
        manager.record_accuracy("test_math", 0.80, step=100)

        all_status = manager.get_all_status()
        assert len(all_status) == 2

        logic_status = manager.get_status("test_logic")
        math_status = manager.get_status("test_math")

        assert logic_status["recent_accuracy"] is not None
        assert math_status["recent_accuracy"] is not None


# =============================================================================
# Test with Real Configs (optional)
# =============================================================================

class TestRealConfigs:
    """Tests with real config files (skipped if not present)."""

    def test_load_real_skills(self):
        """Test loading real skill configs."""
        real_config_dir = project_root / "configs"

        if not (real_config_dir / "skills").exists():
            pytest.skip("No real skill configs found")

        skills = discover_skills(real_config_dir)

        if not skills:
            pytest.skip("No skill configs in configs/skills/")

        # Load first skill
        skill = load_skill_config(skills[0], real_config_dir)
        assert skill.id is not None
        assert skill.name is not None
