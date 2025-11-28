"""
Tests for the Skill Engine system.

Tests cover:
- PrimitiveId creation and validation
- SkillConfig loading from YAML
- SkillState tracking
- Engine eval generation
- Per-primitive accuracy tracking
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict

# Import test targets
from guild.skills.primitives import PrimitiveId, PrimitiveMeta, PRIMITIVE_CATALOG, list_tracks
from guild.skills.types import SkillConfig, SkillState, SkillCategory
from guild.skills.engine import SkillEngine
from guild.skills.eval_types import EvalProblem, EvalBatch, EvalResult, EvalResultItem


class TestPrimitiveId:
    """Tests for PrimitiveId dataclass."""

    def test_create_primitive(self):
        """PrimitiveId can be created with required fields."""
        prim = PrimitiveId(
            name="binary_add_no_carry",
            track="binary",
            version="v1"
        )
        assert prim.name == "binary_add_no_carry"
        assert prim.track == "binary"
        assert prim.version == "v1"

    def test_primitive_str(self):
        """PrimitiveId string format is track/name@version."""
        prim = PrimitiveId(
            name="binary_add_no_carry",
            track="binary",
            version="v1"
        )
        assert str(prim) == "binary/binary_add_no_carry@v1"

    def test_primitive_frozen(self):
        """PrimitiveId is immutable (frozen)."""
        prim = PrimitiveId(name="test", track="test", version="v1")
        with pytest.raises(AttributeError):
            prim.name = "changed"

    def test_primitive_hashable(self):
        """PrimitiveId can be used as dict key."""
        prim1 = PrimitiveId(name="test", track="t", version="v1")
        prim2 = PrimitiveId(name="test", track="t", version="v1")

        d = {prim1: "value"}
        assert d[prim2] == "value"

    def test_primitive_equality(self):
        """PrimitiveId equality works correctly."""
        prim1 = PrimitiveId(name="test", track="t", version="v1")
        prim2 = PrimitiveId(name="test", track="t", version="v1")
        prim3 = PrimitiveId(name="other", track="t", version="v1")

        assert prim1 == prim2
        assert prim1 != prim3


class TestPrimitiveCatalog:
    """Tests for the PRIMITIVE_CATALOG."""

    def test_catalog_has_tracks(self):
        """Catalog has expected tracks."""
        tracks = list_tracks()
        assert "binary" in tracks
        assert "arithmetic" in tracks
        assert "logic" in tracks

    def test_catalog_has_primitives(self):
        """Each track has primitives (as PrimitiveMeta objects)."""
        for track, primitives in PRIMITIVE_CATALOG.items():
            assert len(primitives) > 0, f"Track {track} has no primitives"
            for prim in primitives:
                # Catalog contains PrimitiveMeta, which has .id (a PrimitiveId)
                assert isinstance(prim, PrimitiveMeta)
                assert prim.track == track

    def test_binary_track_primitives(self):
        """Binary track has expected primitives."""
        binary_prims = PRIMITIVE_CATALOG.get("binary", [])
        prim_names = [p.name for p in binary_prims]

        expected = [
            "binary_add_no_carry",
            "binary_add_with_carry",
            "binary_sub_no_borrow",  # Actual name in catalog
            "bitwise_and",
            "bitwise_or",
        ]
        for name in expected:
            assert name in prim_names, f"Missing {name} in binary track"


class TestSkillState:
    """Tests for SkillState dataclass."""

    def test_create_state(self):
        """SkillState can be created with defaults."""
        state = SkillState(skill_id="bin")
        assert state.skill_id == "bin"
        assert state.level == 1
        assert state.xp_total == 0.0
        assert state.primitive_accuracy == {}

    def test_state_with_accuracy(self):
        """SkillState tracks primitive accuracy."""
        state = SkillState(
            skill_id="bin",
            level=3,
            primitive_accuracy={
                "binary_add_no_carry": 0.95,
                "binary_subtract": 0.70,
            }
        )
        assert state.primitive_accuracy["binary_add_no_carry"] == 0.95
        assert state.primitive_accuracy["binary_subtract"] == 0.70

    def test_state_accuracy_property(self):
        """SkillState.accuracy calculates rolling average."""
        state = SkillState(skill_id="bin")
        # Use record_result to add results (recent_results is read-only property)
        for result in [True, True, True, False, True]:
            state.record_result(result)
        assert state.accuracy == 0.8

    def test_state_empty_accuracy(self):
        """Empty results returns 0.0 accuracy."""
        state = SkillState(skill_id="bin")
        assert state.accuracy == 0.0

    def test_record_result(self):
        """record_result adds to recent_results."""
        state = SkillState(skill_id="bin")
        state.record_result(True)
        state.record_result(False)
        assert state.recent_results == [True, False]

    def test_record_result_window(self):
        """record_result trims to window_size."""
        state = SkillState(skill_id="bin", window_size=3)
        for _ in range(5):
            state.record_result(True)
        assert len(state.recent_results) == 3


class TestEvalTypes:
    """Tests for eval-related types."""

    def test_eval_problem(self):
        """EvalProblem stores problem data."""
        prob = EvalProblem(
            prompt="What is 1 + 1?",
            expected="2",
            primitive_id="addition",
            metadata={"level": 1}
        )
        assert prob.prompt == "What is 1 + 1?"
        assert prob.expected == "2"
        assert prob.primitive_id == "addition"

    def test_eval_batch(self):
        """EvalBatch contains multiple problems."""
        problems = [
            EvalProblem(prompt="1+1?", expected="2", primitive_id="add"),
            EvalProblem(prompt="2+2?", expected="4", primitive_id="add"),
        ]
        batch = EvalBatch(
            skill_id="math",
            level=1,
            problems=problems,
        )
        assert len(batch.problems) == 2
        assert batch.skill_id == "math"

    def test_eval_result(self):
        """EvalResult tracks per-primitive accuracy."""
        result = EvalResult(
            accuracy=0.8,
            per_primitive_accuracy={
                "binary_add": 0.9,
                "binary_sub": 0.7,
            },
            num_examples=10,
            metadata={"skill_id": "bin", "level": 1},
        )
        assert result.accuracy == 0.8
        assert result.per_primitive_accuracy["binary_add"] == 0.9
        assert result.num_examples == 10


class TestSkillEngine:
    """Tests for SkillEngine."""

    def test_engine_singleton(self):
        """get_engine returns singleton."""
        from guild.skills import get_engine
        e1 = get_engine()
        e2 = get_engine()
        assert e1 is e2

    def test_list_skills(self):
        """Engine lists available skills."""
        from guild.skills import get_engine
        engine = get_engine()
        skills = engine.list_skills()
        assert isinstance(skills, list)
        # Should have at least bin and sy
        assert "bin" in skills or "sy" in skills

    def test_get_skill(self):
        """Engine can get skill by ID."""
        from guild.skills import get_engine
        engine = get_engine()
        skills = engine.list_skills()
        if skills:
            skill = engine.get(skills[0])
            assert skill is not None

    def test_get_state(self):
        """Engine tracks skill state."""
        from guild.skills import get_engine
        engine = get_engine()
        skills = engine.list_skills()
        if skills:
            state = engine.get_state(skills[0])
            assert isinstance(state, SkillState)

    def test_generate_eval_batch(self):
        """Engine generates eval batches."""
        from guild.skills import get_engine
        engine = get_engine()

        # Try to generate for any available skill
        skills = engine.list_skills()
        for skill_id in skills:
            try:
                batch = engine.generate_eval_batch(skill_id, level=1, count=5)
                if batch is not None:
                    assert isinstance(batch, EvalBatch)
                    assert len(batch.problems) <= 5
                    break
            except Exception:
                continue


class TestSkillConfig:
    """Tests for SkillConfig loading."""

    def test_config_fields(self):
        """SkillConfig has expected fields."""
        config = SkillConfig(
            id="test",
            name="Test Skill",
            description="A test skill",
            category=SkillCategory.MATH,
            max_level=10,
        )
        assert config.id == "test"
        assert config.name == "Test Skill"
        assert config.max_level == 10

    def test_config_primitives(self):
        """SkillConfig stores primitives list."""
        config = SkillConfig(
            id="test",
            name="Test",
            description="Test skill",
            category=SkillCategory.REASONING,
            primitives=[
                {"id": "prim1", "name": "Primitive 1"},
                {"id": "prim2", "name": "Primitive 2"},
            ]
        )
        assert len(config.primitives) == 2

    def test_config_passive_id(self):
        """SkillConfig has passive_id field."""
        config = SkillConfig(
            id="bin",
            name="Binary",
            description="Binary arithmetic",
            category=SkillCategory.MATH,
            passive_id="binary_arithmetic",
        )
        assert config.passive_id == "binary_arithmetic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
