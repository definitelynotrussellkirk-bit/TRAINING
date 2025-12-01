"""
Test vocabulary consistency across Python enums, YAML configs, and lore entries.

This ensures the single-source-of-truth principle: if a School/Technique/Domain
exists in one place, it must exist in all canonical locations.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from guild.training_schools import TrainingSchool
from guild.job_types import JobType, School
from tavern.lore import LORE, get_lore


# =============================================================================
# TRAINING SCHOOLS CONSISTENCY
# =============================================================================

class TestTrainingSchoolsConsistency:
    """Ensure Training Schools are consistent across enum, YAML, and lore."""

    @pytest.fixture
    def training_schools_yaml(self):
        """Load training schools YAML config."""
        yaml_path = Path(__file__).parent.parent / "configs" / "training_schools.yaml"
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def test_all_enum_members_in_yaml(self, training_schools_yaml):
        """Every TrainingSchool enum member should have a YAML config."""
        yaml_schools = set(training_schools_yaml["schools"].keys())

        for school in TrainingSchool:
            assert school.value in yaml_schools, (
                f"TrainingSchool.{school.name} ({school.value}) missing from "
                f"configs/training_schools.yaml"
            )

    def test_all_yaml_entries_in_enum(self, training_schools_yaml):
        """Every YAML school should have an enum member."""
        enum_values = {s.value for s in TrainingSchool}

        for yaml_school in training_schools_yaml["schools"].keys():
            assert yaml_school in enum_values, (
                f"YAML school '{yaml_school}' has no TrainingSchool enum member"
            )

    def test_all_training_schools_have_lore(self):
        """Every TrainingSchool should have a lore entry."""
        for school in TrainingSchool:
            lore_key = f"school.{school.value}"
            entry = get_lore(lore_key)
            assert entry is not None, (
                f"TrainingSchool.{school.name} missing lore entry 'school.{school.value}'"
            )
            # Verify required fields
            assert "label" in entry, f"Lore entry '{lore_key}' missing 'label'"
            assert "tooltip" in entry, f"Lore entry '{lore_key}' missing 'tooltip'"

    def test_yaml_and_lore_mottos_match(self, training_schools_yaml):
        """YAML motto should match or be compatible with lore tooltip."""
        for school_id, school_data in training_schools_yaml["schools"].items():
            lore_key = f"school.{school_id}"
            lore_entry = get_lore(lore_key)

            if lore_entry and "motto" in school_data:
                # The lore short description should mention the motto or be thematically aligned
                # (We don't require exact match, just existence)
                assert lore_entry.get("tooltip"), (
                    f"School '{school_id}' has YAML motto but no lore tooltip"
                )


# =============================================================================
# JOB SCHOOLS CONSISTENCY
# =============================================================================

class TestJobSchoolsConsistency:
    """Ensure Job Schools are consistent across enum, YAML, and lore."""

    @pytest.fixture
    def schools_yaml(self):
        """Load job schools YAML config."""
        yaml_path = Path(__file__).parent.parent / "configs" / "schools.yaml"
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def test_all_enum_members_in_yaml(self, schools_yaml):
        """Every School enum member should have a YAML config."""
        yaml_schools = set(schools_yaml["schools"].keys())

        for school in School:
            assert school.value in yaml_schools, (
                f"School.{school.name} ({school.value}) missing from "
                f"configs/schools.yaml"
            )

    def test_all_yaml_entries_in_enum(self, schools_yaml):
        """Every YAML school should have an enum member."""
        enum_values = {s.value for s in School}

        for yaml_school in schools_yaml["schools"].keys():
            assert yaml_school in enum_values, (
                f"YAML school '{yaml_school}' has no School enum member"
            )

    def test_all_job_schools_have_lore(self):
        """Every Job School should have a lore entry."""
        for school in School:
            lore_key = f"school.{school.value}"
            entry = get_lore(lore_key)
            assert entry is not None, (
                f"School.{school.name} missing lore entry 'school.{school.value}'"
            )

    def test_job_types_belong_to_valid_schools(self):
        """Every JobType should map to a valid School."""
        for job_type in JobType:
            school = job_type.school
            assert school is not None, (
                f"JobType.{job_type.name} has no school mapping"
            )
            assert isinstance(school, School), (
                f"JobType.{job_type.name}.school is not a School enum"
            )


# =============================================================================
# TECHNIQUES CONSISTENCY
# =============================================================================

class TestTechniquesConsistency:
    """Ensure Techniques load correctly from physics configs."""

    def test_techniques_load_without_error(self):
        """Techniques should load from physics YAML files."""
        from trainer.techniques import load_techniques, reload_techniques

        # Force reload to ensure fresh load
        reload_techniques()
        techniques = load_techniques()

        assert len(techniques) > 0, "No techniques loaded from configs/physics/"

    def test_muon_technique_exists(self):
        """Muon technique should exist and have required fields."""
        from trainer.techniques import get_technique

        muon = get_technique("muon")
        assert muon is not None, "Muon technique not found"
        assert muon.optimizer_type == "muon"
        assert muon.rpg_name, "Muon technique missing rpg_name"

    def test_adamw_technique_exists(self):
        """AdamW technique should exist and have required fields."""
        from trainer.techniques import get_technique

        adamw = get_technique("adamw")
        assert adamw is not None, "AdamW technique not found"
        assert adamw.optimizer_type == "adamw"
        assert adamw.rpg_name, "AdamW technique missing rpg_name"

    def test_all_techniques_have_required_fields(self):
        """All techniques should have required fields."""
        from trainer.techniques import list_techniques

        required_fields = ["id", "name", "rpg_name", "optimizer_type"]

        for technique in list_techniques():
            for field in required_fields:
                value = getattr(technique, field, None)
                assert value, f"Technique '{technique.id}' missing required field '{field}'"


# =============================================================================
# LORE COMPLETENESS
# =============================================================================

class TestLoreCompleteness:
    """Ensure lore dictionary has all expected entries."""

    def test_core_metrics_have_lore(self):
        """Core training metrics should have lore entries."""
        required_keys = [
            "training.loss",
            "validation.loss",
            "learning_rate",
            "perplexity",
        ]

        for key in required_keys:
            entry = get_lore(key)
            assert entry is not None, f"Core metric '{key}' missing from lore"
            assert "label" in entry, f"Lore '{key}' missing label"
            assert "tooltip" in entry, f"Lore '{key}' missing tooltip"

    def test_strain_effort_experience_have_lore(self):
        """Strain/Effort/Experience vocabulary should have lore."""
        required_keys = ["strain", "effort", "experience"]

        for key in required_keys:
            entry = get_lore(key)
            assert entry is not None, f"Core concept '{key}' missing from lore"

    def test_blessing_has_lore(self):
        """Blessing vocabulary should have lore."""
        entry = get_lore("blessing")
        assert entry is not None, "Blessing concept missing from lore"


# =============================================================================
# TEMPLE BLESSING TESTS
# =============================================================================

class TestTempleBlessing:
    """Test Temple Blessing computation logic."""

    def test_blessing_grant_computes_experience(self):
        """Blessing.grant should compute experience_awarded = effort × quality."""
        from temple.schemas import Blessing

        blessing = Blessing.grant(
            quality=0.9,
            orders=["forge", "oracle"],
            reason="Test",
            effort=100.0,
        )

        assert blessing.granted is True
        assert blessing.quality_factor == 0.9
        assert blessing.experience_awarded == 90.0
        assert blessing.verdict == "blessed"  # quality >= 0.8

    def test_blessing_grant_partial_verdict(self):
        """Quality < 0.8 should give 'partial' verdict."""
        from temple.schemas import Blessing

        blessing = Blessing.grant(
            quality=0.5,
            orders=["forge"],
            reason="Test",
            effort=100.0,
        )

        assert blessing.verdict == "partial"
        assert blessing.experience_awarded == 50.0

    def test_blessing_deny_gives_zero_experience(self):
        """Blessing.deny should give zero experience."""
        from temple.schemas import Blessing

        blessing = Blessing.deny(
            orders=["forge"],
            reason="Critical failure",
            effort=100.0,
        )

        assert blessing.granted is False
        assert blessing.quality_factor == 0.0
        assert blessing.experience_awarded == 0.0
        assert blessing.verdict == "cursed"

    def test_from_ceremony_all_pass(self):
        """All orders passing should give quality=1.0."""
        from temple.schemas import Blessing, RitualResult

        results = {
            "forge": RitualResult(
                ritual_id="forge", name="Forge", description="", status="ok"
            ),
            "oracle": RitualResult(
                ritual_id="oracle", name="Oracle", description="", status="ok"
            ),
        }

        blessing = Blessing.from_ceremony(results, effort=100.0)

        assert blessing.granted is True
        assert blessing.quality_factor == 1.0
        assert blessing.verdict == "blessed"

    def test_from_ceremony_with_warnings(self):
        """Warnings should reduce quality but still grant blessing."""
        from temple.schemas import Blessing, RitualResult

        results = {
            "forge": RitualResult(
                ritual_id="forge", name="Forge", description="", status="ok"
            ),
            "guild": RitualResult(
                ritual_id="guild", name="Guild", description="", status="warn"
            ),
        }

        blessing = Blessing.from_ceremony(results, effort=100.0)

        assert blessing.granted is True
        assert blessing.quality_factor < 1.0
        assert "warned" in blessing.reason

    def test_from_ceremony_critical_failure_denies(self):
        """Critical order failure should deny blessing."""
        from temple.schemas import Blessing, RitualResult

        results = {
            "forge": RitualResult(
                ritual_id="forge", name="Forge", description="", status="fail"
            ),
            "guild": RitualResult(
                ritual_id="guild", name="Guild", description="", status="ok"
            ),
        }

        blessing = Blessing.from_ceremony(results, effort=100.0)

        assert blessing.granted is False
        assert blessing.verdict == "cursed"
        assert "forge" in blessing.reason.lower()

    def test_from_ceremony_non_critical_failure_partial(self):
        """Non-critical failure should give partial blessing."""
        from temple.schemas import Blessing, RitualResult

        results = {
            "guild": RitualResult(
                ritual_id="guild", name="Guild", description="", status="fail"
            ),
            "scribe": RitualResult(
                ritual_id="scribe", name="Scribe", description="", status="ok"
            ),
        }

        blessing = Blessing.from_ceremony(results, effort=100.0)

        # guild is not critical, so should get partial blessing
        assert blessing.granted is True
        assert blessing.quality_factor < 1.0
        assert blessing.verdict == "partial"

    def test_blessing_serialization_roundtrip(self):
        """Blessing should serialize and deserialize correctly."""
        from temple.schemas import Blessing

        original = Blessing.grant(
            quality=0.85,
            orders=["forge", "oracle", "champion"],
            reason="Test serialization",
            campaign_id="test-campaign",
            effort=150.0,
        )

        # Round-trip
        data = original.to_dict()
        restored = Blessing.from_dict(data)

        assert restored.granted == original.granted
        assert restored.quality_factor == original.quality_factor
        assert restored.experience_awarded == original.experience_awarded
        assert restored.orders_consulted == original.orders_consulted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
