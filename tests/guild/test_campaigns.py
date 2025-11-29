"""
Tests for Campaign types and effort tracking.

Tests the materials-science inspired metrics:
- peak_skill_levels: Highest skill levels achieved
- peak_metrics: Best metrics (lowest_loss, highest_accuracy)
- skill_effort: Cumulative effort per skill
- level_transitions: Effort spent per level-up
"""

import tempfile
from pathlib import Path

import pytest

from guild.campaigns.types import Campaign, Milestone, CampaignStatus


class TestCampaign:
    """Test Campaign dataclass and serialization."""

    def test_campaign_roundtrip(self):
        """Campaign can be serialized and deserialized."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test Campaign",
                path=path,
                description="A test campaign",
            )

            # Serialize
            data = c.to_dict()
            assert data["id"] == "campaign-001"
            assert data["hero_id"] == "dio-qwen3-0.6b"
            assert data["name"] == "Test Campaign"

            # Deserialize
            c2 = Campaign.from_dict(data, path)
            assert c2.id == c.id
            assert c2.hero_id == c.hero_id
            assert c2.name == c.name

    def test_peak_skill_tracking(self):
        """Peak skill levels are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # First update is always a new peak
            assert c.update_peak_skill("sy", 5) is True
            assert c.peak_skill_levels["sy"] == 5

            # Higher level is a new peak
            assert c.update_peak_skill("sy", 10) is True
            assert c.peak_skill_levels["sy"] == 10

            # Lower level is not a new peak
            assert c.update_peak_skill("sy", 8) is False
            assert c.peak_skill_levels["sy"] == 10  # Unchanged

            # Different skill
            assert c.update_peak_skill("bin", 3) is True
            assert c.peak_skill_levels["bin"] == 3

    def test_peak_metric_tracking(self):
        """Peak metrics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # Loss: lower is better
            assert c.update_peak_metric("lowest_loss", 1.0, lower_is_better=True) is True
            assert c.peak_metrics["lowest_loss"] == 1.0

            assert c.update_peak_metric("lowest_loss", 0.5, lower_is_better=True) is True
            assert c.peak_metrics["lowest_loss"] == 0.5

            assert c.update_peak_metric("lowest_loss", 0.8, lower_is_better=True) is False
            assert c.peak_metrics["lowest_loss"] == 0.5  # Unchanged

            # Accuracy: higher is better
            assert c.update_peak_metric("highest_accuracy", 0.7, lower_is_better=False) is True
            assert c.peak_metrics["highest_accuracy"] == 0.7

            assert c.update_peak_metric("highest_accuracy", 0.9, lower_is_better=False) is True
            assert c.peak_metrics["highest_accuracy"] == 0.9

            assert c.update_peak_metric("highest_accuracy", 0.8, lower_is_better=False) is False
            assert c.peak_metrics["highest_accuracy"] == 0.9  # Unchanged


class TestCampaignEffort:
    """Test effort tracking (materials science metaphor)."""

    def test_add_effort(self):
        """Effort can be accumulated per skill."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # Add effort
            c.add_effort("sy", 10.0)
            assert c.skill_effort["sy"] == 10.0

            c.add_effort("sy", 5.0)
            assert c.skill_effort["sy"] == 15.0

            c.add_effort("bin", 8.0)
            assert c.skill_effort["bin"] == 8.0

            assert c.total_effort == 23.0

    def test_level_transition(self):
        """Level transitions record effort spent."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # Record a level transition
            c.record_level_transition(
                skill_id="sy",
                from_level=1,
                to_level=2,
                effort_spent=15.5,
                plastic_gain=0.3,
                steps_taken=1000,
            )

            assert len(c.level_transitions) == 1
            t = c.level_transitions[0]
            assert t["skill_id"] == "sy"
            assert t["from_level"] == 1
            assert t["to_level"] == 2
            assert t["effort_spent"] == 15.5
            assert t["plastic_gain"] == 0.3
            assert t["efficiency"] == 0.3 / 15.5

    def test_effort_summary(self):
        """Effort summary aggregates stats correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # Add effort
            c.add_effort("sy", 15.5)
            c.add_effort("bin", 8.0)

            # Record transitions
            c.record_level_transition(
                skill_id="sy",
                from_level=1,
                to_level=2,
                effort_spent=15.5,
                plastic_gain=0.3,
                steps_taken=1000,
            )

            summary = c.get_effort_summary()

            assert summary["total_effort"] == 23.5
            assert "sy" in summary["skill_stats"]
            assert "bin" in summary["skill_stats"]
            assert summary["total_level_transitions"] == 1
            assert summary["skill_stats"]["sy"]["levels_gained"] == 1
            assert summary["skill_stats"]["bin"]["levels_gained"] == 0

    def test_journey_summary(self):
        """Journey summary provides one-line overview."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test Campaign",
                path=path,
            )

            # Empty campaign
            assert "No skills yet" in c.journey_summary

            # With skills
            c.update_peak_skill("sy", 10)
            c.update_peak_skill("bin", 5)
            c.current_step = 50000

            summary = c.journey_summary
            assert "Test Campaign" in summary
            assert "50,000" in summary
            assert "sy:L10" in summary
            assert "bin:L5" in summary

    def test_serialization_with_effort(self):
        """Effort fields survive serialization roundtrip."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            c = Campaign(
                id="campaign-001",
                hero_id="dio-qwen3-0.6b",
                name="Test",
                path=path,
            )

            # Add data
            c.add_effort("sy", 15.5)
            c.update_peak_skill("sy", 10)
            c.update_peak_metric("lowest_loss", 0.5, lower_is_better=True)
            c.record_level_transition(
                skill_id="sy",
                from_level=1,
                to_level=2,
                effort_spent=15.5,
                plastic_gain=0.3,
                steps_taken=1000,
            )

            # Serialize and deserialize
            data = c.to_dict()
            c2 = Campaign.from_dict(data, path)

            # Verify
            assert c2.skill_effort["sy"] == 15.5
            assert c2.peak_skill_levels["sy"] == 10
            assert c2.peak_metrics["lowest_loss"] == 0.5
            assert len(c2.level_transitions) == 1
            assert c2.level_transitions[0]["skill_id"] == "sy"


class TestStrainMetrics:
    """Test strain/effort metrics module."""

    def test_strain_tracker_basic(self):
        """StrainTracker computes strain correctly."""
        from guild.metrics import StrainTracker, StrainZone

        tracker = StrainTracker(floor=0.5)

        # Loss above floor = positive strain
        metrics = tracker.update(loss=0.8, step=0)
        assert metrics.strain == pytest.approx(0.3)
        assert metrics.zone == StrainZone.STRETCH  # 0.3 is in stretch zone

        # Loss at floor = zero strain
        metrics = tracker.update(loss=0.5, step=100)
        assert metrics.strain == 0.0
        assert metrics.zone == StrainZone.RECOVERY

        # Loss below floor = zero strain (clamped)
        metrics = tracker.update(loss=0.3, step=200)
        assert metrics.strain == 0.0

    def test_strain_zones(self):
        """Strain zones classify correctly."""
        from guild.metrics import StrainTracker, StrainZone

        tracker = StrainTracker(floor=0.0)

        # Recovery zone
        metrics = tracker.update(loss=0.05, step=0)
        assert metrics.zone == StrainZone.RECOVERY

        # Productive zone
        tracker = StrainTracker(floor=0.0)
        metrics = tracker.update(loss=0.2, step=0)
        assert metrics.zone == StrainZone.PRODUCTIVE

        # Stretch zone
        tracker = StrainTracker(floor=0.0)
        metrics = tracker.update(loss=0.4, step=0)
        assert metrics.zone == StrainZone.STRETCH

        # Overload zone
        tracker = StrainTracker(floor=0.0)
        metrics = tracker.update(loss=0.6, step=0)
        assert metrics.zone == StrainZone.OVERLOAD

    def test_cumulative_effort(self):
        """Cumulative effort accumulates correctly."""
        from guild.metrics import StrainTracker

        tracker = StrainTracker(floor=0.0)

        tracker.update(loss=0.3, step=0)
        tracker.update(loss=0.2, step=1)
        tracker.update(loss=0.1, step=2)

        # Total effort = 0.3 + 0.2 + 0.1 = 0.6
        assert tracker.cumulative_effort == 0.6

    def test_curriculum_hint(self):
        """Curriculum hints are generated correctly."""
        from guild.metrics import StrainTracker, CurriculumAction

        tracker = StrainTracker(floor=0.5)

        # Simulate learning (loss decreasing)
        for i in range(15):
            loss = 1.0 - (i * 0.03)  # 1.0 -> 0.58
            tracker.update(loss=loss, step=i * 100)

        hint = tracker.get_curriculum_hint()
        # Should suggest continuing since we're improving
        assert hint.strain_trend == "improving"

    def test_skill_strain_tracker(self):
        """SkillStrainTracker tracks multiple skills."""
        from guild.metrics import SkillStrainTracker

        tracker = SkillStrainTracker()
        tracker.set_floor("sy", 0.8)
        tracker.set_floor("bin", 0.5)

        # Update different skills
        sy_metrics = tracker.update("sy", loss=1.2, step=0)
        bin_metrics = tracker.update("bin", loss=0.7, step=0)

        assert sy_metrics.strain == pytest.approx(0.4)  # 1.2 - 0.8
        assert bin_metrics.strain == pytest.approx(0.2)  # 0.7 - 0.5

        # Total effort across skills
        total = tracker.get_total_effort()
        assert total == pytest.approx(0.6)
