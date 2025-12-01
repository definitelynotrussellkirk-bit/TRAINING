# tests/test_evaluation_ledger.py
"""Tests for core/evaluation_ledger.py - evaluation result storage."""

from pathlib import Path

import pytest

from core.evaluation_ledger import (
    EvalRecord,
    EvaluationLedger,
    extract_checkpoint_step,
    record_job_eval,
)


def test_eval_record_basic():
    """EvalRecord should store basic eval data."""
    record = EvalRecord(
        checkpoint_step=175000,
        skill="bin",
        level=5,
        accuracy=0.85,
        correct=85,
        total=100,
        timestamp="2025-11-29T12:00:00",
    )

    assert record.checkpoint_step == 175000
    assert record.skill == "bin"
    assert record.level == 5
    assert record.accuracy == 0.85
    assert record.key == "175000:bin:5"


def test_eval_record_with_job_fields():
    """EvalRecord should support job system fields."""
    record = EvalRecord(
        checkpoint_step=175000,
        skill="bin",
        level=5,
        accuracy=0.85,
        correct=85,
        total=100,
        timestamp="2025-11-29T12:00:00",
        # Job system fields
        hero_id="dio-qwen3-0.6b",
        campaign_id="campaign-001",
        checkpoint_id="checkpoint-175000",
        checkpoint_path="campaigns/dio-qwen3-0.6b/campaign-001/checkpoints/checkpoint-175000",
        context_hash="abc123",
        job_id="job-xyz",
        worker_id="eval_worker_1",
    )

    assert record.hero_id == "dio-qwen3-0.6b"
    assert record.campaign_id == "campaign-001"
    assert record.job_id == "job-xyz"
    assert record.identity_key == "dio-qwen3-0.6b:campaign-001:175000:bin:5"


def test_eval_record_display_summary():
    """display_summary should format nicely."""
    record = EvalRecord(
        checkpoint_step=175000,
        skill="bin",
        level=5,
        accuracy=0.85,
        correct=85,
        total=100,
        timestamp="2025-11-29T12:00:00",
    )

    assert "85.0%" in record.display_summary
    assert "bin L5" in record.display_summary
    assert "85/100" in record.display_summary


def test_eval_record_serialization():
    """EvalRecord should serialize to/from dict."""
    record = EvalRecord(
        checkpoint_step=175000,
        skill="bin",
        level=5,
        accuracy=0.85,
        correct=85,
        total=100,
        timestamp="2025-11-29T12:00:00",
        hero_id="dio-qwen3-0.6b",
        job_id="job-123",
    )

    data = record.to_dict()
    restored = EvalRecord.from_dict(data)

    assert restored.checkpoint_step == 175000
    assert restored.hero_id == "dio-qwen3-0.6b"
    assert restored.job_id == "job-123"
    assert restored.accuracy == 0.85


def test_extract_checkpoint_step():
    """extract_checkpoint_step should handle various formats."""
    # Simple format
    assert extract_checkpoint_step("checkpoint-175000") == 175000

    # Full path
    assert extract_checkpoint_step(
        "campaigns/dio/campaign-001/checkpoints/checkpoint-175000"
    ) == 175000

    # Canonical name with timestamp
    assert extract_checkpoint_step(
        "checkpoint-175000-20251129-1430"
    ) == 175000

    # Invalid format
    assert extract_checkpoint_step("no-checkpoint-here") == 0


def test_evaluation_ledger_record_and_query(tmp_path: Path):
    """EvaluationLedger should record and query evals."""
    ledger = EvaluationLedger(base_dir=tmp_path)

    # Record an eval
    record = EvalRecord(
        checkpoint_step=175000,
        skill="bin",
        level=5,
        accuracy=0.85,
        correct=85,
        total=100,
        timestamp="2025-11-29T12:00:00",
        hero_id="dio-qwen3-0.6b",
        campaign_id="campaign-001",
    )

    result = ledger.record(record)
    assert result is True

    # Query by checkpoint
    evals = ledger.get_by_checkpoint(175000)
    assert len(evals) == 1
    assert evals[0].skill == "bin"

    # Query by skill
    evals = ledger.get_by_skill("bin", level=5)
    assert len(evals) == 1

    # Query by hero/campaign
    evals = ledger.get_by_hero_campaign("dio-qwen3-0.6b", "campaign-001")
    assert len(evals) == 1

    # Duplicate record should return False
    result = ledger.record(record)
    assert result is False


def test_evaluation_ledger_get_checkpoint_skills(tmp_path: Path):
    """get_checkpoint_skills should return latest eval per skill."""
    ledger = EvaluationLedger(base_dir=tmp_path)

    # Record multiple evals for same checkpoint
    for skill, level, acc in [("bin", 1, 0.8), ("bin", 2, 0.7), ("sy", 1, 0.9)]:
        ledger.record(EvalRecord(
            checkpoint_step=175000,
            skill=skill,
            level=level,
            accuracy=acc,
            correct=int(acc * 100),
            total=100,
            timestamp="2025-11-29T12:00:00",
            hero_id="dio",
            campaign_id="c1",
        ))

    skills = ledger.get_checkpoint_skills(175000)

    assert "bin:1" in skills
    assert "bin:2" in skills
    assert "sy:1" in skills
    assert skills["bin:1"].accuracy == 0.8
    assert skills["sy:1"].accuracy == 0.9


def test_evaluation_ledger_summary(tmp_path: Path):
    """summary() should return aggregate stats."""
    ledger = EvaluationLedger(base_dir=tmp_path)

    # Record some evals
    for step in [175000, 180000]:
        ledger.record(EvalRecord(
            checkpoint_step=step,
            skill="bin",
            level=1,
            accuracy=0.8 + step / 1000000,  # Unique accuracy
            correct=80,
            total=100,
            timestamp="2025-11-29T12:00:00",
        ))

    summary = ledger.summary()

    assert summary["total_evaluations"] == 2
    assert "bin" in summary["by_skill"]
    assert summary["by_skill"]["bin"]["count"] == 2


# =============================================================================
# STABILITY TESTS - Prevent regression of rubber-banding bug
# =============================================================================


class TestRerunSafety:
    """Test that re-running evaluations doesn't double-count."""

    def test_duplicate_write_returns_false(self, tmp_path: Path):
        """Writing the same evaluation twice returns False on second write."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        record = EvalRecord(
            checkpoint_step=1000,
            skill="bin",
            level=1,
            accuracy=0.8,
            correct=4,
            total=5,
            timestamp="2025-11-29T12:00:00",
        )

        # First write should succeed
        assert ledger.record(record) is True

        # Second write of same evaluation should return False
        record2 = EvalRecord(
            checkpoint_step=1000,
            skill="bin",
            level=1,
            accuracy=0.9,  # Different accuracy
            correct=5,
            total=5,
            timestamp="2025-11-29T12:00:00",
        )
        assert ledger.record(record2) is False

    def test_count_stable_after_duplicate(self, tmp_path: Path):
        """Eval count stays stable after duplicate write attempts."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        record = EvalRecord(
            checkpoint_step=1000,
            skill="bin",
            level=1,
            accuracy=0.8,
            correct=4,
            total=5,
            timestamp="2025-11-29T12:00:00",
        )

        ledger.record(record)
        count_after_first = ledger.get_eval_count()

        # Try to write duplicates
        for _ in range(5):
            ledger.record(record)

        count_after_duplicates = ledger.get_eval_count()
        assert count_after_first == count_after_duplicates == 1

    def test_different_levels_count_separately(self, tmp_path: Path):
        """Same checkpoint+skill at different levels count as separate evals."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        for level in [1, 2, 3]:
            record = EvalRecord(
                checkpoint_step=1000,
                skill="bin",
                level=level,
                accuracy=0.8,
                correct=4,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger.record(record)

        assert ledger.get_eval_count() == 3
        assert ledger.get_eval_count(skill="bin") == 3
        assert ledger.get_eval_count(skill="bin", level=1) == 1

    def test_idempotent_write_check(self, tmp_path: Path):
        """is_idempotent_write correctly identifies existing evaluations."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        record = EvalRecord(
            checkpoint_step=1000,
            skill="bin",
            level=1,
            accuracy=0.8,
            correct=4,
            total=5,
            timestamp="2025-11-29T12:00:00",
        )

        # Before write
        assert ledger.is_idempotent_write(1000, "bin", 1) is False

        # After write
        ledger.record(record)
        assert ledger.is_idempotent_write(1000, "bin", 1) is True

        # Different checkpoint
        assert ledger.is_idempotent_write(2000, "bin", 1) is False


class TestCrashRecovery:
    """Test that partial/crashed writes don't corrupt state."""

    def test_reload_after_write(self, tmp_path: Path):
        """Creating new ledger instance sees all previous writes."""
        ledger1 = EvaluationLedger(base_dir=tmp_path)

        # Write some evaluations
        for step in [1000, 2000, 3000]:
            record = EvalRecord(
                checkpoint_step=step,
                skill="bin",
                level=1,
                accuracy=0.8,
                correct=4,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger1.record(record)

        count_before = ledger1.get_eval_count()

        # Create new instance (simulating restart)
        ledger2 = EvaluationLedger(base_dir=tmp_path)

        count_after = ledger2.get_eval_count()
        assert count_before == count_after == 3

    def test_partial_session_recovery(self, tmp_path: Path):
        """Simulating crash after N writes, restart continues correctly."""
        ledger1 = EvaluationLedger(base_dir=tmp_path)

        # Write N evaluations
        N = 5
        for i in range(N):
            record = EvalRecord(
                checkpoint_step=1000 + i * 1000,
                skill="bin",
                level=1,
                accuracy=0.8,
                correct=4,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger1.record(record)

        # "Crash" and restart
        ledger2 = EvaluationLedger(base_dir=tmp_path)
        assert ledger2.get_eval_count() == N

        # Continue with more writes
        for i in range(3):
            record = EvalRecord(
                checkpoint_step=10000 + i * 1000,
                skill="sy",
                level=1,
                accuracy=0.9,
                correct=5,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger2.record(record)

        assert ledger2.get_eval_count() == N + 3

        # Verify with fresh instance
        ledger3 = EvaluationLedger(base_dir=tmp_path)
        assert ledger3.get_eval_count() == N + 3


class TestLedgerVsFilesystem:
    """Test that stray files don't affect reported counts."""

    def test_ledger_is_source_of_truth(self, tmp_path: Path):
        """Ledger file is the only source of truth, not filesystem."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        # Write 5 evaluations
        for i in range(5):
            record = EvalRecord(
                checkpoint_step=1000 + i * 1000,
                skill="bin",
                level=1,
                accuracy=0.8,
                correct=4,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger.record(record)

        # Create many fake checkpoint directories
        models_dir = tmp_path / "models" / "current_model"
        models_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            fake_dir = models_dir / f"checkpoint-fake{i}"
            fake_dir.mkdir(parents=True)

        # Count should still be 5 (filesystem ignored)
        assert ledger.get_eval_count() == 5


class TestCanonicalInterface:
    """Test the canonical interface methods."""

    def test_get_eval_count_filters(self, tmp_path: Path):
        """get_eval_count correctly filters by skill and level."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        # Add mixed evaluations
        for skill in ["bin", "sy"]:
            for level in [1, 2, 3]:
                record = EvalRecord(
                    checkpoint_step=1000 + level * 100,
                    skill=skill,
                    level=level,
                    accuracy=0.8,
                    correct=4,
                    total=5,
                    timestamp="2025-11-29T12:00:00",
                )
                ledger.record(record)

        assert ledger.get_eval_count() == 6
        assert ledger.get_eval_count(skill="bin") == 3
        assert ledger.get_eval_count(skill="sy") == 3
        assert ledger.get_eval_count(skill="bin", level=1) == 1
        assert ledger.get_eval_count(skill="nonexistent") == 0

    def test_get_eval_breakdown(self, tmp_path: Path):
        """get_eval_breakdown returns correct counts by skill."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        for skill, count in [("bin", 3), ("sy", 5)]:
            for i in range(count):
                record = EvalRecord(
                    checkpoint_step=1000 + i * 1000,
                    skill=skill,
                    level=1,
                    accuracy=0.8,
                    correct=4,
                    total=5,
                    timestamp="2025-11-29T12:00:00",
                )
                ledger.record(record)

        breakdown = ledger.get_eval_breakdown()
        assert breakdown == {"bin": 3, "sy": 5}

    def test_unique_key_format(self, tmp_path: Path):
        """get_unique_key returns correct format."""
        ledger = EvaluationLedger(base_dir=tmp_path)
        key = ledger.get_unique_key(1000, "bin", 5)
        assert key == "1000:bin:5"

    def test_summary_count_matches_get_eval_count(self, tmp_path: Path):
        """summary() total_evaluations matches get_eval_count()."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        for i in range(7):
            record = EvalRecord(
                checkpoint_step=1000 + i * 1000,
                skill="bin" if i % 2 == 0 else "sy",
                level=1,
                accuracy=0.8,
                correct=4,
                total=5,
                timestamp="2025-11-29T12:00:00",
            )
            ledger.record(record)

        summary = ledger.summary()
        assert summary["total_evaluations"] == ledger.get_eval_count()

    def test_summary_by_skill_matches_breakdown(self, tmp_path: Path):
        """summary() by_skill counts match get_eval_breakdown()."""
        ledger = EvaluationLedger(base_dir=tmp_path)

        for skill, count in [("bin", 4), ("sy", 3)]:
            for i in range(count):
                record = EvalRecord(
                    checkpoint_step=1000 + i * 1000,
                    skill=skill,
                    level=1,
                    accuracy=0.8,
                    correct=4,
                    total=5,
                    timestamp="2025-11-29T12:00:00",
                )
                ledger.record(record)

        summary = ledger.summary()
        breakdown = ledger.get_eval_breakdown()

        for skill, count in breakdown.items():
            assert summary["by_skill"][skill]["count"] == count
