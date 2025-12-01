# tests/test_reset.py
"""Tests for core/reset.py - environment reset logic."""

from pathlib import Path

from core.reset import reset_environment


def test_reset_environment_clears_pid_and_state_files(tmp_path: Path):
    """
    reset_environment should remove PID files and known state files
    under the provided base_dir, without raising.
    """
    base_dir = tmp_path

    # .pids with a fake PID file
    pids_dir = base_dir / ".pids"
    pids_dir.mkdir()
    pid_file = pids_dir / "daemon.pid"
    pid_file.write_text("999999")  # Likely invalid PID; should be handled gracefully.

    # control/state.json and status files
    control_dir = base_dir / "control"
    status_dir = base_dir / "status"
    control_dir.mkdir()
    status_dir.mkdir()

    state_json = control_dir / "state.json"
    training_status = status_dir / "training_status.json"
    events = status_dir / "events.jsonl"

    state_json.write_text("{}")
    training_status.write_text("{}")
    events.write_text("event\n")

    # Call reset with keep_jobs=True to avoid touching job store
    result = reset_environment(keep_jobs=True, base_dir=base_dir)

    # PID files removed
    assert list(pids_dir.glob("*.pid")) == []
    assert len(result.pids_cleared) == 1

    # State files removed
    assert not state_json.exists()
    assert not training_status.exists()
    assert not events.exists()
    assert len(result.state_files_cleared) == 3

    # No jobs cancelled in this mode
    assert result.jobs_cancelled == 0


def test_reset_environment_handles_missing_dirs(tmp_path: Path):
    """reset_environment should not fail if directories don't exist."""
    base_dir = tmp_path
    # No .pids, control, or status directories

    result = reset_environment(keep_jobs=True, base_dir=base_dir)

    assert result.daemons_stopped == []
    assert result.pids_cleared == []
    assert result.state_files_cleared == []
    assert result.jobs_cancelled == 0


def test_reset_result_as_counts():
    """ResetResult.as_counts() should return counts dict."""
    from core.reset import ResetResult

    result = ResetResult(
        daemons_stopped=[123, 456],
        pids_cleared=[Path("/tmp/a.pid"), Path("/tmp/b.pid"), Path("/tmp/c.pid")],
        state_files_cleared=[Path("/tmp/state.json")],
        jobs_cancelled=5,
    )

    counts = result.as_counts()

    assert counts["daemons_stopped"] == 2
    assert counts["pids_cleared"] == 3
    assert counts["state_files_cleared"] == 1
    assert counts["jobs_cancelled"] == 5
