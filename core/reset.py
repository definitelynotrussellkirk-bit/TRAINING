# core/reset.py
"""
Reset training environment while preserving models/campaigns.

Centralizes reset logic used by both CLI (training reset) and
HTTP API (POST /api/reset).
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ResetResult:
    """
    Result of a reset_environment call.

    We keep enough detail for CLI to print nice messages, while
    HTTP/JSON callers can just look at counts.
    """
    daemons_stopped: List[int]
    pids_cleared: List[Path]
    state_files_cleared: List[Path]
    jobs_cancelled: int

    def as_counts(self) -> dict:
        """Return a dict of simple counts for JSON responses."""
        return {
            "daemons_stopped": len(self.daemons_stopped),
            "pids_cleared": len(self.pids_cleared),
            "state_files_cleared": len(self.state_files_cleared),
            "jobs_cancelled": self.jobs_cancelled,
        }


def reset_environment(*, keep_jobs: bool = False, base_dir: Path | None = None) -> ResetResult:
    """
    Reset training environment while preserving models/campaigns.

    Clears runtime state that may be stale or corrupted:
      - .pids/ (daemon PID files)
      - control/state.json
      - status/training_status.json
      - status/events.jsonl

    Optionally cancels pending/claimed/running jobs in the job store
    unless keep_jobs=True.

    Args:
        keep_jobs: If True, do NOT cancel jobs in the job store.
        base_dir: Override base directory (used mainly for tests).

    Returns:
        ResetResult with counts and details of what was cleared.
    """
    if base_dir is None:
        from core.paths import get_base_dir  # local import to avoid cycles
        base_dir = get_base_dir()

    daemons_stopped: list[int] = []
    pids_cleared: list[Path] = []
    state_files_cleared: list[Path] = []
    jobs_cancelled = 0

    # Step 1: Stop daemons
    pids_dir = base_dir / ".pids"
    if pids_dir.exists():
        for pid_file in pids_dir.glob("*.pid"):
            try:
                pid_text = pid_file.read_text().strip()
                pid = int(pid_text)
            except (ValueError, OSError):
                logger.debug("Invalid PID file %s", pid_file)
                continue

            try:
                os.kill(pid, signal.SIGTERM)
                daemons_stopped.append(pid)
            except (ProcessLookupError, PermissionError, OSError) as e:
                logger.debug("Failed to kill PID %s from %s: %s", pid, pid_file, e)

        # Step 2: Clear PID files
        for pid_file in pids_dir.glob("*.pid"):
            try:
                pid_file.unlink()
                pids_cleared.append(pid_file)
            except OSError as e:
                logger.debug("Failed to delete PID file %s: %s", pid_file, e)

    # Step 3: Clear state files
    state_files = [
        base_dir / "control" / "state.json",
        base_dir / "status" / "training_status.json",
        base_dir / "status" / "events.jsonl",
    ]
    for state_file in state_files:
        if state_file.exists():
            try:
                state_file.unlink()
                state_files_cleared.append(state_file)
            except OSError as e:
                logger.debug("Failed to delete state file %s: %s", state_file, e)

    # Step 4: Cancel pending jobs (unless keep_jobs)
    if not keep_jobs:
        try:
            from jobs.store import get_store
            from guild.job_types import JobStatus

            store = get_store()

            for status in (JobStatus.PENDING, JobStatus.CLAIMED, JobStatus.RUNNING):
                jobs = store.list_jobs(status=status, limit=1000)
                for job in jobs:
                    try:
                        store.cancel(job.job_id, actor="reset")
                        jobs_cancelled += 1
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Failed to cancel job %s: %s", getattr(job, "job_id", job), e)
        except Exception as e:  # noqa: BLE001
            # Keep reset best-effort; logging is enough here.
            logger.warning("Could not clear jobs during reset: %s", e)

    return ResetResult(
        daemons_stopped=daemons_stopped,
        pids_cleared=pids_cleared,
        state_files_cleared=state_files_cleared,
        jobs_cancelled=jobs_cancelled,
    )
