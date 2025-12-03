# core/reset.py
"""
Multi-Level Reset System for the Realm of Training.

Reset Levels:
    CAMPAIGN - Reset ONE campaign (keep hero, delete checkpoints, reset curriculum)
    HERO     - Reset a hero (all their campaigns)
    FULL     - Reset everything (all campaigns, queue, state)
    DEEP     - Nuclear option (also reset base evals, passives, DBs)

Usage:
    python3 core/reset.py --level campaign --target gou-qwen3-4b/campaign-001
    python3 core/reset.py --level hero --target gou-qwen3-4b
    python3 core/reset.py --level full
    python3 core/reset.py --level deep --confirm

    # Dry run to see what would be deleted:
    python3 core/reset.py --level full --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResetLevel(Enum):
    """Reset depth levels."""
    CAMPAIGN = "campaign"  # Reset ONE campaign
    HERO = "hero"          # Reset all campaigns for a hero
    FULL = "full"          # Reset everything
    DEEP = "deep"          # Nuclear - also reset base evals, DBs


@dataclass
class ResetReport:
    """Detailed report of what was (or would be) reset."""
    level: ResetLevel
    target: Optional[str] = None
    dry_run: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    # Counts
    daemons_stopped: int = 0
    pids_cleared: int = 0
    checkpoints_deleted: int = 0
    checkpoint_bytes_freed: int = 0
    state_files_cleared: int = 0
    symlinks_cleared: int = 0
    jobs_cancelled: int = 0
    queue_files_cleared: int = 0
    dbs_cleared: int = 0
    archived_campaigns_purged: int = 0

    # Details
    campaigns_reset: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON."""
        return {
            "level": self.level.value,
            "target": self.target,
            "dry_run": self.dry_run,
            "timestamp": self.timestamp.isoformat(),
            "daemons_stopped": self.daemons_stopped,
            "pids_cleared": self.pids_cleared,
            "checkpoints_deleted": self.checkpoints_deleted,
            "checkpoint_bytes_freed": self.checkpoint_bytes_freed,
            "checkpoint_gb_freed": round(self.checkpoint_bytes_freed / (1024**3), 2),
            "state_files_cleared": self.state_files_cleared,
            "symlinks_cleared": self.symlinks_cleared,
            "jobs_cancelled": self.jobs_cancelled,
            "queue_files_cleared": self.queue_files_cleared,
            "dbs_cleared": self.dbs_cleared,
            "archived_campaigns_purged": self.archived_campaigns_purged,
            "campaigns_reset": self.campaigns_reset,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Reset Level: {self.level.value.upper()}",
            f"Target: {self.target or 'all'}",
            f"Dry Run: {self.dry_run}",
            "",
        ]

        if self.daemons_stopped:
            lines.append(f"  Daemons stopped: {self.daemons_stopped}")
        if self.pids_cleared:
            lines.append(f"  PID files cleared: {self.pids_cleared}")
        if self.checkpoints_deleted:
            gb = self.checkpoint_bytes_freed / (1024**3)
            lines.append(f"  Checkpoints deleted: {self.checkpoints_deleted} ({gb:.2f} GB)")
        if self.state_files_cleared:
            lines.append(f"  State files cleared: {self.state_files_cleared}")
        if self.symlinks_cleared:
            lines.append(f"  Symlinks cleared: {self.symlinks_cleared}")
        if self.jobs_cancelled:
            lines.append(f"  Jobs cancelled: {self.jobs_cancelled}")
        if self.queue_files_cleared:
            lines.append(f"  Queue files cleared: {self.queue_files_cleared}")
        if self.dbs_cleared:
            lines.append(f"  Databases cleared: {self.dbs_cleared}")
        if self.archived_campaigns_purged:
            lines.append(f"  Archived campaigns purged: {self.archived_campaigns_purged}")
        if self.campaigns_reset:
            lines.append(f"  Campaigns reset: {', '.join(self.campaigns_reset)}")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")

        return "\n".join(lines)


class ResetManager:
    """
    Manages multi-level resets for the Realm.

    Usage:
        manager = ResetManager()
        report = manager.reset(ResetLevel.CAMPAIGN, "gou-qwen3-4b/campaign-001")
        print(report.summary())
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)

    def reset(
        self,
        level: ResetLevel,
        target: Optional[str] = None,
        dry_run: bool = False,
        confirm_deep: bool = False,
        purge_archive: bool = False,
    ) -> ResetReport:
        """
        Perform a reset at the specified level.

        Args:
            level: ResetLevel enum value
            target: For CAMPAIGN/HERO levels, the target to reset
            dry_run: If True, don't actually delete anything
            confirm_deep: Required True for DEEP level (safety)
            purge_archive: If True, also delete archived campaigns (FULL/DEEP only)

        Returns:
            ResetReport with details of what was reset
        """
        report = ResetReport(level=level, target=target, dry_run=dry_run)

        if level == ResetLevel.DEEP and not confirm_deep and not dry_run:
            report.errors.append("DEEP reset requires confirm_deep=True (use dry_run=True to preview)")
            return report

        try:
            if level == ResetLevel.CAMPAIGN:
                self._reset_campaign(target, report, dry_run)
            elif level == ResetLevel.HERO:
                self._reset_hero(target, report, dry_run)
            elif level == ResetLevel.FULL:
                self._reset_full(report, dry_run, purge_archive=purge_archive)
            elif level == ResetLevel.DEEP:
                self._reset_deep(report, dry_run, purge_archive=purge_archive)
        except Exception as e:
            report.errors.append(f"Reset failed: {e}")
            logger.exception("Reset failed")

        return report

    # --- Level: CAMPAIGN ---

    def _reset_campaign(
        self,
        target: Optional[str],
        report: ResetReport,
        dry_run: bool,
    ):
        """
        Reset a single campaign.

        Target format: "hero_id/campaign_id" (e.g., "gou-qwen3-4b/campaign-001")

        This will:
        - Delete all checkpoints in the campaign
        - Reset curriculum state to level 1
        - Clear eval history
        - Keep campaign.json and config.json
        - Keep hero config
        """
        if not target:
            report.errors.append("CAMPAIGN reset requires target (hero_id/campaign_id)")
            return

        parts = target.split("/")
        if len(parts) != 2:
            report.errors.append(f"Invalid target format: {target} (expected hero_id/campaign_id)")
            return

        hero_id, campaign_id = parts
        campaign_dir = self.base_dir / "campaigns" / hero_id / campaign_id

        if not campaign_dir.exists():
            report.errors.append(f"Campaign not found: {campaign_dir}")
            return

        report.campaigns_reset.append(target)

        # Delete checkpoints
        checkpoints_dir = campaign_dir / "checkpoints"
        if checkpoints_dir.exists():
            for item in checkpoints_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    size = self._dir_size(item)
                    report.checkpoints_deleted += 1
                    report.checkpoint_bytes_freed += size
                    report.files_deleted.append(str(item))
                    if not dry_run:
                        shutil.rmtree(item)

        # Reset curriculum state
        curriculum_file = campaign_dir / "status" / "curriculum_state.json"
        if curriculum_file.exists():
            report.state_files_cleared += 1
            if not dry_run:
                self._reset_curriculum_file(curriculum_file)

        # Clear eval history
        eval_history_file = campaign_dir / "status" / "eval_results_history.json"
        if eval_history_file.exists():
            report.state_files_cleared += 1
            if not dry_run:
                eval_history_file.write_text('{"results": []}')

        # Clear evaluation ledger
        eval_ledger_file = campaign_dir / "status" / "evaluation_ledger.json"
        if eval_ledger_file.exists():
            report.state_files_cleared += 1
            if not dry_run:
                eval_ledger_file.write_text('{"entries": []}')

        # Clear training status
        training_status_file = campaign_dir / "status" / "training_status.json"
        if training_status_file.exists():
            report.state_files_cleared += 1
            if not dry_run:
                training_status_file.unlink()

        # Update campaign.json to reset step counter
        campaign_file = campaign_dir / "campaign.json"
        if campaign_file.exists() and not dry_run:
            try:
                data = json.loads(campaign_file.read_text())
                data["current_step"] = 0
                data["total_steps_trained"] = 0
                data["reset_at"] = datetime.now().isoformat()
                campaign_file.write_text(json.dumps(data, indent=2))
            except Exception as e:
                report.errors.append(f"Failed to update campaign.json: {e}")

    # --- Level: HERO ---

    def _reset_hero(
        self,
        target: Optional[str],
        report: ResetReport,
        dry_run: bool,
    ):
        """
        Reset all campaigns for a hero.

        Target format: "hero_id" (e.g., "gou-qwen3-4b")
        """
        if not target:
            report.errors.append("HERO reset requires target (hero_id)")
            return

        hero_dir = self.base_dir / "campaigns" / target
        if not hero_dir.exists():
            report.errors.append(f"Hero campaigns not found: {hero_dir}")
            return

        # Find all campaigns for this hero
        for campaign_dir in hero_dir.iterdir():
            if campaign_dir.is_dir() and campaign_dir.name.startswith("campaign-"):
                campaign_target = f"{target}/{campaign_dir.name}"
                self._reset_campaign(campaign_target, report, dry_run)

    # --- Level: FULL ---

    def _reset_full(self, report: ResetReport, dry_run: bool, purge_archive: bool = False):
        """
        Full reset - everything except base models and hero configs.

        This will:
        - Stop all daemons
        - Clear all PID files
        - Clear symlinks in status/
        - Clear queue (active dirs only)
        - Reset all campaigns
        - Reset global curriculum state
        - Cancel all jobs
        - Clear status files
        - Clear control files
        - Optionally purge archived campaigns (if purge_archive=True)
        """
        # Stop daemons first
        report.daemons_stopped = self._stop_daemons(dry_run)

        # Clear PID files
        report.pids_cleared = self._clear_pids(dry_run)

        # Clear symlinks in status/ (these point to campaign-specific files)
        report.symlinks_cleared = self._clear_status_symlinks(dry_run)

        # Clear queue (active dirs only - processing, high, normal, low)
        report.queue_files_cleared = self._clear_queue(dry_run, full_clear=False)

        # Reset all campaigns
        campaigns_dir = self.base_dir / "campaigns"
        if campaigns_dir.exists():
            for hero_dir in campaigns_dir.iterdir():
                if hero_dir.is_dir() and hero_dir.name != "archive":
                    for campaign_dir in hero_dir.iterdir():
                        if campaign_dir.is_dir() and campaign_dir.name.startswith("campaign-"):
                            campaign_target = f"{hero_dir.name}/{campaign_dir.name}"
                            self._reset_campaign(campaign_target, report, dry_run)

        # Reset global curriculum state
        global_curriculum = self.base_dir / "data_manager" / "curriculum_state.json"
        if global_curriculum.exists() and not global_curriculum.is_symlink():
            report.state_files_cleared += 1
            if not dry_run:
                self._reset_curriculum_file(global_curriculum)

        # Clear status files (non-symlinks, runtime state)
        status_files = [
            "status/events.jsonl",
            "status/task_state.json",
            "status/task_master.json",
            "status/eval_queue.json",
            "status/passive_queue.json",
            "status/scheduler_state.json",
            "status/training_daemon.json",
            "status/summoner.json",
            "status/groundskeeper_state.json",
        ]
        for relpath in status_files:
            path = self.base_dir / relpath
            if path.exists() and not path.is_symlink():
                report.state_files_cleared += 1
                report.files_deleted.append(str(path))
                if not dry_run:
                    path.unlink()

        # Clear control files (runtime state)
        control_files = [
            "control/state.json",
            "control/momentum.json",
            "control/realm_state.json",
            "control/train_request.json",
        ]
        for relpath in control_files:
            path = self.base_dir / relpath
            if path.exists():
                report.state_files_cleared += 1
                report.files_deleted.append(str(path))
                if not dry_run:
                    path.unlink()

        # Cancel jobs
        report.jobs_cancelled = self._cancel_jobs(dry_run)

        # Clear active campaign pointer (keep file, clear content)
        active_campaign = self.base_dir / "control" / "active_campaign.json"
        if active_campaign.exists():
            report.state_files_cleared += 1
            if not dry_run:
                active_campaign.write_text('{}')

        # Reset RealmState training data (clear stale training state)
        self._reset_realm_state_training(report, dry_run)

        # Reset realm_store.json (file-based fallback)
        realm_store = self.base_dir / "status" / "realm_store.json"
        if realm_store.exists():
            report.state_files_cleared += 1
            if not dry_run:
                self._reset_realm_store_file(realm_store)

        # Purge archive if requested
        if purge_archive:
            self._purge_archive(report, dry_run)

    # --- Level: DEEP ---

    def _reset_deep(self, report: ResetReport, dry_run: bool, purge_archive: bool = False):
        """
        Nuclear reset - everything including base evals and databases.

        This will:
        - Everything in FULL
        - Clear ALL queue directories (including history)
        - Clear passives ledger
        - Clear job store database
        - Clear realm state database
        - Clear VaultKeeper catalog
        - Clear checkpoint ledger
        - Optionally purge archived campaigns (if purge_archive=True)
        """
        # First do full reset (which will handle archive if purge_archive is set)
        self._reset_full(report, dry_run, purge_archive=purge_archive)

        # Clear ALL queue files (including history - recently_completed, failed, etc.)
        # _reset_full only cleared active queues, now clear the rest
        additional_queue_cleared = self._clear_queue(dry_run, full_clear=True)
        # Subtract what was already counted in full reset to avoid double-counting
        report.queue_files_cleared = additional_queue_cleared

        # Clear passives ledger
        passives_ledger = self.base_dir / "status" / "passives_ledger.json"
        if passives_ledger.exists():
            report.state_files_cleared += 1
            if not dry_run:
                passives_ledger.write_text('{"passives": {}}')

        # Clear checkpoint ledger
        checkpoint_ledger = self.base_dir / "status" / "checkpoint_ledger.json"
        if checkpoint_ledger.exists():
            report.state_files_cleared += 1
            if not dry_run:
                checkpoint_ledger.write_text('{"entries": {}, "by_step": {}}')

        # Clear databases
        db_files = [
            "jobs/job_store.db",
            "status/realm_state.db",
            "vault/catalog.db",
        ]
        for relpath in db_files:
            path = self.base_dir / relpath
            if path.exists():
                report.dbs_cleared += 1
                report.files_deleted.append(str(path))
                if not dry_run:
                    path.unlink()

        # Clear all status/*.json files (nuclear)
        status_dir = self.base_dir / "status"
        if status_dir.exists():
            for f in status_dir.glob("*.json"):
                if str(f) not in report.files_deleted:
                    report.state_files_cleared += 1
                    report.files_deleted.append(str(f))
                    if not dry_run:
                        f.unlink()

    # --- Helper Methods ---

    def _stop_daemons(self, dry_run: bool) -> int:
        """Stop all running daemons."""
        count = 0
        pids_dir = self.base_dir / ".pids"
        if not pids_dir.exists():
            return 0

        for pid_file in pids_dir.glob("*.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                if not dry_run:
                    os.kill(pid, signal.SIGTERM)
                count += 1
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                pass

        return count

    def _clear_pids(self, dry_run: bool) -> int:
        """Clear all PID files."""
        count = 0
        pids_dir = self.base_dir / ".pids"
        if not pids_dir.exists():
            return 0

        for pid_file in pids_dir.glob("*.pid"):
            count += 1
            if not dry_run:
                try:
                    pid_file.unlink()
                except OSError:
                    pass

        return count

    def _clear_status_symlinks(self, dry_run: bool) -> int:
        """Clear symlinks in status/ directory.

        These symlinks point to campaign-specific files and should be
        recreated when a new campaign is activated.
        """
        count = 0
        status_dir = self.base_dir / "status"
        if not status_dir.exists():
            return 0

        # Known symlinks that point to campaign data
        symlink_names = [
            "checkpoint_ledger.json",
            "curriculum_state.json",
            "eval_results_history.json",
            "evaluation_ledger.json",
            "training_status.json",
        ]

        for name in symlink_names:
            path = status_dir / name
            if path.is_symlink():
                count += 1
                if not dry_run:
                    try:
                        path.unlink()
                    except OSError:
                        pass

        return count

    def _clear_queue(self, dry_run: bool, full_clear: bool = False) -> int:
        """Clear queue files.

        Args:
            dry_run: If True, don't actually delete
            full_clear: If True, clear ALL queue dirs including history.
                        If False, only clear active queue dirs.
        """
        count = 0
        queue_dir = self.base_dir / "queue"
        if not queue_dir.exists():
            return 0

        # Active queue directories (always cleared)
        active_dirs = ["high", "normal", "low", "processing"]

        # History/archive directories (only cleared on full_clear)
        history_dirs = [
            "recently_completed", "failed", "rejected",
            "deferred_old_format", "deferred_stale", "deferred_binary",
            "corrections", "unvalidated"
        ]

        dirs_to_clear = active_dirs + (history_dirs if full_clear else [])

        for subdir in dirs_to_clear:
            subdir_path = queue_dir / subdir
            if subdir_path.exists():
                for f in subdir_path.glob("*.jsonl"):
                    count += 1
                    if not dry_run:
                        try:
                            f.unlink()
                        except OSError:
                            pass

        return count

    def _purge_archive(self, report: ResetReport, dry_run: bool) -> None:
        """
        Purge all archived campaigns.

        This permanently deletes the campaigns/archive/ directory and all its contents.
        """
        archive_dir = self.base_dir / "campaigns" / "archive"
        if not archive_dir.exists():
            return

        # Count archived campaigns before deletion
        for hero_dir in archive_dir.iterdir():
            if hero_dir.is_dir():
                for campaign_dir in hero_dir.iterdir():
                    if campaign_dir.is_dir() and campaign_dir.name.startswith("campaign-"):
                        report.archived_campaigns_purged += 1
                        # Count checkpoints for byte tracking
                        checkpoints_dir = campaign_dir / "checkpoints"
                        if checkpoints_dir.exists():
                            for item in checkpoints_dir.iterdir():
                                if item.is_dir() and item.name.startswith("checkpoint-"):
                                    size = self._dir_size(item)
                                    report.checkpoints_deleted += 1
                                    report.checkpoint_bytes_freed += size
                        report.files_deleted.append(str(campaign_dir))

        if not dry_run:
            try:
                shutil.rmtree(archive_dir)
            except OSError as e:
                report.errors.append(f"Failed to purge archive: {e}")

    def _cancel_jobs(self, dry_run: bool) -> int:
        """Cancel all pending/running jobs."""
        if dry_run:
            # Just count them
            try:
                from jobs.store import get_store
                from guild.job_types import JobStatus
                store = get_store()
                count = 0
                for status in (JobStatus.PENDING, JobStatus.CLAIMED, JobStatus.RUNNING):
                    jobs = store.list_jobs(status=status, limit=1000)
                    count += len(jobs)
                return count
            except Exception:
                return 0

        # Actually cancel
        try:
            from jobs.store import get_store
            from guild.job_types import JobStatus
            store = get_store()
            count = 0
            for status in (JobStatus.PENDING, JobStatus.CLAIMED, JobStatus.RUNNING):
                jobs = store.list_jobs(status=status, limit=1000)
                for job in jobs:
                    try:
                        store.cancel(job.job_id, actor="reset")
                        count += 1
                    except Exception:
                        pass
            return count
        except Exception:
            return 0

    def _reset_curriculum_file(self, path: Path):
        """Reset a curriculum state file to initial state."""
        initial = {
            "skills": {
                "sy": {
                    "current_level": 1,
                    "training_level": 1,
                    "mastered_level": 0,
                    "accuracy_history": [],
                    "progression_history": [],
                },
                "bin": {
                    "current_level": 1,
                    "training_level": 1,
                    "mastered_level": 0,
                    "accuracy_history": [],
                    "progression_history": [],
                },
            },
            "active_skill": "sy",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "skill_rotation": {
                "enabled": True,
                "skills": ["sy", "bin"],
                "index": 0,
            },
        }
        path.write_text(json.dumps(initial, indent=2))

    def _dir_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total

    def _reset_realm_state_training(self, report: ResetReport, dry_run: bool):
        """Reset the training section in RealmState database."""
        db_path = self.base_dir / "data" / "realm_state.db"
        if not db_path.exists():
            return

        report.state_files_cleared += 1
        if dry_run:
            return

        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path), timeout=10)
            cur = conn.cursor()

            # Reset training state to empty/idle
            reset_data = json.dumps({
                "status": "idle",
                "step": 0,
                "total_steps": 0,
                "loss": 0,
                "learning_rate": 0,
                "file": None,
                "speed": 0,
                "eta_seconds": 0,
                "updated_at": None
            })
            cur.execute('UPDATE state SET data = ? WHERE section = ?', (reset_data, 'training'))

            # Also clear workers
            cur.execute('DELETE FROM workers')

            # Clear events
            cur.execute('DELETE FROM events')

            conn.commit()
            conn.close()
        except Exception as e:
            report.errors.append(f"Failed to reset RealmState DB: {e}")

    def _reset_realm_store_file(self, path: Path):
        """Reset the realm_store.json file to clean state."""
        try:
            data = json.loads(path.read_text())

            # Reset training state
            if "state" in data:
                data["state"]["training"] = {
                    "status": "idle",
                    "step": 0,
                    "total_steps": 0,
                    "loss": 0,
                    "learning_rate": 0,
                    "file": None,
                    "speed": 0,
                    "eta_seconds": 0,
                    "updated_at": None
                }
                # Clear workers
                data["state"]["workers"] = {}

            # Clear events
            data["events"] = []

            path.write_text(json.dumps(data, indent=2))
        except Exception:
            # If we can't parse it, just write a fresh structure
            fresh = {
                "state": {
                    "training": {
                        "status": "idle",
                        "step": 0,
                        "total_steps": 0,
                        "loss": 0,
                        "learning_rate": 0,
                        "file": None,
                        "speed": 0,
                        "eta_seconds": 0,
                        "updated_at": None
                    },
                    "workers": {},
                    "queue": {"depth": 0, "high_priority": 0, "normal_priority": 0, "low_priority": 0},
                    "mode": "idle",
                },
                "events": []
            }
            path.write_text(json.dumps(fresh, indent=2))


# --- Legacy Compatibility ---

@dataclass
class ResetResult:
    """
    Legacy result class for backward compatibility.
    Use ResetReport for new code.
    """
    daemons_stopped: List[int]
    pids_cleared: List[Path]
    state_files_cleared: List[Path]
    jobs_cancelled: int

    def as_counts(self) -> dict:
        return {
            "daemons_stopped": len(self.daemons_stopped),
            "pids_cleared": len(self.pids_cleared),
            "state_files_cleared": len(self.state_files_cleared),
            "jobs_cancelled": self.jobs_cancelled,
        }


def reset_environment(*, keep_jobs: bool = False, base_dir: Path | None = None) -> ResetResult:
    """
    Legacy function for backward compatibility.

    This is equivalent to ResetLevel.FULL but returns the old format.
    For new code, use ResetManager.reset(ResetLevel.FULL).
    """
    if base_dir is None:
        from core.paths import get_base_dir
        base_dir = get_base_dir()

    manager = ResetManager(base_dir)
    report = manager.reset(ResetLevel.FULL, dry_run=False)

    # Convert to legacy format
    return ResetResult(
        daemons_stopped=list(range(report.daemons_stopped)),
        pids_cleared=[Path(f) for f in report.files_deleted if ".pids" in f],
        state_files_cleared=[Path(f) for f in report.files_deleted if "status" in f or "control" in f],
        jobs_cancelled=report.jobs_cancelled,
    )


# --- CLI ---

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Level Reset System for the Realm of Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reset a single campaign
  python3 core/reset.py --level campaign --target gou-qwen3-4b/campaign-001

  # Reset all campaigns for a hero
  python3 core/reset.py --level hero --target gou-qwen3-4b

  # Full reset (everything except base models, preserves archive)
  python3 core/reset.py --level full

  # Full reset + purge archive (total clean slate)
  python3 core/reset.py --level full --purge-archive

  # Nuclear reset (requires --confirm)
  python3 core/reset.py --level deep --confirm

  # Nuclear + purge archive (absolutely everything)
  python3 core/reset.py --level deep --confirm --purge-archive

  # Dry run to see what would be deleted
  python3 core/reset.py --level full --dry-run
  python3 core/reset.py --level full --purge-archive --dry-run
        """,
    )

    parser.add_argument(
        "--level", "-l",
        choices=["campaign", "hero", "full", "deep"],
        required=True,
        help="Reset level",
    )
    parser.add_argument(
        "--target", "-t",
        help="Target for campaign/hero reset (e.g., gou-qwen3-4b/campaign-001)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for DEEP level reset",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--purge-archive",
        action="store_true",
        help="Also delete archived campaigns (FULL/DEEP only)",
    )

    args = parser.parse_args()

    level = ResetLevel(args.level)

    # Validate target requirements
    if level in (ResetLevel.CAMPAIGN, ResetLevel.HERO) and not args.target:
        parser.error(f"--target is required for {level.value} level")

    # Confirm for DEEP
    if level == ResetLevel.DEEP and not args.confirm and not args.dry_run:
        print("WARNING: DEEP reset will delete:")
        print("  - All checkpoints")
        print("  - All databases (job store, vault catalog, realm state)")
        print("  - All status files")
        print("  - All eval history and passives")
        print()
        print("This cannot be undone!")
        print()
        print("Use --dry-run to see what would be deleted.")
        print("Use --confirm to proceed with the reset.")
        return 1

    # Warn if purge_archive is used with CAMPAIGN/HERO level
    if args.purge_archive and level in (ResetLevel.CAMPAIGN, ResetLevel.HERO):
        print("Note: --purge-archive only applies to FULL and DEEP levels")

    manager = ResetManager()
    report = manager.reset(
        level=level,
        target=args.target,
        dry_run=args.dry_run,
        confirm_deep=args.confirm,
        purge_archive=args.purge_archive,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"{prefix}Reset Report")
        print("=" * 40)
        print(report.summary())

        if report.errors:
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
