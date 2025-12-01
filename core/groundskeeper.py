#!/usr/bin/env python3
"""
Groundskeeper - Unified cleanup daemon for all leaky systems

The Groundskeeper handles ALL cleanup tasks in one place:
- JSONL file rotation (events.jsonl, job_history.jsonl, data_file_impact.jsonl)
- Queue cleanup (recently_completed/ older than 7 days)
- Battle log cleanup (entries older than 7 days)
- Log file rotation (delete logs older than 30 days)
- Stale PID cleanup (remove PIDs for dead processes)
- SQLite VACUUM (monthly to reclaim space)
- Worker cleanup (remove long-offline workers)
- Job events cleanup (remove old job_events records)

Usage:
    # Single cleanup run
    python3 core/groundskeeper.py

    # Show what would be cleaned (dry run)
    python3 core/groundskeeper.py --dry-run

    # Run as daemon (hourly checks)
    python3 core/groundskeeper.py --daemon

    # Specific tasks only
    python3 core/groundskeeper.py --task jsonl
    python3 core/groundskeeper.py --task queue
    python3 core/groundskeeper.py --task battle_log
    python3 core/groundskeeper.py --task logs
    python3 core/groundskeeper.py --task pids
    python3 core/groundskeeper.py --task vacuum

Integration with Weaver:
    The Weaver can call Groundskeeper.sweep() periodically.
"""

import errno
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CleanupPolicy:
    """
    Retention policy for a resource type.

    Not all fields apply to all task types:
    - jsonl: uses max_age_days, max_size_mb, max_lines, keep_fraction, keep_lines_cap
    - queue: uses max_age_days, max_files (backstop for file count limit)
    - battle_log: uses max_age_days only
    - logs: uses max_age_days only (max_size_mb reserved for future total footprint bound)
    - pids: ignores all - cleanup is purely liveness-based (dead process = delete)
    - vacuum: uses max_age_days as interval between VACUUMs
    - workers: uses max_age_days
    - job_events: uses max_age_days
    """
    max_age_days: int = 7
    max_size_mb: float = 50.0
    max_lines: int = 10000
    max_files: int = 100  # Reserved for future use (queue file count cap)
    keep_fraction: float = 0.5  # Fraction of max_lines to keep after rotation
    keep_lines_cap: int = 25000  # Hard cap on kept lines regardless of fraction
    enabled: bool = True


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    task: str
    items_cleaned: int = 0
    bytes_freed: int = 0
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def mb_freed(self) -> float:
        return self.bytes_freed / (1024 * 1024)

    def __str__(self) -> str:
        mode = "[DRY RUN] " if self.dry_run else ""
        if self.errors:
            return f"{mode}{self.task}: {self.items_cleaned} items, {self.mb_freed:.2f}MB, {len(self.errors)} errors"
        return f"{mode}{self.task}: {self.items_cleaned} items, {self.mb_freed:.2f}MB freed"


class Groundskeeper:
    """
    The Groundskeeper - Keeps the realm tidy

    Handles all cleanup tasks:
    - JSONL rotation (events, job_history, data_file_impact)
    - Queue file cleanup (recently_completed)
    - Battle log table cleanup
    - Log file rotation
    - Stale PID cleanup
    - SQLite VACUUM
    """

    DEFAULT_POLICIES = {
        "jsonl": CleanupPolicy(max_age_days=7, max_size_mb=50.0, max_lines=50000),
        "queue": CleanupPolicy(max_age_days=7, max_files=1000),
        "battle_log": CleanupPolicy(max_age_days=7),
        "logs": CleanupPolicy(max_age_days=30, max_size_mb=500.0),
        "pids": CleanupPolicy(max_age_days=1),  # Always clean stale PIDs
        "vacuum": CleanupPolicy(max_age_days=30),  # Monthly VACUUM
        "workers": CleanupPolicy(max_age_days=30),  # Workers offline > 30 days
        "job_events": CleanupPolicy(max_age_days=14),  # Job events older than 14 days
    }

    def __init__(self, base_dir: Optional[Path] = None, policies: Optional[Dict[str, CleanupPolicy]] = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.policies = {**self.DEFAULT_POLICIES, **(policies or {})}
        self.last_vacuum = self._load_last_vacuum_time()
        self.running = False

    def _load_last_vacuum_time(self) -> datetime:
        """Load last vacuum time from state file."""
        state_file = self.base_dir / "status" / "groundskeeper_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                return datetime.fromisoformat(data.get("last_vacuum", "2000-01-01"))
            except Exception as e:
                logger.warning(f"Failed to load groundskeeper state: {e}")
        return datetime(2000, 1, 1)

    def _save_state(self):
        """Save groundskeeper state."""
        state_file = self.base_dir / "status" / "groundskeeper_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({
            "last_vacuum": self.last_vacuum.isoformat(),
            "last_sweep": datetime.now().isoformat(),
        }))

    # ========== JSONL Rotation ==========

    def _rotate_jsonl_file(self, file_path: Path, policy: CleanupPolicy, dry_run: bool = False) -> CleanupResult:
        """
        Rotate a JSONL file if it exceeds size/line limits.

        Strategy:
        1. If file > max_size_mb or > max_lines, rotate
        2. Keep current data in main file
        3. Archive old data to {filename}.{date}.jsonl
        4. Delete archives older than max_age_days
        """
        result = CleanupResult(task=f"jsonl:{file_path.name}", dry_run=dry_run)

        if not file_path.exists():
            return result

        try:
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Count lines (approximate for large files)
            line_count = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f:
                    line_count += 1

            needs_rotation = file_size_mb > policy.max_size_mb or line_count > policy.max_lines

            if needs_rotation:
                logger.info(f"Rotating {file_path.name}: {file_size_mb:.2f}MB, {line_count} lines")

                if not dry_run:
                    # Read all lines
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    # Keep last N lines in main file (configurable via policy)
                    keep_lines = min(
                        int(policy.max_lines * policy.keep_fraction),
                        policy.keep_lines_cap
                    )

                    if len(lines) > keep_lines:
                        # Archive old lines
                        archive_name = f"{file_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                        archive_path = file_path.parent / archive_name

                        old_lines = lines[:-keep_lines]
                        new_lines = lines[-keep_lines:]

                        # Write archive
                        with open(archive_path, 'w', encoding='utf-8') as f:
                            f.writelines(old_lines)

                        # Write new main file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)

                        result.items_cleaned = len(old_lines)
                        result.bytes_freed = len(''.join(old_lines).encode('utf-8'))
                        logger.info(f"Archived {len(old_lines)} lines to {archive_name}")
                else:
                    keep_lines = min(
                        int(policy.max_lines * policy.keep_fraction),
                        policy.keep_lines_cap
                    )
                    result.items_cleaned = max(0, line_count - keep_lines)
                    result.bytes_freed = int(file_size * (1 - policy.keep_fraction))

            # Clean old archives
            cutoff = datetime.now() - timedelta(days=policy.max_age_days)
            archive_pattern = f"{file_path.stem}.*.jsonl"

            for archive in file_path.parent.glob(archive_pattern):
                if archive == file_path:
                    continue
                try:
                    mtime = datetime.fromtimestamp(archive.stat().st_mtime)
                    if mtime < cutoff:
                        size = archive.stat().st_size
                        if not dry_run:
                            archive.unlink()
                        result.items_cleaned += 1
                        result.bytes_freed += size
                        logger.info(f"Deleted old archive: {archive.name}")
                except Exception as e:
                    result.errors.append(f"Failed to delete {archive}: {e}")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error rotating {file_path}: {e}")

        return result

    def rotate_jsonl_files(self, dry_run: bool = False) -> List[CleanupResult]:
        """Rotate all JSONL files that need it."""
        results = []
        policy = self.policies["jsonl"]

        if not policy.enabled:
            return results

        # Files to rotate
        jsonl_files = [
            self.base_dir / "status" / "events.jsonl",
            self.base_dir / "status" / "job_history.jsonl",
            self.base_dir / "status" / "data_file_impact.jsonl",
        ]

        for file_path in jsonl_files:
            result = self._rotate_jsonl_file(file_path, policy, dry_run)
            if result.items_cleaned > 0 or result.errors:
                results.append(result)

        return results

    # ========== Queue Cleanup ==========

    def cleanup_queue(self, dry_run: bool = False) -> CleanupResult:
        """
        Clean up queue/recently_completed/ directory.

        Strategy:
        1. Delete files older than max_age_days
        2. If still over max_files, delete oldest until under limit (backstop)
        """
        result = CleanupResult(task="queue:recently_completed", dry_run=dry_run)
        policy = self.policies["queue"]

        if not policy.enabled:
            return result

        completed_dir = self.base_dir / "queue" / "recently_completed"

        if not completed_dir.exists():
            return result

        cutoff = datetime.now() - timedelta(days=policy.max_age_days)

        try:
            files = list(completed_dir.glob("*.jsonl"))

            # Phase 1: Age-based cleanup
            for file_path in files:
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff:
                        size = file_path.stat().st_size
                        if not dry_run:
                            file_path.unlink()
                        result.items_cleaned += 1
                        result.bytes_freed += size
                        logger.debug(f"Deleted old queue file: {file_path.name}")
                except Exception as e:
                    result.errors.append(f"Failed to delete {file_path.name}: {e}")

            # Phase 2: max_files backstop (in case of clock issues or burst of files)
            remaining_files = [f for f in completed_dir.glob("*.jsonl") if f.exists()]
            if len(remaining_files) > policy.max_files:
                # Sort by mtime (oldest first) and delete extras
                remaining_files.sort(key=lambda p: p.stat().st_mtime)
                excess = remaining_files[:-policy.max_files]  # Keep newest max_files
                for file_path in excess:
                    try:
                        size = file_path.stat().st_size
                        if not dry_run:
                            file_path.unlink()
                        result.items_cleaned += 1
                        result.bytes_freed += size
                        logger.debug(f"Deleted excess queue file: {file_path.name} (max_files backstop)")
                    except Exception as e:
                        result.errors.append(f"Failed to delete {file_path.name}: {e}")

            if result.items_cleaned > 0:
                logger.info(f"Cleaned {result.items_cleaned} queue files ({result.mb_freed:.2f}MB)")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning queue: {e}")

        return result

    # ========== Battle Log Cleanup ==========

    def cleanup_battle_log(self, dry_run: bool = False) -> CleanupResult:
        """Clean up old battle log entries."""
        result = CleanupResult(task="battle_log", dry_run=dry_run)
        policy = self.policies["battle_log"]

        if not policy.enabled:
            return result

        db_path = self.base_dir / "vault" / "jobs.db"

        if not db_path.exists():
            return result

        try:
            cutoff = (datetime.utcnow() - timedelta(days=policy.max_age_days)).isoformat() + "Z"

            conn = sqlite3.connect(str(db_path))

            if not dry_run:
                cursor = conn.execute(
                    "DELETE FROM battle_log WHERE timestamp < ?",
                    (cutoff,)
                )
                result.items_cleaned = cursor.rowcount
                conn.commit()
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM battle_log WHERE timestamp < ?",
                    (cutoff,)
                )
                result.items_cleaned = cursor.fetchone()[0]

            conn.close()

            if result.items_cleaned > 0:
                logger.info(f"Cleaned {result.items_cleaned} old battle log entries")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning battle log: {e}")

        return result

    # ========== Log File Cleanup ==========

    def cleanup_logs(self, dry_run: bool = False) -> CleanupResult:
        """Clean up old log files."""
        result = CleanupResult(task="logs", dry_run=dry_run)
        policy = self.policies["logs"]

        if not policy.enabled:
            return result

        logs_dir = self.base_dir / "logs"

        if not logs_dir.exists():
            return result

        cutoff = datetime.now() - timedelta(days=policy.max_age_days)

        try:
            # All log files
            for log_file in logs_dir.glob("*.log"):
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff:
                        size = log_file.stat().st_size
                        if not dry_run:
                            log_file.unlink()
                        result.items_cleaned += 1
                        result.bytes_freed += size
                        logger.debug(f"Deleted old log: {log_file.name}")
                except Exception as e:
                    result.errors.append(f"Failed to delete {log_file.name}: {e}")

            # Also check for dated log files like daemon_20251128.log
            for log_file in logs_dir.glob("*_????????.log"):
                # Skip if already deleted by first loop (patterns can overlap)
                if not log_file.exists():
                    continue
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff:
                        size = log_file.stat().st_size
                        if not dry_run:
                            log_file.unlink()
                        result.items_cleaned += 1
                        result.bytes_freed += size
                except Exception as e:
                    result.errors.append(f"Failed to delete {log_file.name}: {e}")

            if result.items_cleaned > 0:
                logger.info(f"Cleaned {result.items_cleaned} old log files ({result.mb_freed:.2f}MB)")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning logs: {e}")

        return result

    # ========== Stale PID Cleanup ==========

    def cleanup_stale_pids(self, dry_run: bool = False) -> CleanupResult:
        """Clean up PID files for dead processes."""
        result = CleanupResult(task="pids", dry_run=dry_run)
        policy = self.policies["pids"]

        if not policy.enabled:
            return result

        pids_dir = self.base_dir / ".pids"

        if not pids_dir.exists():
            return result

        # Also check root-level .daemon.pid
        pid_files = list(pids_dir.glob("*.pid"))
        root_pid = self.base_dir / ".daemon.pid"
        if root_pid.exists():
            pid_files.append(root_pid)

        for pid_file in pid_files:
            try:
                pid = int(pid_file.read_text().strip())

                # Check if process is alive using os.kill(pid, 0)
                try:
                    os.kill(pid, 0)
                    # Process exists, leave it alone
                except OSError as e:
                    if e.errno == errno.ESRCH:
                        # ESRCH: No such process - definitely dead, clean up
                        if not dry_run:
                            pid_file.unlink()
                        result.items_cleaned += 1
                        logger.info(f"Removed stale PID file: {pid_file.name} (PID {pid} dead)")
                    elif e.errno == errno.EPERM:
                        # EPERM: Process exists but we lack permission - leave it alone
                        logger.debug(f"PID {pid} exists but no permission to signal (leaving {pid_file.name})")
                    else:
                        # Other OS error - log and skip
                        logger.warning(f"Unexpected error checking PID {pid}: {e}")

            except ValueError:
                # Invalid PID in file (not a number), remove it
                if not dry_run:
                    pid_file.unlink()
                result.items_cleaned += 1
                logger.info(f"Removed invalid PID file: {pid_file.name} (not a valid PID)")
            except OSError as e:
                # Can't read file, remove it
                if not dry_run:
                    try:
                        pid_file.unlink()
                    except OSError:
                        pass
                result.items_cleaned += 1
                logger.info(f"Removed unreadable PID file: {pid_file.name}")

        return result

    # ========== SQLite VACUUM ==========

    def vacuum_databases(self, dry_run: bool = False, force: bool = False) -> List[CleanupResult]:
        """
        VACUUM SQLite databases to reclaim space.

        Only runs monthly unless forced.
        """
        results = []
        policy = self.policies["vacuum"]

        if not policy.enabled:
            return results

        # Check if vacuum is due
        days_since_vacuum = (datetime.now() - self.last_vacuum).days

        if not force and days_since_vacuum < policy.max_age_days:
            logger.debug(f"Skipping VACUUM ({days_since_vacuum} days since last, policy is {policy.max_age_days} days)")
            return results

        # Databases to vacuum
        db_files = [
            self.base_dir / "vault" / "jobs.db",
            self.base_dir / "vault" / "catalog.db",
            self.base_dir / "data" / "realm_state.db",
        ]

        for db_path in db_files:
            result = CleanupResult(task=f"vacuum:{db_path.name}", dry_run=dry_run)

            if not db_path.exists():
                continue

            try:
                size_before = db_path.stat().st_size

                if not dry_run:
                    conn = sqlite3.connect(str(db_path))
                    conn.execute("VACUUM")
                    conn.close()

                    size_after = db_path.stat().st_size
                    result.bytes_freed = max(0, size_before - size_after)
                else:
                    # Estimate 10% savings for dry run
                    result.bytes_freed = int(size_before * 0.1)

                result.items_cleaned = 1

                if result.bytes_freed > 0:
                    logger.info(f"VACUUMed {db_path.name}: freed {result.mb_freed:.2f}MB")

            except Exception as e:
                result.errors.append(str(e))
                logger.error(f"Error vacuuming {db_path}: {e}")

            results.append(result)

        if not dry_run:
            self.last_vacuum = datetime.now()
            self._save_state()

        return results

    # ========== Worker Cleanup ==========

    def cleanup_workers(self, dry_run: bool = False) -> CleanupResult:
        """Clean up long-offline workers from job store."""
        result = CleanupResult(task="workers", dry_run=dry_run)
        policy = self.policies["workers"]

        if not policy.enabled:
            return result

        try:
            from jobs.store import SQLiteJobStore

            db_path = self.base_dir / "vault" / "jobs.db"
            if not db_path.exists():
                return result

            store = SQLiteJobStore(str(db_path))

            if not dry_run:
                result.items_cleaned = store.cleanup_old_workers(max_offline_days=policy.max_age_days)
            else:
                # Count would-be deleted
                cutoff = (datetime.now() - timedelta(days=policy.max_age_days)).isoformat()
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM workers WHERE last_heartbeat_at < ?",
                    (cutoff,)
                )
                result.items_cleaned = cursor.fetchone()[0]
                conn.close()

            if result.items_cleaned > 0:
                logger.info(f"Cleaned {result.items_cleaned} old workers")

        except ImportError:
            logger.debug("jobs.store not available, skipping worker cleanup")
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning workers: {e}")

        return result

    # ========== Job Events Cleanup ==========

    def cleanup_job_events(self, dry_run: bool = False) -> CleanupResult:
        """Clean up old job events."""
        result = CleanupResult(task="job_events", dry_run=dry_run)
        policy = self.policies["job_events"]

        if not policy.enabled:
            return result

        try:
            from jobs.store import SQLiteJobStore

            db_path = self.base_dir / "vault" / "jobs.db"
            if not db_path.exists():
                return result

            store = SQLiteJobStore(str(db_path))

            if not dry_run:
                result.items_cleaned = store.cleanup_old_events(max_age_days=policy.max_age_days)
            else:
                cutoff = (datetime.now() - timedelta(days=policy.max_age_days)).isoformat()
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM job_events WHERE timestamp < ?",
                    (cutoff,)
                )
                result.items_cleaned = cursor.fetchone()[0]
                conn.close()

            if result.items_cleaned > 0:
                logger.info(f"Cleaned {result.items_cleaned} old job events")

        except ImportError:
            logger.debug("jobs.store not available, skipping job events cleanup")
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning job events: {e}")

        return result

    # ========== Stale Checkpoint Jobs Cleanup ==========

    def cleanup_stale_checkpoint_jobs(self, dry_run: bool = False) -> CleanupResult:
        """Cancel pending jobs that reference non-existent checkpoints."""
        result = CleanupResult(task="stale_jobs", dry_run=dry_run)

        try:
            from jobs.store import SQLiteJobStore

            db_path = self.base_dir / "vault" / "jobs.db"
            if not db_path.exists():
                return result

            store = SQLiteJobStore(str(db_path))

            if not dry_run:
                result.items_cleaned = store.prune_stale_checkpoint_jobs(self.base_dir)
            else:
                # For dry run, just count pending jobs with checkpoint paths
                from guild.job_types import JobStatus
                pending = store.list_jobs(status=JobStatus.PENDING, limit=1000)
                stale_count = 0
                for job in pending:
                    payload = job.spec.payload or {}
                    ckpt_path = payload.get("checkpoint_path")
                    if ckpt_path:
                        full_path = self.base_dir / ckpt_path
                        if not full_path.exists():
                            stale_count += 1
                result.items_cleaned = stale_count

            if result.items_cleaned > 0:
                logger.info(f"{'Would cancel' if dry_run else 'Cancelled'} {result.items_cleaned} stale checkpoint jobs")

        except ImportError:
            logger.debug("jobs.store not available, skipping stale jobs cleanup")
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error cleaning stale jobs: {e}")

        return result

    # ========== Main Sweep ==========

    def sweep(self, dry_run: bool = False, tasks: Optional[List[str]] = None) -> Dict[str, CleanupResult]:
        """
        Run all cleanup tasks.

        Args:
            dry_run: If True, show what would be cleaned without actually cleaning
            tasks: Optional list of specific tasks to run (default: all)

        Returns:
            Dict mapping task name to CleanupResult
        """
        all_results = {}

        task_map = {
            "jsonl": self.rotate_jsonl_files,
            "queue": self.cleanup_queue,
            "battle_log": self.cleanup_battle_log,
            "logs": self.cleanup_logs,
            "pids": self.cleanup_stale_pids,
            "vacuum": self.vacuum_databases,
            "workers": self.cleanup_workers,
            "job_events": self.cleanup_job_events,
            "stale_jobs": self.cleanup_stale_checkpoint_jobs,
        }

        tasks_to_run = tasks if tasks else list(task_map.keys())

        logger.info(f"Groundskeeper sweep starting (dry_run={dry_run}, tasks={tasks_to_run})")

        for task_name in tasks_to_run:
            if task_name not in task_map:
                logger.warning(f"Unknown task: {task_name}")
                continue

            result = task_map[task_name](dry_run=dry_run)

            # Handle both single results and lists
            if isinstance(result, list):
                for r in result:
                    all_results[r.task] = r
            else:
                all_results[result.task] = result

        # Summary
        total_items = sum(r.items_cleaned for r in all_results.values())
        total_bytes = sum(r.bytes_freed for r in all_results.values())
        total_errors = sum(len(r.errors) for r in all_results.values())

        mode = "[DRY RUN] " if dry_run else ""
        logger.info(f"{mode}Sweep complete: {total_items} items, {total_bytes/(1024*1024):.2f}MB freed, {total_errors} errors")

        self._save_state()

        return all_results

    def run_daemon(self, interval: int = 3600):
        """
        Run as daemon, sweeping every interval seconds.

        Args:
            interval: Seconds between sweeps (default: 3600 = 1 hour)
        """
        pid_file = self.base_dir / ".pids" / "groundskeeper.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        # Single-instance guard: check if already running
        if pid_file.exists():
            try:
                existing_pid = int(pid_file.read_text().strip())
                os.kill(existing_pid, 0)  # Check if process exists
                logger.error(f"Groundskeeper daemon already running (PID {existing_pid})")
                sys.exit(1)
            except (OSError, ProcessLookupError, ValueError):
                # Process dead or invalid PID, safe to continue
                pass

        logger.info(f"Groundskeeper daemon starting (interval={interval}s)")

        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Write PID
        pid_file.write_text(str(os.getpid()))

        try:
            while self.running:
                start = time.time()
                self.sweep()
                elapsed = time.time() - start
                # Sleep for remaining interval time (don't drift if sweep takes long)
                sleep_for = max(0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            logger.info("Groundskeeper daemon stopping")
            pid_file.unlink(missing_ok=True)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Groundskeeper - Unified cleanup daemon")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without actually cleaning")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (hourly sweeps)")
    parser.add_argument("--interval", type=int, default=3600, help="Daemon interval in seconds (default: 3600)")
    parser.add_argument("--task", type=str, action="append", dest="tasks",
                       help="Specific task(s) to run: jsonl, queue, battle_log, logs, pids, vacuum, workers, job_events")
    parser.add_argument("--force-vacuum", action="store_true", help="Force VACUUM even if not due")
    parser.add_argument("--base-dir", type=str, help="Base directory (auto-detected if not provided)")

    args = parser.parse_args()

    # Validate incompatible flag combinations
    if args.daemon:
        if args.dry_run:
            parser.error("--daemon and --dry-run are incompatible (daemon always runs real sweeps)")
        if args.tasks:
            parser.error("--daemon and --task are incompatible (daemon always runs all tasks)")
        if args.force_vacuum:
            parser.error("--daemon and --force-vacuum are incompatible (use single run for force-vacuum)")

    base_dir = Path(args.base_dir) if args.base_dir else None
    gk = Groundskeeper(base_dir=base_dir)

    if args.daemon:
        gk.run_daemon(interval=args.interval)
    else:
        # Single run
        if args.force_vacuum:
            gk.last_vacuum = datetime(2000, 1, 1)  # Force vacuum

        results = gk.sweep(dry_run=args.dry_run, tasks=args.tasks)

        # Print summary
        print("\n" + "=" * 60)
        print("GROUNDSKEEPER SWEEP REPORT")
        print("=" * 60)

        for task, result in results.items():
            print(f"\n{result}")
            if result.errors:
                for err in result.errors:
                    print(f"  ERROR: {err}")

        print("\n" + "=" * 60)
        total_mb = sum(r.bytes_freed for r in results.values()) / (1024 * 1024)
        print(f"Total: {total_mb:.2f}MB {'would be freed' if args.dry_run else 'freed'}")
        print("=" * 60)

        # Exit with non-zero code if any task had errors
        had_errors = any(r.errors for r in results.values())
        sys.exit(1 if had_errors else 0)


if __name__ == "__main__":
    main()
