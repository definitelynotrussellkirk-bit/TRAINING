"""
Job Store - Central persistence for distributed jobs.

The JobStore is the single source of truth for all jobs in the system.
Workers claim jobs atomically with lease-based locking for crash recovery.

Usage:
    from jobs.store import get_store, SQLiteJobStore

    store = get_store()

    # Submit a job
    job = store.submit(Job.create(spec))

    # Worker claims next available job
    job = store.claim_next(
        device_id="macmini_eval_1",
        roles=["eval_worker"],
        lease_duration=300
    )

    # Worker completes job
    store.mark_complete(job.job_id, result={"accuracy": 0.95})

Architecture:
    - SQLite database for durability
    - Atomic claim_next with lease expiration
    - Stale lease cleanup for crash recovery
    - Job history for analytics
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from guild.job_types import (
    Job,
    JobSpec,
    JobResult,
    JobStatus,
    JobType,
    JobPriority,
)

logger = logging.getLogger("job_store")


# =============================================================================
# ABSTRACT BASE
# =============================================================================

class JobStore(ABC):
    """
    Abstract base class for job storage.

    Defines the interface that all job stores must implement.
    """

    @abstractmethod
    def submit(self, job: Job) -> Job:
        """Submit a new job to the store."""
        pass

    @abstractmethod
    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        pass

    @abstractmethod
    def update(self, job: Job) -> None:
        """Update an existing job."""
        pass

    @abstractmethod
    def claim_next(
        self,
        device_id: str,
        roles: List[str],
        lease_duration: int = 300,
    ) -> Optional[Job]:
        """
        Atomically claim the next available job.

        Args:
            device_id: ID of the claiming device
            roles: Roles this device can handle
            lease_duration: Seconds before lease expires

        Returns:
            Claimed job or None if no jobs available
        """
        pass

    @abstractmethod
    def mark_running(self, job_id: str, device_id: str) -> bool:
        """Mark a job as running."""
        pass

    @abstractmethod
    def mark_complete(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark a job as completed with result."""
        pass

    @abstractmethod
    def mark_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed with error."""
        pass

    @abstractmethod
    def release(self, job_id: str) -> bool:
        """Release a claimed job back to pending."""
        pass

    @abstractmethod
    def expire_stale_leases(self) -> int:
        """Expire jobs with stale leases. Returns count expired."""
        pass

    @abstractmethod
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        device_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filters."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        pass

    @abstractmethod
    def cleanup_old(self, max_age_days: int = 7) -> int:
        """Clean up old completed jobs. Returns count deleted."""
        pass


# =============================================================================
# SQLITE IMPLEMENTATION
# =============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    priority TEXT NOT NULL DEFAULT 'normal',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    queued_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    requested_roles TEXT NOT NULL,
    target_device_id TEXT,
    claimed_by TEXT,
    lease_expires_at TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    tags TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority);
CREATE INDEX IF NOT EXISTS idx_jobs_claimed ON jobs(claimed_by);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at);
"""


class SQLiteJobStore(JobStore):
    """
    SQLite-backed job store.

    Features:
    - Atomic claim_next with lease-based locking
    - Automatic lease expiration for crash recovery
    - Priority ordering (CRITICAL > HIGH > NORMAL > LOW > IDLE)
    - Role-based job matching
    - Full job history with cleanup

    Thread-safe via connection pooling.
    """

    # Priority order for queue processing
    PRIORITY_ORDER = {
        JobPriority.CRITICAL: 0,
        JobPriority.HIGH: 1,
        JobPriority.NORMAL: 2,
        JobPriority.LOW: 3,
        JobPriority.IDLE: 4,
    }

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize SQLite job store.

        Args:
            db_path: Path to SQLite database file.
                    Defaults to vault/jobs.db
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            try:
                from core.paths import get_base_dir
                self.db_path = get_base_dir() / "vault" / "jobs.db"
            except ImportError:
                self.db_path = Path.cwd() / "vault" / "jobs.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

        logger.info(f"SQLiteJobStore initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(SCHEMA)
        conn.commit()

    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.utcnow().isoformat()

    def _job_to_row(self, job: Job) -> Dict[str, Any]:
        """Convert Job to database row."""
        return {
            "id": job.job_id,
            "type": job.spec.job_type.value,
            "payload": json.dumps(job.spec.payload),
            "status": job.status.value,
            "result": json.dumps(job.result.to_dict()) if job.result else None,
            "error": job.result.error if job.result else None,
            "priority": job.spec.priority.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": self._now(),
            "queued_at": job.queued_at.isoformat() if job.queued_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "requested_roles": json.dumps(self._get_roles_for_job(job)),
            "target_device_id": job.spec.target_device,
            "claimed_by": job.worker_id,
            "lease_expires_at": None,  # Set during claim
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "tags": json.dumps(job.spec.tags),
        }

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert database row to Job."""
        spec = JobSpec(
            job_type=JobType(row["type"]),
            payload=json.loads(row["payload"]),
            priority=JobPriority(row["priority"]),
            timeout=None,
            tags=json.loads(row["tags"]) if row["tags"] else [],
            target_device=row["target_device_id"],
        )

        job = Job(
            job_id=row["id"],
            spec=spec,
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
        )

        if row["queued_at"]:
            job.queued_at = datetime.fromisoformat(row["queued_at"])
        if row["started_at"]:
            job.started_at = datetime.fromisoformat(row["started_at"])
        if row["completed_at"]:
            job.completed_at = datetime.fromisoformat(row["completed_at"])
        if row["claimed_by"]:
            job.worker_id = row["claimed_by"]
        # Note: We store result as raw dict, not JobResult
        # Store it in a custom attribute for API access
        if row["result"]:
            job._raw_result = json.loads(row["result"])

        return job

    def _get_roles_for_job(self, job: Job) -> List[str]:
        """Get required roles for a job type."""
        from guild.job_router import JobRouter
        return JobRouter.JOB_TYPE_ROLES.get(job.spec.job_type, [])

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def submit(self, job: Job) -> Job:
        """Submit a new job."""
        row = self._job_to_row(job)

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, type, payload, status, result, error, priority,
                    created_at, updated_at, queued_at, started_at, completed_at,
                    requested_roles, target_device_id, claimed_by, lease_expires_at,
                    attempts, max_attempts, tags
                ) VALUES (
                    :id, :type, :payload, :status, :result, :error, :priority,
                    :created_at, :updated_at, :queued_at, :started_at, :completed_at,
                    :requested_roles, :target_device_id, :claimed_by, :lease_expires_at,
                    :attempts, :max_attempts, :tags
                )
                """,
                row,
            )

        logger.info(f"Submitted job {job.job_id}: {job.spec.job_type.value}")
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return self._row_to_job(row) if row else None

    def update(self, job: Job) -> None:
        """Update an existing job."""
        row = self._job_to_row(job)

        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = :status,
                    result = :result,
                    error = :error,
                    updated_at = :updated_at,
                    queued_at = :queued_at,
                    started_at = :started_at,
                    completed_at = :completed_at,
                    claimed_by = :claimed_by,
                    attempts = :attempts
                WHERE id = :id
                """,
                row,
            )

    def claim_next(
        self,
        device_id: str,
        roles: List[str],
        lease_duration: int = 300,
    ) -> Optional[Job]:
        """
        Atomically claim the next available job.

        Uses a single UPDATE with RETURNING for atomicity.
        Finds jobs where:
        - Status is PENDING, or
        - Status is CLAIMED/RUNNING with expired lease
        - Requested roles overlap with worker roles
        - Target device matches (if specified)

        Orders by priority, then created_at.
        """
        now = self._now()
        lease_expires = (datetime.utcnow() + timedelta(seconds=lease_duration)).isoformat()

        # Build role matching condition
        # Jobs store requested_roles as JSON array, we need to check overlap
        role_conditions = " OR ".join(
            f"requested_roles LIKE '%\"{role}\"%'" for role in roles
        )

        with self._transaction() as conn:
            # Find and claim in one atomic operation
            # SQLite doesn't have FOR UPDATE, but single connection + transaction is safe
            cursor = conn.execute(
                f"""
                UPDATE jobs SET
                    status = 'claimed',
                    claimed_by = ?,
                    lease_expires_at = ?,
                    updated_at = ?,
                    queued_at = COALESCE(queued_at, ?)
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE (
                        status = 'pending'
                        OR (status IN ('claimed', 'running') AND lease_expires_at < ?)
                    )
                    AND ({role_conditions})
                    AND (target_device_id IS NULL OR target_device_id = ?)
                    ORDER BY
                        CASE priority
                            WHEN 'critical' THEN 0
                            WHEN 'high' THEN 1
                            WHEN 'normal' THEN 2
                            WHEN 'low' THEN 3
                            WHEN 'idle' THEN 4
                        END,
                        created_at
                    LIMIT 1
                )
                RETURNING *
                """,
                (device_id, lease_expires, now, now, now, device_id),
            )

            row = cursor.fetchone()
            if row:
                job = self._row_to_job(row)
                logger.info(
                    f"Job {job.job_id} claimed by {device_id} "
                    f"(lease expires {lease_expires})"
                )
                return job

        return None

    def mark_running(self, job_id: str, device_id: str) -> bool:
        """Mark a claimed job as running."""
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'running',
                    started_at = ?,
                    updated_at = ?
                WHERE id = ? AND claimed_by = ? AND status = 'claimed'
                """,
                (now, now, job_id, device_id),
            )
            if cursor.rowcount > 0:
                logger.info(f"Job {job_id} now running on {device_id}")
                return True
        return False

    def mark_complete(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark a job as completed."""
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'completed',
                    result = ?,
                    completed_at = ?,
                    updated_at = ?,
                    lease_expires_at = NULL
                WHERE id = ? AND status IN ('claimed', 'running')
                """,
                (json.dumps(result), now, now, job_id),
            )
            if cursor.rowcount > 0:
                logger.info(f"Job {job_id} completed")
                return True
        return False

    def mark_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed."""
        now = self._now()

        with self._transaction() as conn:
            # Get current attempts
            cursor = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            attempts = row["attempts"] + 1
            max_attempts = row["max_attempts"]

            # If we can retry, go back to pending
            if attempts < max_attempts:
                new_status = "pending"
                logger.info(
                    f"Job {job_id} failed (attempt {attempts}/{max_attempts}), "
                    "returning to queue"
                )
            else:
                new_status = "failed"
                logger.info(
                    f"Job {job_id} failed (attempt {attempts}/{max_attempts}), "
                    "no more retries"
                )

            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    error = ?,
                    attempts = ?,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE NULL END,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    claimed_by = CASE WHEN ? = 'pending' THEN NULL ELSE claimed_by END
                WHERE id = ?
                """,
                (new_status, error, attempts, new_status, now, now, new_status, job_id),
            )
            return True

    def release(self, job_id: str) -> bool:
        """Release a claimed job back to pending."""
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'pending',
                    claimed_by = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ? AND status IN ('claimed', 'running')
                """,
                (now, job_id),
            )
            if cursor.rowcount > 0:
                logger.info(f"Job {job_id} released back to queue")
                return True
        return False

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'cancelled',
                    completed_at = ?,
                    updated_at = ?,
                    lease_expires_at = NULL
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (now, now, job_id),
            )
            if cursor.rowcount > 0:
                logger.info(f"Job {job_id} cancelled")
                return True
        return False

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def expire_stale_leases(self) -> int:
        """
        Expire jobs with stale leases.

        Jobs whose lease has expired are returned to pending status
        so they can be claimed by another worker.

        Returns:
            Number of jobs expired
        """
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'pending',
                    claimed_by = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE status IN ('claimed', 'running')
                AND lease_expires_at < ?
                """,
                (now, now),
            )
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Expired {count} stale job leases")
        return count

    def cleanup_old(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed/failed/cancelled jobs.

        Args:
            max_age_days: Delete jobs older than this

        Returns:
            Number of jobs deleted
        """
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < ?
                """,
                (cutoff,),
            )
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Cleaned up {count} old jobs")
        return count

    # =========================================================================
    # QUERIES
    # =========================================================================

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        device_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filters."""
        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if job_type:
            conditions.append("type = ?")
            params.append(job_type.value)

        if device_id:
            conditions.append("claimed_by = ?")
            params.append(device_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = self._get_conn()
        cursor = conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )

        return [self._row_to_job(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        conn = self._get_conn()

        # Status counts
        cursor = conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM jobs
            GROUP BY status
            """
        )
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Type counts
        cursor = conn.execute(
            """
            SELECT type, COUNT(*) as count
            FROM jobs
            GROUP BY type
            """
        )
        type_counts = {row["type"]: row["count"] for row in cursor.fetchall()}

        # Recent activity
        cursor = conn.execute(
            """
            SELECT COUNT(*) as count
            FROM jobs
            WHERE created_at > datetime('now', '-1 hour')
            """
        )
        recent_hour = cursor.fetchone()["count"]

        cursor = conn.execute(
            """
            SELECT COUNT(*) as count
            FROM jobs
            WHERE completed_at > datetime('now', '-1 hour')
            AND status = 'completed'
            """
        )
        completed_hour = cursor.fetchone()["count"]

        # Queue depth by priority
        cursor = conn.execute(
            """
            SELECT priority, COUNT(*) as count
            FROM jobs
            WHERE status = 'pending'
            GROUP BY priority
            """
        )
        queue_by_priority = {row["priority"]: row["count"] for row in cursor.fetchall()}

        # Active workers
        cursor = conn.execute(
            """
            SELECT claimed_by, COUNT(*) as count
            FROM jobs
            WHERE status IN ('claimed', 'running')
            AND claimed_by IS NOT NULL
            GROUP BY claimed_by
            """
        )
        active_workers = {row["claimed_by"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_jobs": sum(status_counts.values()),
            "by_status": status_counts,
            "by_type": type_counts,
            "queue_depth": status_counts.get("pending", 0),
            "queue_by_priority": queue_by_priority,
            "active_workers": active_workers,
            "submitted_last_hour": recent_hour,
            "completed_last_hour": completed_hour,
        }

    def get_queue_depth(self) -> int:
        """Get number of pending jobs."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM jobs WHERE status = 'pending'"
        )
        return cursor.fetchone()["count"]

    def get_active_jobs(self, device_id: Optional[str] = None) -> List[Job]:
        """Get currently running/claimed jobs."""
        conn = self._get_conn()

        if device_id:
            cursor = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('claimed', 'running')
                AND claimed_by = ?
                """,
                (device_id,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('claimed', 'running')"
            )

        return [self._row_to_job(row) for row in cursor.fetchall()]


# =============================================================================
# SINGLETON
# =============================================================================

_store: Optional[SQLiteJobStore] = None
_store_lock = threading.Lock()


def get_store(db_path: Optional[Path] = None) -> SQLiteJobStore:
    """Get or create the job store singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SQLiteJobStore(db_path)
    return _store


def reset_store():
    """Reset the store singleton (for testing)."""
    global _store
    _store = None


# =============================================================================
# BACKGROUND MAINTENANCE
# =============================================================================

class StoreMaintenanceWorker:
    """
    Background worker for store maintenance.

    Runs periodic tasks:
    - Expire stale leases (every 30 seconds)
    - Clean up old jobs (every hour)
    """

    def __init__(self, store: JobStore, lease_interval: int = 30, cleanup_interval: int = 3600):
        self.store = store
        self.lease_interval = lease_interval
        self.cleanup_interval = cleanup_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the maintenance worker."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Store maintenance worker started")

    def stop(self):
        """Stop the maintenance worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Main loop."""
        last_cleanup = time.time()

        while self._running:
            try:
                # Expire stale leases
                self.store.expire_stale_leases()

                # Periodic cleanup
                if time.time() - last_cleanup > self.cleanup_interval:
                    self.store.cleanup_old()
                    last_cleanup = time.time()

            except Exception as e:
                logger.error(f"Maintenance error: {e}")

            time.sleep(self.lease_interval)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Job Store CLI")
    parser.add_argument(
        "command",
        choices=["stats", "list", "pending", "active", "expire", "cleanup"],
        help="Command to run",
    )
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--type", help="Filter by job type")
    parser.add_argument("--limit", type=int, default=20, help="Limit results")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    store = get_store()

    if args.command == "stats":
        stats = store.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nJob Store Statistics")
            print("=" * 50)
            print(f"Total jobs: {stats['total_jobs']}")
            print(f"Queue depth: {stats['queue_depth']}")
            print(f"\nBy status:")
            for status, count in stats["by_status"].items():
                print(f"  {status}: {count}")
            print(f"\nBy type:")
            for jtype, count in stats["by_type"].items():
                print(f"  {jtype}: {count}")
            print(f"\nActive workers:")
            for worker, count in stats["active_workers"].items():
                print(f"  {worker}: {count} jobs")

    elif args.command == "list":
        status = JobStatus(args.status) if args.status else None
        job_type = JobType(args.type) if args.type else None
        jobs = store.list_jobs(status=status, job_type=job_type, limit=args.limit)

        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2))
        else:
            print(f"\nJobs ({len(jobs)} shown):")
            print("-" * 70)
            for job in jobs:
                print(
                    f"  {job.job_id}: {job.spec.job_type.value} "
                    f"[{job.status.value}] → {job.worker_id or 'unclaimed'}"
                )

    elif args.command == "pending":
        jobs = store.list_jobs(status=JobStatus.PENDING, limit=args.limit)
        print(f"\nPending jobs: {len(jobs)}")
        for job in jobs:
            print(f"  {job.job_id}: {job.spec.job_type.value} ({job.spec.priority.value})")

    elif args.command == "active":
        jobs = store.get_active_jobs()
        print(f"\nActive jobs: {len(jobs)}")
        for job in jobs:
            print(f"  {job.job_id}: {job.spec.job_type.value} → {job.worker_id}")

    elif args.command == "expire":
        count = store.expire_stale_leases()
        print(f"Expired {count} stale leases")

    elif args.command == "cleanup":
        count = store.cleanup_old()
        print(f"Cleaned up {count} old jobs")
