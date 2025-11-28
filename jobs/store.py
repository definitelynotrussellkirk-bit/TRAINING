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
    JobErrorCode,
    JobError,
    JobEventType,
    JobEvent,
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
    def mark_failed(
        self,
        job_id: str,
        error: str,
        error_code: Optional[JobErrorCode] = None,
    ) -> bool:
        """
        Mark a job as failed with error.

        Args:
            job_id: Job ID
            error: Error message string
            error_code: Structured error code (defaults to UNKNOWN)

        Returns:
            True if updated, False if job not found
        """
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

# Base table schema (no error_code - added via migration for existing DBs)
SCHEMA_BASE = """
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
"""

# Indexes created AFTER migrations ensure all columns exist
SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority);
CREATE INDEX IF NOT EXISTS idx_jobs_claimed ON jobs(claimed_by);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_jobs_error_code ON jobs(error_code);
"""

# Migration for existing databases
MIGRATION_ADD_ERROR_CODE = """
ALTER TABLE jobs ADD COLUMN error_code TEXT NOT NULL DEFAULT 'none';
CREATE INDEX IF NOT EXISTS idx_jobs_error_code ON jobs(error_code);
"""

# Workers table schema
WORKERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    worker_kind TEXT NOT NULL,
    roles_json TEXT NOT NULL,
    version TEXT,
    hostname TEXT,
    registered_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    last_seen_ip TEXT,
    status TEXT NOT NULL DEFAULT 'online',
    active_jobs INTEGER DEFAULT 0,
    -- Heterogeneous Cluster (HC) fields
    resource_class TEXT,
    priority_class TEXT,
    max_concurrent_jobs INTEGER DEFAULT 1,
    capabilities_json TEXT,
    gpus_json TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_heartbeat ON workers(last_heartbeat_at);
CREATE INDEX IF NOT EXISTS idx_workers_device ON workers(device_id);
CREATE INDEX IF NOT EXISTS idx_workers_resource_class ON workers(resource_class);
"""

# Migration for existing workers table (add HC columns)
MIGRATION_WORKERS_HC = """
ALTER TABLE workers ADD COLUMN resource_class TEXT;
ALTER TABLE workers ADD COLUMN priority_class TEXT;
ALTER TABLE workers ADD COLUMN max_concurrent_jobs INTEGER DEFAULT 1;
ALTER TABLE workers ADD COLUMN capabilities_json TEXT;
ALTER TABLE workers ADD COLUMN gpus_json TEXT;
CREATE INDEX IF NOT EXISTS idx_workers_resource_class ON workers(resource_class);
"""

# Job events table schema (audit trail)
EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    message TEXT,
    details_json TEXT,

    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_time ON job_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_job_events_type ON job_events(event_type);
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

        # 1. Create base tables (without error_code for backwards compat)
        conn.executescript(SCHEMA_BASE)
        conn.executescript(WORKERS_SCHEMA)
        conn.executescript(EVENTS_SCHEMA)
        conn.commit()

        # 2. Run migrations (adds error_code column if missing)
        self._migrate(conn)

        # 3. Create indexes AFTER migrations (so error_code column exists)
        conn.executescript(SCHEMA_INDEXES)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection):
        """Run migrations on existing database."""
        # Check if error_code column exists in jobs table
        cursor = conn.execute("PRAGMA table_info(jobs)")
        job_columns = {row[1] for row in cursor.fetchall()}

        if "error_code" not in job_columns:
            logger.info("Running migration: adding error_code column to jobs")
            try:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN error_code TEXT NOT NULL DEFAULT 'none'"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_error_code ON jobs(error_code)"
                )
                conn.commit()
                logger.info("Migration complete: error_code column added")
            except sqlite3.OperationalError as e:
                # Column might already exist (race condition)
                if "duplicate column" not in str(e).lower():
                    raise

        # Check if HC columns exist in workers table
        cursor = conn.execute("PRAGMA table_info(workers)")
        worker_columns = {row[1] for row in cursor.fetchall()}

        hc_columns = ["resource_class", "priority_class", "max_concurrent_jobs", "capabilities_json", "gpus_json"]
        missing_hc = [c for c in hc_columns if c not in worker_columns]

        if missing_hc:
            logger.info(f"Running migration: adding HC columns to workers: {missing_hc}")
            try:
                if "resource_class" not in worker_columns:
                    conn.execute("ALTER TABLE workers ADD COLUMN resource_class TEXT")
                if "priority_class" not in worker_columns:
                    conn.execute("ALTER TABLE workers ADD COLUMN priority_class TEXT")
                if "max_concurrent_jobs" not in worker_columns:
                    conn.execute("ALTER TABLE workers ADD COLUMN max_concurrent_jobs INTEGER DEFAULT 1")
                if "capabilities_json" not in worker_columns:
                    conn.execute("ALTER TABLE workers ADD COLUMN capabilities_json TEXT")
                if "gpus_json" not in worker_columns:
                    conn.execute("ALTER TABLE workers ADD COLUMN gpus_json TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workers_resource_class ON workers(resource_class)")
                conn.commit()
                logger.info("Migration complete: HC columns added to workers")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

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

        # Parse error_code (may be None for old rows)
        error_code_str = row["error_code"] if "error_code" in row.keys() else "none"
        try:
            error_code = JobErrorCode(error_code_str or "none")
        except ValueError:
            error_code = JobErrorCode.UNKNOWN

        job = Job(
            job_id=row["id"],
            spec=spec,
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            error_code=error_code,
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
        # Use registry as single source of truth for roles
        from jobs.registry import get_roles_for_job_type
        return get_roles_for_job_type(job.spec.job_type.value)

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

        # Record CREATED event
        self.record_event(
            job_id=job.job_id,
            event_type=JobEventType.CREATED,
            actor="system",
            message=f"Job submitted: {job.spec.job_type.value}",
            details={"job_type": job.spec.job_type.value, "priority": job.spec.priority.value},
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

                # Record CLAIMED event
                self.record_event(
                    job_id=job.job_id,
                    event_type=JobEventType.CLAIMED,
                    actor=device_id,
                    message=f"Job claimed by {device_id}",
                    details={"lease_expires": lease_expires},
                )

                logger.info(
                    f"Job {job.job_id} claimed by {device_id} "
                    f"(lease expires {lease_expires})"
                )
                return job

        return None

    def claim_next_smart(
        self,
        worker: Dict[str, Any],
        lease_duration: int = 300,
    ) -> Optional[Job]:
        """
        Smart job claiming using heterogeneous cluster routing.

        Uses the routing module to determine job type priority order
        based on worker capabilities, resource class, and cluster mode.

        Args:
            worker: Full worker info dict with resource_class, capabilities, etc.
            lease_duration: Seconds before lease expires

        Returns:
            Claimed job or None if no jobs available
        """
        try:
            from jobs.routing import ordered_job_types_for_worker, compute_cluster_mode
        except ImportError:
            # Fall back to basic claim if routing module not available
            device_id = worker.get("device_id")
            roles = worker.get("roles", [])
            return self.claim_next(device_id, roles, lease_duration)

        device_id = worker.get("device_id")
        roles = worker.get("roles", [])

        # Get queue stats for routing decisions
        queue_stats = self.get_queue_stats()

        # Compute cluster mode
        cluster_mode = compute_cluster_mode(queue_stats)

        # Get queue depths for routing
        queue_depths = {
            jt: stats.get("pending", 0)
            for jt, stats in queue_stats.items()
        }

        # Get ordered job types for this worker
        ordered_types = ordered_job_types_for_worker(
            worker=worker,
            cluster_mode=cluster_mode,
            queue_depths=queue_depths,
        )

        if not ordered_types:
            logger.debug(f"No job types available for worker {worker.get('worker_id')}")
            return None

        # Try claiming in priority order
        for job_type in ordered_types:
            job = self._claim_by_type(
                device_id=device_id,
                job_type=job_type,
                lease_duration=lease_duration,
            )
            if job:
                # Log routing decision
                logger.info(
                    f"Smart claim: {device_id} claimed {job_type} job {job.job_id} "
                    f"(cluster_mode={cluster_mode}, priority_order={ordered_types[:5]})"
                )
                return job

        return None

    def _claim_by_type(
        self,
        device_id: str,
        job_type: str,
        lease_duration: int = 300,
    ) -> Optional[Job]:
        """
        Claim next available job of a specific type.

        Internal helper for smart routing.
        """
        now = self._now()
        lease_expires = (datetime.utcnow() + timedelta(seconds=lease_duration)).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
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
                    AND type = ?
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
                (device_id, lease_expires, now, now, now, job_type, device_id),
            )

            row = cursor.fetchone()
            if row:
                job = self._row_to_job(row)

                # Record CLAIMED event
                self.record_event(
                    job_id=job.job_id,
                    event_type=JobEventType.CLAIMED,
                    actor=device_id,
                    message=f"Job claimed by {device_id} (smart routing)",
                    details={"lease_expires": lease_expires, "job_type": job_type},
                )

                return job

        return None

    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get queue statistics by job type.

        Returns:
            Dict of {job_type: {"pending": N, "running": M, "claimed": K}}
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT type, status, COUNT(*) as count
            FROM jobs
            WHERE status IN ('pending', 'claimed', 'running')
            GROUP BY type, status
            """
        )

        stats: Dict[str, Dict[str, int]] = {}
        for row in cursor.fetchall():
            job_type = row["type"]
            status = row["status"]
            count = row["count"]

            if job_type not in stats:
                stats[job_type] = {"pending": 0, "claimed": 0, "running": 0}
            stats[job_type][status] = count

        return stats

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
                # Record STARTED event
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.STARTED,
                    actor=device_id,
                    message="Execution started",
                )
                logger.info(f"Job {job_id} now running on {device_id}")
                return True
        return False

    def mark_complete(self, job_id: str, result: Dict[str, Any], worker_id: str = None) -> bool:
        """Mark a job as completed."""
        now = self._now()

        with self._transaction() as conn:
            # Get worker_id from job if not provided
            if not worker_id:
                cursor = conn.execute("SELECT claimed_by FROM jobs WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                worker_id = row["claimed_by"] if row else "unknown"

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
                # Record COMPLETED event
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.COMPLETED,
                    actor=worker_id or "unknown",
                    message="Completed successfully",
                    details={"result_keys": list(result.keys()) if result else []},
                )
                logger.info(f"Job {job_id} completed")
                return True
        return False

    def mark_failed(
        self,
        job_id: str,
        error: str,
        error_code: Optional[JobErrorCode] = None,
    ) -> bool:
        """
        Mark a job as failed with structured error.

        Args:
            job_id: Job ID
            error: Error message
            error_code: Structured error code (defaults to UNKNOWN)

        If the error code indicates a retryable error AND attempts < max_attempts,
        the job returns to pending. Otherwise it's marked terminal failed.
        """
        now = self._now()
        code = error_code or JobErrorCode.UNKNOWN

        with self._transaction() as conn:
            # Get current attempts and max_attempts
            cursor = conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            attempts = row["attempts"] + 1
            max_attempts = row["max_attempts"]

            # Decide whether to retry based on error code and attempt count
            can_retry = code.is_retryable and attempts < max_attempts

            if can_retry:
                new_status = "pending"
                logger.info(
                    f"Job {job_id} failed with {code.value} "
                    f"(attempt {attempts}/{max_attempts}), returning to queue"
                )
            else:
                new_status = "failed"
                reason = "not retryable" if not code.is_retryable else "max attempts"
                logger.info(
                    f"Job {job_id} failed with {code.value} "
                    f"(attempt {attempts}/{max_attempts}), {reason}"
                )

            # Get worker_id for event
            cursor = conn.execute("SELECT claimed_by FROM jobs WHERE id = ?", (job_id,))
            worker_row = cursor.fetchone()
            worker_id = worker_row["claimed_by"] if worker_row else "unknown"

            conn.execute(
                """
                UPDATE jobs SET
                    status = ?,
                    error = ?,
                    error_code = ?,
                    attempts = ?,
                    completed_at = CASE WHEN ? = 'failed' THEN ? ELSE NULL END,
                    updated_at = ?,
                    lease_expires_at = NULL,
                    claimed_by = CASE WHEN ? = 'pending' THEN NULL ELSE claimed_by END
                WHERE id = ?
                """,
                (new_status, error, code.value, attempts, new_status, now, now, new_status, job_id),
            )

            # Record FAILED or RETRIED event
            if can_retry:
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.RETRIED,
                    actor=worker_id,
                    message=f"Failed with {code.value}, returning to queue (attempt {attempts}/{max_attempts})",
                    details={"error_code": code.value, "error": error, "attempt": attempts},
                )
            else:
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.FAILED,
                    actor=worker_id,
                    message=f"Failed with {code.value}",
                    details={"error_code": code.value, "error": error, "attempt": attempts},
                )
            return True

    def release(self, job_id: str, worker_id: str = None) -> bool:
        """Release a claimed job back to pending."""
        now = self._now()

        with self._transaction() as conn:
            # Get worker_id if not provided
            if not worker_id:
                cursor = conn.execute("SELECT claimed_by FROM jobs WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                worker_id = row["claimed_by"] if row else "unknown"

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
                # Record RELEASED event
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.RELEASED,
                    actor=worker_id or "unknown",
                    message="Job released back to queue",
                )
                logger.info(f"Job {job_id} released back to queue")
                return True
        return False

    def cancel(self, job_id: str, actor: str = "user") -> bool:
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
                # Record CANCELLED event
                self.record_event(
                    job_id=job_id,
                    event_type=JobEventType.CANCELLED,
                    actor=actor,
                    message="Job cancelled",
                )
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
        so they can be claimed by another worker. Error code is set to
        LEASE_EXPIRED for tracking.

        Returns:
            Number of jobs expired
        """
        now = self._now()

        with self._transaction() as conn:
            # First get the jobs that will be expired (for event recording)
            cursor = conn.execute(
                """
                SELECT id, claimed_by FROM jobs
                WHERE status IN ('claimed', 'running')
                AND lease_expires_at < ?
                """,
                (now,),
            )
            expired_jobs = [(row["id"], row["claimed_by"]) for row in cursor.fetchall()]

            # Now update them
            cursor = conn.execute(
                """
                UPDATE jobs SET
                    status = 'pending',
                    claimed_by = NULL,
                    lease_expires_at = NULL,
                    error_code = ?,
                    error = 'Lease expired - worker may have crashed',
                    updated_at = ?
                WHERE status IN ('claimed', 'running')
                AND lease_expires_at < ?
                """,
                (JobErrorCode.LEASE_EXPIRED.value, now, now),
            )
            count = cursor.rowcount

        # Record events for each expired job (outside transaction for performance)
        for job_id, worker_id in expired_jobs:
            self.record_event(
                job_id=job_id,
                event_type=JobEventType.LEASE_EXPIRED,
                actor="system",
                message=f"Lease expired, worker {worker_id or 'unknown'} may have crashed",
                details={"original_worker": worker_id},
            )

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
        """Get job statistics including error breakdown."""
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

        # Error code breakdown (last 24 hours)
        cursor = conn.execute(
            """
            SELECT error_code, COUNT(*) as count
            FROM jobs
            WHERE status = 'failed'
            AND completed_at > datetime('now', '-24 hours')
            AND error_code IS NOT NULL
            AND error_code != 'none'
            GROUP BY error_code
            """
        )
        errors_by_code = {row["error_code"]: row["count"] for row in cursor.fetchall()}

        # Failed jobs by type (last 24 hours)
        cursor = conn.execute(
            """
            SELECT type, COUNT(*) as count
            FROM jobs
            WHERE status = 'failed'
            AND completed_at > datetime('now', '-24 hours')
            GROUP BY type
            """
        )
        errors_by_type = {row["type"]: row["count"] for row in cursor.fetchall()}

        # Calculate error rate (last 24h)
        cursor = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM jobs
            WHERE completed_at > datetime('now', '-24 hours')
            """
        )
        row = cursor.fetchone()
        completed_24h = row["completed"] or 0
        failed_24h = row["failed"] or 0
        total_24h = completed_24h + failed_24h
        error_rate_24h = (failed_24h / total_24h) if total_24h > 0 else 0

        return {
            "total_jobs": sum(status_counts.values()),
            "by_status": status_counts,
            "by_type": type_counts,
            "queue_depth": status_counts.get("pending", 0),
            "queue_by_priority": queue_by_priority,
            "active_workers": active_workers,
            "submitted_last_hour": recent_hour,
            "completed_last_hour": completed_hour,
            # New error tracking fields
            "errors_24h": {
                "by_code": errors_by_code,
                "by_type": errors_by_type,
                "total": failed_24h,
                "error_rate": round(error_rate_24h, 4),
            },
        }

    def get_queue_depth(self) -> int:
        """Get number of pending jobs."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM jobs WHERE status = 'pending'"
        )
        return cursor.fetchone()["count"]

    def count_jobs(self, status: JobStatus, job_type: Optional[str] = None) -> int:
        """
        Count jobs by status and optionally type.

        Much more efficient than list_jobs() for backpressure checks.

        Args:
            status: Job status to count
            job_type: Optional job type string to filter by

        Returns:
            Number of matching jobs
        """
        conn = self._get_conn()
        if job_type:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM jobs WHERE status = ? AND type = ?",
                (status.value, job_type),
            )
        else:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM jobs WHERE status = ?",
                (status.value,),
            )
        return cursor.fetchone()["count"] or 0

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

    # =========================================================================
    # WORKER MANAGEMENT
    # =========================================================================

    def register_worker(
        self,
        worker_id: str,
        device_id: str,
        worker_kind: str,
        roles: List[str],
        version: str = None,
        hostname: str = None,
        client_ip: str = None,
        metadata: Dict[str, Any] = None,
        # Heterogeneous Cluster fields
        resource_class: str = None,
        priority_class: str = None,
        max_concurrent_jobs: int = 1,
        capabilities: List[str] = None,
        gpus: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register or update a worker.

        Args:
            worker_id: Unique worker ID (device_id.worker_kind)
            device_id: Device ID from devices.json
            worker_kind: Type of worker (claiming, eval, etc.)
            roles: List of roles this worker can handle
            version: Worker code version
            hostname: Worker hostname
            client_ip: IP address of worker
            metadata: Additional metadata
            resource_class: Device resource class (gpu_heavy, gpu_medium, cpu_heavy, cpu_light)
            priority_class: Device priority class (critical, support, auxiliary)
            max_concurrent_jobs: Max concurrent jobs for this worker
            capabilities: List of capability tags
            gpus: List of GPU info dicts with name, vram_gb

        Returns:
            Registration result with allowed job types
        """
        now = self._now()

        with self._transaction() as conn:
            # Check if worker exists
            cursor = conn.execute(
                "SELECT id FROM workers WHERE id = ?",
                (worker_id,),
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing worker
                conn.execute(
                    """
                    UPDATE workers SET
                        roles_json = ?,
                        version = ?,
                        hostname = ?,
                        last_heartbeat_at = ?,
                        last_seen_ip = ?,
                        status = 'online',
                        resource_class = ?,
                        priority_class = ?,
                        max_concurrent_jobs = ?,
                        capabilities_json = ?,
                        gpus_json = ?,
                        metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(roles),
                        version,
                        hostname,
                        now,
                        client_ip,
                        resource_class,
                        priority_class,
                        max_concurrent_jobs,
                        json.dumps(capabilities) if capabilities else None,
                        json.dumps(gpus) if gpus else None,
                        json.dumps(metadata) if metadata else None,
                        worker_id,
                    ),
                )
                logger.info(f"Worker {worker_id} re-registered (resource_class={resource_class})")
            else:
                # Insert new worker
                conn.execute(
                    """
                    INSERT INTO workers (
                        id, device_id, worker_kind, roles_json,
                        version, hostname, registered_at, last_heartbeat_at,
                        last_seen_ip, status, active_jobs,
                        resource_class, priority_class, max_concurrent_jobs,
                        capabilities_json, gpus_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'online', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        worker_id,
                        device_id,
                        worker_kind,
                        json.dumps(roles),
                        version,
                        hostname,
                        now,
                        now,
                        client_ip,
                        resource_class,
                        priority_class,
                        max_concurrent_jobs,
                        json.dumps(capabilities) if capabilities else None,
                        json.dumps(gpus) if gpus else None,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                logger.info(f"Worker {worker_id} registered (device={device_id}, resource_class={resource_class})")

        # Get allowed job types for this worker
        from jobs.registry import get_allowed_job_types
        allowed_types = get_allowed_job_types(roles)

        return {
            "worker_id": worker_id,
            "registered": True,
            "allowed_job_types": allowed_types,
            "resource_class": resource_class,
            "priority_class": priority_class,
            "max_concurrent_jobs": max_concurrent_jobs,
        }

    def heartbeat_worker(
        self,
        worker_id: str,
        active_jobs: int = 0,
        status: str = "online",
    ) -> bool:
        """
        Update worker heartbeat.

        Args:
            worker_id: Worker ID
            active_jobs: Number of active jobs
            status: Worker status (online, draining)

        Returns:
            True if worker exists and was updated
        """
        now = self._now()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE workers SET
                    last_heartbeat_at = ?,
                    active_jobs = ?,
                    status = ?
                WHERE id = ?
                """,
                (now, active_jobs, status, worker_id),
            )
            return cursor.rowcount > 0

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM workers WHERE id = ?",
            (worker_id,),
        )
        row = cursor.fetchone()
        return self._row_to_worker(row) if row else None

    def list_workers(
        self,
        status: Optional[str] = None,
        max_age_sec: int = None,
    ) -> List[Dict[str, Any]]:
        """
        List workers with optional filters.

        Args:
            status: Filter by status (online, offline, draining)
            max_age_sec: Only include workers seen within this many seconds

        Returns:
            List of worker dictionaries
        """
        conn = self._get_conn()
        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status)

        if max_age_sec:
            cutoff = (datetime.utcnow() - timedelta(seconds=max_age_sec)).isoformat()
            conditions.append("last_heartbeat_at > ?")
            params.append(cutoff)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = conn.execute(
            f"""
            SELECT * FROM workers
            WHERE {where_clause}
            ORDER BY last_heartbeat_at DESC
            """,
            params,
        )

        return [self._row_to_worker(row) for row in cursor.fetchall()]

    def get_online_workers(self, max_age_sec: int = 120) -> List[Dict[str, Any]]:
        """Get workers that have heartbeated recently."""
        return self.list_workers(status="online", max_age_sec=max_age_sec)

    def mark_stale_workers_offline(self, max_age_sec: int = 120) -> int:
        """
        Mark workers as offline if they haven't heartbeated recently.

        Args:
            max_age_sec: Mark offline if no heartbeat within this time

        Returns:
            Number of workers marked offline
        """
        cutoff = (datetime.utcnow() - timedelta(seconds=max_age_sec)).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE workers SET
                    status = 'offline'
                WHERE status = 'online'
                AND last_heartbeat_at < ?
                """,
                (cutoff,),
            )
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Marked {count} workers as offline (no heartbeat for {max_age_sec}s)")
        return count

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        conn = self._get_conn()

        # Count by status
        cursor = conn.execute(
            """
            SELECT status, COUNT(*) as count
            FROM workers
            GROUP BY status
            """
        )
        by_status = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Count by role
        cursor = conn.execute("SELECT roles_json FROM workers WHERE status = 'online'")
        by_role: Dict[str, int] = {}
        for row in cursor.fetchall():
            roles = json.loads(row["roles_json"])
            for role in roles:
                by_role[role] = by_role.get(role, 0) + 1

        # Total active jobs
        cursor = conn.execute(
            "SELECT SUM(active_jobs) as total FROM workers WHERE status = 'online'"
        )
        total_active = cursor.fetchone()["total"] or 0

        return {
            "total": sum(by_status.values()),
            "by_status": by_status,
            "by_role": by_role,
            "total_active_jobs": total_active,
            "online": by_status.get("online", 0),
            "offline": by_status.get("offline", 0),
        }

    def _row_to_worker(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to worker dict."""
        heartbeat_at = datetime.fromisoformat(row["last_heartbeat_at"])
        age_sec = (datetime.utcnow() - heartbeat_at).total_seconds()

        # Helper to safely get optional columns (may not exist in old DBs)
        def safe_get(column: str, default=None):
            try:
                return row[column]
            except (IndexError, KeyError):
                return default

        # Parse HC JSON fields safely (may be None for old workers)
        capabilities = None
        gpus = None
        capabilities_json = safe_get("capabilities_json")
        gpus_json = safe_get("gpus_json")
        if capabilities_json:
            capabilities = json.loads(capabilities_json)
        if gpus_json:
            gpus = json.loads(gpus_json)

        return {
            "worker_id": row["id"],
            "device_id": row["device_id"],
            "worker_kind": row["worker_kind"],
            "roles": json.loads(row["roles_json"]),
            "version": row["version"],
            "hostname": row["hostname"],
            "registered_at": row["registered_at"],
            "last_heartbeat_at": row["last_heartbeat_at"],
            "heartbeat_age_sec": int(age_sec),
            "last_seen_ip": row["last_seen_ip"],
            "status": row["status"],
            "active_jobs": row["active_jobs"],
            # Heterogeneous Cluster fields
            "resource_class": safe_get("resource_class"),
            "priority_class": safe_get("priority_class"),
            "max_concurrent_jobs": safe_get("max_concurrent_jobs", 1),
            "capabilities": capabilities or [],
            "gpus": gpus or [],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        }

    def cleanup_old_workers(self, max_offline_days: int = 30) -> int:
        """
        Remove workers that have been offline for too long.

        Args:
            max_offline_days: Remove workers offline longer than this

        Returns:
            Number of workers removed
        """
        cutoff = (datetime.utcnow() - timedelta(days=max_offline_days)).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM workers
                WHERE status = 'offline'
                AND last_heartbeat_at < ?
                """,
                (cutoff,),
            )
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Removed {count} workers offline for >{max_offline_days} days")
        return count

    def cleanup_old_events(self, max_age_days: int = 14) -> int:
        """
        Remove old job events for jobs that are already cleaned up.

        Args:
            max_age_days: Remove events older than this for completed jobs

        Returns:
            Number of events removed
        """
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

        with self._transaction() as conn:
            # Delete events for jobs that no longer exist or are old completed/failed
            cursor = conn.execute(
                """
                DELETE FROM job_events
                WHERE timestamp < ?
                AND (
                    job_id NOT IN (SELECT id FROM jobs)
                    OR job_id IN (
                        SELECT id FROM jobs
                        WHERE status IN ('completed', 'failed', 'cancelled')
                        AND completed_at < ?
                    )
                )
                """,
                (cutoff, cutoff),
            )
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Cleaned up {count} old job events")
        return count

    # =========================================================================
    # JOB EVENTS (AUDIT TRAIL)
    # =========================================================================

    def record_event(
        self,
        job_id: str,
        event_type: JobEventType,
        actor: str = "system",
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record an event in the job's audit trail.

        Args:
            job_id: ID of the job
            event_type: Type of event
            actor: Who/what caused the event (worker_id, "system", "user")
            message: Human-readable description
            details: Additional structured data

        Returns:
            Event ID
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO job_events (job_id, timestamp, event_type, actor, message, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                self._now(),
                event_type.value,
                actor,
                message,
                json.dumps(details) if details else None,
            ),
        )
        conn.commit()

        # Also emit to battle log for real-time MMO-style feed
        self._emit_battle_log(job_id, event_type, actor, message, details)

        return cursor.lastrowid

    def _emit_battle_log(
        self,
        job_id: str,
        event_type: JobEventType,
        actor: str,
        message: Optional[str],
        details: Optional[Dict[str, Any]],
    ):
        """Emit job events to the battle log with MMO-style formatting."""
        try:
            from core.battle_log import log_jobs

            # Get job info if available
            job = self.get(job_id)
            job_type = job.spec.job_type.value if job else "unknown"
            short_id = job_id[:8]

            # Map event types to severity and message style
            if event_type == JobEventType.CREATED:
                log_jobs(
                    f"New {job_type} quest submitted",
                    severity="info",
                    details={"job_id": job_id, "job_type": job_type},
                )
            elif event_type == JobEventType.CLAIMED:
                worker = actor if actor != "system" else "a worker"
                log_jobs(
                    f"{job_type.capitalize()} quest claimed by {worker}",
                    severity="info",
                    details={"job_id": job_id, "worker": actor},
                )
            elif event_type == JobEventType.STARTED:
                log_jobs(
                    f"Battle started: {job_type} ({short_id})",
                    severity="info",
                    details={"job_id": job_id, "job_type": job_type},
                )
            elif event_type == JobEventType.COMPLETED:
                duration = details.get("duration_seconds", 0) if details else 0
                log_jobs(
                    f"Victory! {job_type} completed in {duration:.1f}s",
                    severity="success",
                    details={"job_id": job_id, "duration": duration},
                )
            elif event_type == JobEventType.FAILED:
                error_code = details.get("error_code", "unknown") if details else "unknown"
                log_jobs(
                    f"Defeat! {job_type} failed [{error_code}]",
                    severity="error",
                    details={"job_id": job_id, "error_code": error_code},
                )
            elif event_type == JobEventType.RETRIED:
                attempt = details.get("attempt", "?") if details else "?"
                log_jobs(
                    f"{job_type} quest retrying (attempt {attempt})",
                    severity="warning",
                    details={"job_id": job_id, "attempt": attempt},
                )
            elif event_type == JobEventType.LEASE_EXPIRED:
                log_jobs(
                    f"Lease expired on {job_type} quest, returning to board",
                    severity="warning",
                    details={"job_id": job_id},
                )
            elif event_type == JobEventType.CANCELLED:
                log_jobs(
                    f"{job_type} quest cancelled",
                    severity="info",
                    details={"job_id": job_id},
                )
            # Skip STATUS_CHANGE as it's usually redundant with other events

        except Exception as e:
            # Don't let battle log errors affect job processing
            logger.debug(f"Battle log emit failed: {e}")

    def get_events(
        self,
        job_id: str,
        limit: int = 100,
        event_types: Optional[List[JobEventType]] = None,
    ) -> List[JobEvent]:
        """
        Get events for a job.

        Args:
            job_id: ID of the job
            limit: Maximum number of events to return
            event_types: Filter to specific event types (optional)

        Returns:
            List of JobEvent objects, newest first
        """
        conn = self._get_conn()

        if event_types:
            type_values = [et.value for et in event_types]
            placeholders = ",".join("?" * len(type_values))
            cursor = conn.execute(
                f"""
                SELECT id, job_id, timestamp, event_type, actor, message, details_json
                FROM job_events
                WHERE job_id = ? AND event_type IN ({placeholders})
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                [job_id] + type_values + [limit],
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, job_id, timestamp, event_type, actor, message, details_json
                FROM job_events
                WHERE job_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (job_id, limit),
            )

        events = []
        for row in cursor.fetchall():
            try:
                event_type = JobEventType(row["event_type"])
            except ValueError:
                # Unknown event type - skip or use a placeholder
                continue

            events.append(JobEvent(
                event_id=row["id"],
                job_id=row["job_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type=event_type,
                actor=row["actor"],
                message=row["message"],
                details=json.loads(row["details_json"]) if row["details_json"] else None,
            ))

        # Return in chronological order (oldest first)
        return list(reversed(events))

    def get_recent_events(
        self,
        limit: int = 50,
        event_types: Optional[List[JobEventType]] = None,
    ) -> List[JobEvent]:
        """
        Get recent events across all jobs.

        Useful for monitoring/dashboard.

        Args:
            limit: Maximum number of events to return
            event_types: Filter to specific event types (optional)

        Returns:
            List of JobEvent objects, newest first
        """
        conn = self._get_conn()

        if event_types:
            type_values = [et.value for et in event_types]
            placeholders = ",".join("?" * len(type_values))
            cursor = conn.execute(
                f"""
                SELECT id, job_id, timestamp, event_type, actor, message, details_json
                FROM job_events
                WHERE event_type IN ({placeholders})
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                type_values + [limit],
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, job_id, timestamp, event_type, actor, message, details_json
                FROM job_events
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )

        events = []
        for row in cursor.fetchall():
            try:
                event_type = JobEventType(row["event_type"])
            except ValueError:
                continue

            events.append(JobEvent(
                event_id=row["id"],
                job_id=row["job_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type=event_type,
                actor=row["actor"],
                message=row["message"],
                details=json.loads(row["details_json"]) if row["details_json"] else None,
            ))

        return events  # Already newest first


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
    - Mark stale workers offline (every 60 seconds)
    - Clean up old jobs, events, workers (every hour)
    """

    def __init__(self, store: JobStore, lease_interval: int = 30, cleanup_interval: int = 3600):
        self.store = store
        self.lease_interval = lease_interval
        self.cleanup_interval = cleanup_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_worker_check = 0

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
                # Expire stale job leases (every cycle)
                self.store.expire_stale_leases()

                # Mark stale workers offline (every 60 seconds)
                if time.time() - self._last_worker_check > 60:
                    self.store.mark_stale_workers_offline(max_age_sec=120)
                    self._last_worker_check = time.time()

                # Periodic deep cleanup (every hour)
                if time.time() - last_cleanup > self.cleanup_interval:
                    self.store.cleanup_old()           # Old jobs
                    self.store.cleanup_old_events()    # Old job events
                    self.store.cleanup_old_workers()   # Long-offline workers
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
