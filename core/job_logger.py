#!/usr/bin/env python3
"""
Job Logger - Persistent job history for training files

Provides a persistent job history that tracks training file lifecycles:
- Job creation (file queued)
- Job start (training begins)
- Job completion (training finishes with metrics)
- Job failure (training failed with reason)

This enables:
- "What happened to file X?" queries
- Historical analysis of training runs
- Dashboard job tables
- Alerting on job failures

Usage (New API with Job):
    from core.job_logger import JobLogger
    from core.job import Job

    logger = JobLogger(base_dir / "status" / "job_history.jsonl")

    # Create job from file
    job = Job.from_file("data.jsonl", priority="high")
    logger.log_job(job)

    # When training starts
    job.start(step=84000, checkpoint="checkpoint-84000")
    logger.log_job(job)

    # When training completes
    job.complete(final_step=85000, final_loss=0.44)
    logger.log_job(job)

Legacy Usage (deprecated, for backwards compatibility):
    from core.job_logger import JobLogger, JobRecord

    logger = JobLogger(base_dir / "status" / "job_history.jsonl")
    record = logger.create_job("data.jsonl", priority="high")
    logger.start_job(record.job_id, checkpoint="checkpoint-84000")
    logger.complete_job(job_id=record.job_id, final_step=85000, final_loss=0.44)
"""

import json
import hashlib
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Import Job for first-class support
try:
    from core.job import Job, job_to_record_dict
    JOB_AVAILABLE = True
except ImportError:
    JOB_AVAILABLE = False
    Job = None
    job_to_record_dict = None


@dataclass
class JobRecord:
    """
    Complete record of a training job.

    Schema aligns with TrainingStatus fields where applicable to enable
    easy joins between job history and training metrics.

    Lifecycle States:
        - queued: File in queue, waiting for training
        - validating: Running pre-training validation
        - estimating: Computing time/resource estimates
        - running: Training in progress
        - completed: Training finished successfully
        - failed: Training failed (see reason field)
        - skipped: Skipped by user or system
    """
    # Identity
    job_id: str                         # Unique ID: "{timestamp}_{filename}"
    file_name: str                      # Original filename (e.g., "syllo_batch_0042.jsonl")
    file_path: str                      # Full path when queued
    priority: str                       # "high" | "normal" | "low"

    # Lifecycle
    status: str                         # queued | validating | running | completed | failed | skipped
    created_at: str                     # ISO timestamp when queued
    started_at: Optional[str] = None    # ISO timestamp when training started
    finished_at: Optional[str] = None   # ISO timestamp when training finished
    reason: Optional[str] = None        # Failure/skip reason

    # Dataset metrics
    num_examples: Optional[int] = None          # Number of training examples
    estimated_tokens: Optional[int] = None      # Estimated total tokens
    estimated_hours: Optional[float] = None     # Pre-flight time estimate
    actual_hours: Optional[float] = None        # Actual training duration

    # Training metrics (filled on completion)
    start_step: Optional[int] = None            # Step when training started
    final_step: Optional[int] = None            # Step when training finished
    final_loss: Optional[float] = None          # Final training loss
    final_val_loss: Optional[float] = None      # Final validation loss
    val_train_gap: Optional[float] = None       # val_loss - train_loss
    tokens_per_sec_avg: Optional[float] = None  # Average throughput

    # System context
    gpu_type: Optional[str] = None              # "RTX 4090", "RTX 3090", etc.
    gpu_memory_gb: Optional[float] = None       # VRAM used
    checkpoint_used: Optional[str] = None       # Checkpoint training started from
    best_checkpoint_after: Optional[str] = None # Best checkpoint produced
    config_hash: Optional[str] = None           # Hash of config.json for reproducibility
    git_commit: Optional[str] = None            # Git commit for reproducibility

    # Alerting
    regression_flag: bool = False               # True if regression detected
    alert_count: int = 0                        # Number of alerts during training

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extensible metadata


class JobLogger:
    """
    Append-only job history logger.

    Writes JobRecord entries to a JSONL file. Each state change appends a new
    line, allowing full audit trail of job lifecycle.

    Thread Safety:
        File appends are atomic on POSIX systems when writing < PIPE_BUF bytes.
        For safety, we write complete lines with explicit flush.
    """

    def __init__(self, history_file: Path):
        """
        Initialize job logger.

        Args:
            history_file: Path to job_history.jsonl file
        """
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Cache for looking up jobs by ID
        self._job_cache: Dict[str, JobRecord] = {}

    @staticmethod
    def now() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    @staticmethod
    def generate_job_id(filename: str) -> str:
        """
        Generate unique job ID from filename and timestamp.

        Format: YYYY-MM-DDTHH:MM:SS_{filename}
        """
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return f"{ts}_{filename}"

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.history_file.parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_config_hash(self, config_path: Optional[Path] = None) -> Optional[str]:
        """Get hash of config.json for reproducibility."""
        if config_path is None:
            config_path = self.history_file.parent.parent / "config.json"
        try:
            if config_path.exists():
                content = config_path.read_bytes()
                return f"sha256:{hashlib.sha256(content).hexdigest()[:16]}"
        except Exception:
            pass
        return None

    def append(self, record: JobRecord) -> None:
        """
        Append a job record to the history file.

        Args:
            record: JobRecord to append
        """
        line = json.dumps(asdict(record), ensure_ascii=False)
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        # Update cache
        self._job_cache[record.job_id] = record

    def log_job(self, job: 'Job') -> None:
        """
        Log a Job object to the history file (preferred API).

        This is the recommended method for logging jobs. It accepts the
        unified Job dataclass and converts it to the storage format.

        Args:
            job: Job object to log

        Example:
            job = Job.from_file("data.jsonl", priority="high")
            logger.log_job(job)  # Log queued state

            job.start(step=84000)
            logger.log_job(job)  # Log started state

            job.complete(final_step=85000, final_loss=0.44)
            logger.log_job(job)  # Log completed state
        """
        if not JOB_AVAILABLE:
            raise RuntimeError("Job class not available. Check core/job.py import.")

        # Convert Job to record-compatible dict and write
        record_dict = job_to_record_dict(job)
        line = json.dumps(record_dict, ensure_ascii=False)
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        # Update cache with job_id key
        self._job_cache[job.id] = job

    def create_job(
        self,
        file_name: str,
        file_path: str,
        priority: str = "normal",
        num_examples: Optional[int] = None,
        estimated_tokens: Optional[int] = None,
        estimated_hours: Optional[float] = None
    ) -> JobRecord:
        """
        Create a new job record when file is queued.

        Args:
            file_name: Name of the training file
            file_path: Full path to the file
            priority: Queue priority ("high", "normal", "low")
            num_examples: Number of training examples (if known)
            estimated_tokens: Estimated token count (if known)
            estimated_hours: Time estimate from TimeEstimator (if known)

        Returns:
            JobRecord: The created job record
        """
        job_id = self.generate_job_id(file_name)

        record = JobRecord(
            job_id=job_id,
            file_name=file_name,
            file_path=file_path,
            priority=priority,
            status="queued",
            created_at=self.now(),
            num_examples=num_examples,
            estimated_tokens=estimated_tokens,
            estimated_hours=estimated_hours,
            config_hash=self._get_config_hash(),
            git_commit=self._get_git_commit()
        )

        self.append(record)
        return record

    def start_job(
        self,
        job_id: str,
        checkpoint_used: Optional[str] = None,
        start_step: Optional[int] = None,
        gpu_type: Optional[str] = None,
        gpu_memory_gb: Optional[float] = None,
        num_examples: Optional[int] = None
    ) -> Optional[JobRecord]:
        """
        Record job start.

        Args:
            job_id: The job ID to update
            checkpoint_used: Checkpoint training started from
            start_step: Current global step when training started
            gpu_type: GPU type (auto-detected if None)
            gpu_memory_gb: VRAM available
            num_examples: Number of examples (if newly known)

        Returns:
            Updated JobRecord or None if job not found
        """
        record = self._job_cache.get(job_id)
        if record is None:
            return None

        # Create new record with updated fields
        updated = JobRecord(
            job_id=record.job_id,
            file_name=record.file_name,
            file_path=record.file_path,
            priority=record.priority,
            status="running",
            created_at=record.created_at,
            started_at=self.now(),
            num_examples=num_examples or record.num_examples,
            estimated_tokens=record.estimated_tokens,
            estimated_hours=record.estimated_hours,
            start_step=start_step,
            checkpoint_used=checkpoint_used,
            gpu_type=gpu_type,
            gpu_memory_gb=gpu_memory_gb,
            config_hash=record.config_hash,
            git_commit=record.git_commit,
            metadata=record.metadata
        )

        self.append(updated)
        return updated

    def complete_job(
        self,
        job_id: str,
        final_step: Optional[int] = None,
        final_loss: Optional[float] = None,
        final_val_loss: Optional[float] = None,
        tokens_per_sec_avg: Optional[float] = None,
        best_checkpoint_after: Optional[str] = None,
        regression_flag: bool = False,
        alert_count: int = 0
    ) -> Optional[JobRecord]:
        """
        Record job completion with final metrics.

        Args:
            job_id: The job ID to update
            final_step: Final global step
            final_loss: Final training loss
            final_val_loss: Final validation loss
            tokens_per_sec_avg: Average throughput
            best_checkpoint_after: Best checkpoint produced
            regression_flag: Whether regression was detected
            alert_count: Number of alerts during training

        Returns:
            Updated JobRecord or None if job not found
        """
        record = self._job_cache.get(job_id)
        if record is None:
            return None

        finished_at = self.now()

        # Calculate actual duration
        actual_hours = None
        if record.started_at:
            try:
                start = datetime.fromisoformat(record.started_at)
                end = datetime.fromisoformat(finished_at)
                actual_hours = (end - start).total_seconds() / 3600
            except Exception:
                pass

        # Calculate val/train gap
        val_train_gap = None
        if final_val_loss is not None and final_loss is not None:
            val_train_gap = final_val_loss - final_loss

        updated = JobRecord(
            job_id=record.job_id,
            file_name=record.file_name,
            file_path=record.file_path,
            priority=record.priority,
            status="completed",
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=finished_at,
            num_examples=record.num_examples,
            estimated_tokens=record.estimated_tokens,
            estimated_hours=record.estimated_hours,
            actual_hours=actual_hours,
            start_step=record.start_step,
            final_step=final_step,
            final_loss=final_loss,
            final_val_loss=final_val_loss,
            val_train_gap=val_train_gap,
            tokens_per_sec_avg=tokens_per_sec_avg,
            gpu_type=record.gpu_type,
            gpu_memory_gb=record.gpu_memory_gb,
            checkpoint_used=record.checkpoint_used,
            best_checkpoint_after=best_checkpoint_after,
            config_hash=record.config_hash,
            git_commit=record.git_commit,
            regression_flag=regression_flag,
            alert_count=alert_count,
            metadata=record.metadata
        )

        self.append(updated)
        return updated

    def fail_job(
        self,
        job_id: str,
        reason: str,
        final_step: Optional[int] = None,
        final_loss: Optional[float] = None
    ) -> Optional[JobRecord]:
        """
        Record job failure.

        Args:
            job_id: The job ID to update
            reason: Failure reason
            final_step: Step when failure occurred
            final_loss: Last known loss

        Returns:
            Updated JobRecord or None if job not found
        """
        record = self._job_cache.get(job_id)
        if record is None:
            return None

        finished_at = self.now()

        # Calculate actual duration if started
        actual_hours = None
        if record.started_at:
            try:
                start = datetime.fromisoformat(record.started_at)
                end = datetime.fromisoformat(finished_at)
                actual_hours = (end - start).total_seconds() / 3600
            except Exception:
                pass

        updated = JobRecord(
            job_id=record.job_id,
            file_name=record.file_name,
            file_path=record.file_path,
            priority=record.priority,
            status="failed",
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=finished_at,
            reason=reason,
            num_examples=record.num_examples,
            estimated_hours=record.estimated_hours,
            actual_hours=actual_hours,
            start_step=record.start_step,
            final_step=final_step,
            final_loss=final_loss,
            gpu_type=record.gpu_type,
            checkpoint_used=record.checkpoint_used,
            config_hash=record.config_hash,
            git_commit=record.git_commit,
            metadata=record.metadata
        )

        self.append(updated)
        return updated

    def skip_job(
        self,
        job_id: str,
        reason: str
    ) -> Optional[JobRecord]:
        """
        Record job skip.

        Args:
            job_id: The job ID to update
            reason: Skip reason

        Returns:
            Updated JobRecord or None if job not found
        """
        record = self._job_cache.get(job_id)
        if record is None:
            return None

        updated = JobRecord(
            job_id=record.job_id,
            file_name=record.file_name,
            file_path=record.file_path,
            priority=record.priority,
            status="skipped",
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=self.now(),
            reason=reason,
            num_examples=record.num_examples,
            config_hash=record.config_hash,
            git_commit=record.git_commit,
            metadata=record.metadata
        )

        self.append(updated)
        return updated

    def get_recent_jobs(self, limit: int = 20) -> List[JobRecord]:
        """
        Get most recent jobs from history.

        Reads from the end of the file for efficiency.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of JobRecord, most recent first
        """
        if not self.history_file.exists():
            return []

        jobs: Dict[str, JobRecord] = {}

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = JobRecord(**data)
                        jobs[record.job_id] = record  # Later entries override earlier
                    except Exception:
                        continue
        except Exception:
            return []

        # Sort by created_at descending, take limit
        sorted_jobs = sorted(
            jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )[:limit]

        return sorted_jobs

    def get_job_by_filename(self, filename: str) -> List[JobRecord]:
        """
        Get all jobs for a specific filename.

        Args:
            filename: The filename to search for

        Returns:
            List of JobRecord matching the filename
        """
        if not self.history_file.exists():
            return []

        jobs: Dict[str, JobRecord] = {}

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("file_name") == filename:
                            record = JobRecord(**data)
                            jobs[record.job_id] = record
                    except Exception:
                        continue
        except Exception:
            return []

        return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from job history.

        Returns:
            Dict with counts, averages, and recent trends
        """
        if not self.history_file.exists():
            return {
                "total_jobs": 0,
                "by_status": {},
                "avg_duration_hours": None,
                "total_examples_trained": 0
            }

        jobs: Dict[str, JobRecord] = {}

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = JobRecord(**data)
                        jobs[record.job_id] = record
                    except Exception:
                        continue
        except Exception:
            return {"error": "Failed to read job history"}

        # Aggregate stats
        by_status: Dict[str, int] = {}
        durations: List[float] = []
        total_examples = 0

        for job in jobs.values():
            by_status[job.status] = by_status.get(job.status, 0) + 1
            if job.actual_hours is not None:
                durations.append(job.actual_hours)
            if job.num_examples is not None and job.status == "completed":
                total_examples += job.num_examples

        avg_duration = sum(durations) / len(durations) if durations else None

        return {
            "total_jobs": len(jobs),
            "by_status": by_status,
            "avg_duration_hours": round(avg_duration, 2) if avg_duration else None,
            "total_examples_trained": total_examples,
            "total_hours": round(sum(durations), 2) if durations else 0
        }


def main():
    """Test job logger."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JobLogger(Path(tmpdir) / "job_history.jsonl")

        # Create a job
        job = logger.create_job(
            file_name="test_data.jsonl",
            file_path="/path/to/test_data.jsonl",
            priority="high",
            num_examples=1000,
            estimated_hours=0.5
        )
        print(f"Created job: {job.job_id}")

        # Start the job
        logger.start_job(
            job_id=job.job_id,
            checkpoint_used="checkpoint-84000",
            start_step=84000,
            gpu_type="RTX 4090"
        )
        print(f"Started job: {job.job_id}")

        # Complete the job
        logger.complete_job(
            job_id=job.job_id,
            final_step=85000,
            final_loss=0.44,
            final_val_loss=0.52,
            tokens_per_sec_avg=2400.0
        )
        print(f"Completed job: {job.job_id}")

        # Get recent jobs
        recent = logger.get_recent_jobs(limit=10)
        print(f"\nRecent jobs: {len(recent)}")
        for j in recent:
            print(f"  - {j.file_name}: {j.status}")

        # Get stats
        stats = logger.get_stats()
        print(f"\nStats: {stats}")


if __name__ == "__main__":
    main()
