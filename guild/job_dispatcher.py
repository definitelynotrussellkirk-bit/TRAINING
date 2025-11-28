"""
Job Dispatcher - Submit jobs to remote workers.

The dispatcher routes jobs to appropriate workers and tracks their execution.
Workers run HTTP servers that accept job submissions via REST API.

Usage:
    from guild.job_dispatcher import JobDispatcher, get_dispatcher

    dispatcher = get_dispatcher()

    # Submit a job
    job_id = dispatcher.submit(eval_job("bin", level=5))

    # Wait for result
    result = dispatcher.wait(job_id, timeout=300)
    print(f"Accuracy: {result.output.get('accuracy')}")

    # Or submit and wait in one call
    result = dispatcher.run(eval_job("bin", level=5))

Worker API:
    Workers expose these endpoints:
    - POST /jobs          - Submit a job
    - GET  /jobs/{id}     - Get job status
    - POST /jobs/{id}/cancel - Cancel a job
    - GET  /health        - Health check
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from guild.job_types import (
    Job,
    JobSpec,
    JobResult,
    JobStatus,
    JobType,
    JobPriority,
)
from guild.job_router import JobRouter, WorkerInfo, get_router

logger = logging.getLogger("job_dispatcher")


# Default worker port
DEFAULT_WORKER_PORT = 8900


@dataclass
class DispatcherConfig:
    """Configuration for the job dispatcher."""
    worker_port: int = DEFAULT_WORKER_PORT
    default_timeout: int = 300  # Default wait timeout
    poll_interval: float = 2.0  # Status poll interval
    max_retries: int = 3  # Retries for failed jobs
    retry_delay: float = 5.0  # Delay between retries
    persist_jobs: bool = True  # Save jobs to disk
    jobs_dir: Optional[Path] = None  # Where to save jobs


class JobDispatcher:
    """
    Dispatches jobs to workers and tracks their execution.

    The dispatcher:
    1. Routes jobs to appropriate workers via JobRouter
    2. Submits jobs via HTTP to worker endpoints
    3. Tracks job status and handles retries
    4. Provides sync/async interfaces for waiting on results
    """

    def __init__(
        self,
        config: Optional[DispatcherConfig] = None,
        router: Optional[JobRouter] = None,
    ):
        """
        Initialize the job dispatcher.

        Args:
            config: Dispatcher configuration
            router: Job router (uses singleton if None)
        """
        self.config = config or DispatcherConfig()
        self.router = router or get_router()

        # Job tracking
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

        # Setup jobs directory
        if self.config.persist_jobs:
            if self.config.jobs_dir:
                self._jobs_dir = self.config.jobs_dir
            else:
                from core.paths import get_base_dir
                self._jobs_dir = get_base_dir() / "status" / "jobs"
            self._jobs_dir.mkdir(parents=True, exist_ok=True)
            self._load_pending_jobs()

    def _load_pending_jobs(self) -> None:
        """Load pending jobs from disk on startup."""
        if not self._jobs_dir.exists():
            return

        for job_file in self._jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)

                job = self._job_from_dict(data)
                if not job.is_terminal:
                    self._jobs[job.job_id] = job
                    logger.info(f"Loaded pending job: {job.job_id}")
            except Exception as e:
                logger.warning(f"Failed to load job from {job_file}: {e}")

    def _job_from_dict(self, data: Dict[str, Any]) -> Job:
        """Create Job from dict."""
        spec = JobSpec.from_dict(data["spec"])
        job = Job(
            job_id=data["job_id"],
            spec=spec,
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
        )
        if data.get("queued_at"):
            job.queued_at = datetime.fromisoformat(data["queued_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("worker_id"):
            job.worker_id = data["worker_id"]
        if data.get("result"):
            job.result = JobResult.from_dict(data["result"])
        return job

    def _save_job(self, job: Job) -> None:
        """Save job to disk."""
        if not self.config.persist_jobs:
            return

        job_file = self._jobs_dir / f"{job.job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

    def _get_worker_url(self, worker: WorkerInfo) -> str:
        """Get base URL for a worker."""
        # Check if host registry has a worker service port
        try:
            from core.hosts import get_host_for_device

            host = get_host_for_device(worker.device_id)
            if host and host.services:
                # Look for 'worker' service
                for svc_name, svc_info in host.services.items():
                    if svc_name == "worker":
                        port = svc_info.get("port", self.config.worker_port)
                        return f"http://{worker.hostname}:{port}"
        except Exception:
            pass

        # Default to configured port
        return f"http://{worker.hostname}:{self.config.worker_port}"

    # =========================================================================
    # JOB SUBMISSION
    # =========================================================================

    def submit(
        self,
        spec: JobSpec,
        wait: bool = False,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Submit a job for execution.

        Args:
            spec: Job specification
            wait: If True, block until completion
            timeout: Wait timeout (only if wait=True)

        Returns:
            Job ID
        """
        # Create job
        job = Job.create(spec)

        with self._lock:
            self._jobs[job.job_id] = job

        logger.info(f"Created job {job.job_id}: {spec.job_type.value}")

        # Route to worker
        decision = self.router.route(spec)

        if not decision.success:
            job.status = JobStatus.FAILED
            job.result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                job_type=spec.job_type,
                worker_id="",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error=f"Routing failed: {decision.reason}",
            )
            self._save_job(job)
            logger.warning(f"Job {job.job_id} routing failed: {decision.reason}")

            if wait:
                return job.job_id

            raise RuntimeError(f"Cannot route job: {decision.reason}")

        worker = decision.worker
        job.worker_id = worker.device_id
        job.queued_at = datetime.now()
        job.status = JobStatus.QUEUED

        # Submit to worker
        success = self._submit_to_worker(job, worker)

        if not success and decision.alternatives:
            # Try alternative workers
            for alt_worker in decision.alternatives:
                logger.info(f"Trying alternative worker: {alt_worker.device_id}")
                job.worker_id = alt_worker.device_id
                success = self._submit_to_worker(job, alt_worker)
                if success:
                    break

        if not success:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            self._save_job(job)

        if wait:
            self.wait(job.job_id, timeout=timeout or self.config.default_timeout)

        return job.job_id

    def _submit_to_worker(self, job: Job, worker: WorkerInfo) -> bool:
        """Submit job to worker via HTTP."""
        url = f"{self._get_worker_url(worker)}/jobs"

        try:
            payload = {
                "job_id": job.job_id,
                "spec": job.spec.to_dict(),
            }

            response = requests.post(
                url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200 or response.status_code == 202:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                self.router.mark_job_started(worker.device_id)
                self._save_job(job)
                logger.info(f"Job {job.job_id} submitted to {worker.device_id}")
                return True
            else:
                logger.warning(
                    f"Worker {worker.device_id} rejected job: "
                    f"{response.status_code} - {response.text}"
                )
                return False

        except requests.RequestException as e:
            logger.warning(f"Failed to submit to {worker.device_id}: {e}")
            self.router.update_worker_status(worker.device_id, is_online=False)
            return False

    # =========================================================================
    # JOB MONITORING
    # =========================================================================

    def get_status(self, job_id: str) -> Optional[Job]:
        """Get job status."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        # If job is running, poll worker for status
        if job.status == JobStatus.RUNNING and job.worker_id:
            self._poll_job_status(job)

        return job

    def _poll_job_status(self, job: Job) -> None:
        """Poll worker for job status."""
        worker = self.router.get_worker_status(job.worker_id)
        if not worker:
            return

        url = f"{self._get_worker_url(worker)}/jobs/{job.job_id}"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = JobStatus(data.get("status", "running"))

                if status != job.status:
                    job.status = status
                    logger.info(f"Job {job.job_id} status: {status.value}")

                if status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                    job.completed_at = datetime.now()
                    if "result" in data:
                        job.result = JobResult.from_dict(data["result"])
                    self.router.mark_job_completed(job.worker_id)
                    self._save_job(job)

        except requests.RequestException as e:
            logger.debug(f"Failed to poll job status: {e}")

    def wait(
        self,
        job_id: str,
        timeout: Optional[int] = None,
        poll_interval: Optional[float] = None,
    ) -> JobResult:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: How often to poll status

        Returns:
            JobResult when job completes

        Raises:
            TimeoutError: If timeout exceeded
            KeyError: If job not found
        """
        timeout = timeout or self.config.default_timeout
        poll_interval = poll_interval or self.config.poll_interval

        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")

        start_time = time.time()

        while not job.is_terminal:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                job.status = JobStatus.TIMEOUT
                job.completed_at = datetime.now()
                self._save_job(job)
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            # Poll status
            self._poll_job_status(job)

            if not job.is_terminal:
                time.sleep(poll_interval)

        if job.result:
            return job.result

        # Create result from job state
        return JobResult(
            job_id=job.job_id,
            status=job.status,
            job_type=job.spec.job_type,
            worker_id=job.worker_id or "",
            started_at=job.started_at or job.created_at,
            completed_at=job.completed_at,
            duration_seconds=(
                (job.completed_at - job.started_at).total_seconds()
                if job.completed_at and job.started_at
                else 0
            ),
        )

    def run(
        self,
        spec: JobSpec,
        timeout: Optional[int] = None,
    ) -> JobResult:
        """
        Submit a job and wait for completion.

        Convenience method combining submit() and wait().

        Args:
            spec: Job specification
            timeout: Maximum wait time

        Returns:
            JobResult
        """
        job_id = self.submit(spec)
        return self.wait(job_id, timeout=timeout)

    # =========================================================================
    # JOB MANAGEMENT
    # =========================================================================

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if not running
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.is_terminal:
            return False

        # Try to cancel on worker
        if job.worker_id:
            worker = self.router.get_worker_status(job.worker_id)
            if worker:
                try:
                    url = f"{self._get_worker_url(worker)}/jobs/{job_id}/cancel"
                    response = requests.post(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Cancelled job {job_id} on worker")
                except requests.RequestException:
                    pass

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.router.mark_job_completed(job.worker_id)
        self._save_job(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    def retry(self, job_id: str) -> Optional[str]:
        """
        Retry a failed job.

        Args:
            job_id: Job ID to retry

        Returns:
            New job ID if retry submitted, None if can't retry
        """
        job = self._jobs.get(job_id)
        if not job or not job.can_retry:
            return None

        job.attempts += 1
        logger.info(f"Retrying job {job_id} (attempt {job.attempts})")

        return self.submit(job.spec)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
    ) -> List[Job]:
        """
        List jobs with optional filters.

        Args:
            status: Filter by status
            job_type: Filter by job type
            limit: Maximum jobs to return

        Returns:
            List of jobs, newest first
        """
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if job_type:
            jobs = [j for j in jobs if j.spec.job_type == job_type]

        # Sort by created_at (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed jobs.

        Args:
            max_age_hours: Remove jobs older than this

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        removed = 0

        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.is_terminal:
                    if job.completed_at and job.completed_at.timestamp() < cutoff:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                # Remove from disk
                job_file = self._jobs_dir / f"{job_id}.json"
                if job_file.exists():
                    job_file.unlink()
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old jobs")

        return removed

    # =========================================================================
    # INFO
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get dispatcher summary."""
        jobs = list(self._jobs.values())

        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = len([j for j in jobs if j.status == status])

        return {
            "total_jobs": len(jobs),
            "by_status": status_counts,
            "router_summary": self.router.get_summary(),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_dispatcher: Optional[JobDispatcher] = None


def get_dispatcher() -> JobDispatcher:
    """Get or create the job dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = JobDispatcher()
    return _dispatcher


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    from guild.job_types import eval_job, sparring_job, data_gen_job

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Job Dispatcher - Submit jobs to workers")
    parser.add_argument("command", nargs="?", default="summary",
                        choices=["summary", "list", "submit", "status", "cancel"])
    parser.add_argument("--job-type", choices=[jt.value for jt in JobType],
                        help="Job type for submit")
    parser.add_argument("--skill", default="bin", help="Skill ID for eval/sparring")
    parser.add_argument("--level", type=int, default=5, help="Level for eval")
    parser.add_argument("--count", type=int, default=100, help="Count for sparring/gen")
    parser.add_argument("--job-id", help="Job ID for status/cancel")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--timeout", type=int, default=300, help="Wait timeout")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    dispatcher = get_dispatcher()

    if args.command == "summary":
        summary = dispatcher.get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("\nJob Dispatcher Summary")
            print("=" * 50)
            print(f"Total jobs: {summary['total_jobs']}")
            print("\nBy status:")
            for status, count in summary['by_status'].items():
                if count > 0:
                    print(f"  {status}: {count}")

    elif args.command == "list":
        jobs = dispatcher.list_jobs()
        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2))
        else:
            print("\nJobs:")
            print("-" * 70)
            for job in jobs[:20]:
                status_icon = {
                    JobStatus.PENDING: "⏳",
                    JobStatus.QUEUED: "📤",
                    JobStatus.RUNNING: "🔄",
                    JobStatus.COMPLETED: "✅",
                    JobStatus.FAILED: "❌",
                    JobStatus.CANCELLED: "🚫",
                    JobStatus.TIMEOUT: "⏱️",
                }.get(job.status, "?")

                print(f"{status_icon} {job.job_id}: {job.spec.job_type.value} "
                      f"[{job.status.value}] → {job.worker_id or 'unassigned'}")

    elif args.command == "submit":
        if not args.job_type:
            print("Error: --job-type required for submit")
            exit(1)

        # Create job spec based on type
        job_type = JobType(args.job_type)
        if job_type == JobType.EVAL:
            spec = eval_job(args.skill, args.level, args.count)
        elif job_type == JobType.SPARRING:
            spec = sparring_job(args.skill, args.count)
        elif job_type == JobType.DATA_GEN:
            spec = data_gen_job(args.skill, args.count)
        else:
            spec = JobSpec(job_type=job_type)

        try:
            job_id = dispatcher.submit(spec, wait=args.wait, timeout=args.timeout)
            print(f"Submitted job: {job_id}")

            if args.wait:
                job = dispatcher.get_status(job_id)
                if job:
                    print(f"Status: {job.status.value}")
                    if job.result:
                        print(f"Result: {job.result.output}")

        except Exception as e:
            print(f"Error: {e}")
            exit(1)

    elif args.command == "status":
        if not args.job_id:
            print("Error: --job-id required")
            exit(1)

        job = dispatcher.get_status(args.job_id)
        if job:
            if args.json:
                print(json.dumps(job.to_dict(), indent=2))
            else:
                print(f"\nJob {job.job_id}:")
                print(f"  Type: {job.spec.job_type.value}")
                print(f"  Status: {job.status.value}")
                print(f"  Worker: {job.worker_id or 'unassigned'}")
                print(f"  Created: {job.created_at}")
                if job.result:
                    print(f"  Duration: {job.result.duration_seconds:.1f}s")
        else:
            print(f"Job not found: {args.job_id}")

    elif args.command == "cancel":
        if not args.job_id:
            print("Error: --job-id required")
            exit(1)

        if dispatcher.cancel(args.job_id):
            print(f"Cancelled job: {args.job_id}")
        else:
            print(f"Could not cancel job: {args.job_id}")
