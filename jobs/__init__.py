"""
Jobs - Distributed job execution system.

This module provides a central job store and execution infrastructure
for running tasks across multiple devices in the training lab.

Components:
- JobStore: Central persistence for all jobs (SQLite-backed)
- ClaimingWorker: Worker that claims jobs from the store
- Job API: REST endpoints for job management

Usage:
    from jobs import get_store, submit_job

    # Submit a job
    from guild.job_types import eval_job
    job = submit_job(eval_job("bin", level=5))

    # Workers claim and execute
    job = store.claim_next(device_id="worker1", roles=["eval_worker"])
    if job:
        # execute job...
        store.mark_complete(job.job_id, result={...})

Architecture:
    - All jobs go through the central store
    - Workers poll/claim jobs atomically
    - Lease-based claiming for crash recovery
    - Jobs visible via Tavern UI
"""

from jobs.store import (
    JobStore,
    SQLiteJobStore,
    get_store,
    reset_store,
    StoreMaintenanceWorker,
)
from jobs.client import (
    JobStoreClient,
    get_client,
    reset_client,
)

# Re-export job types from guild for convenience
from guild.job_types import (
    Job,
    JobSpec,
    JobResult,
    JobStatus,
    JobType,
    JobPriority,
    eval_job,
    sparring_job,
    data_gen_job,
    archive_job,
)


def submit_job(spec: JobSpec) -> Job:
    """
    Submit a job to the store.

    Convenience function that creates a Job and submits it.

    Args:
        spec: Job specification

    Returns:
        Submitted Job
    """
    job = Job.create(spec)
    store = get_store()
    return store.submit(job)


__all__ = [
    # Store
    "JobStore",
    "SQLiteJobStore",
    "get_store",
    "reset_store",
    "StoreMaintenanceWorker",
    # Client
    "JobStoreClient",
    "get_client",
    "reset_client",
    # Job types (re-exported)
    "Job",
    "JobSpec",
    "JobResult",
    "JobStatus",
    "JobType",
    "JobPriority",
    # Convenience constructors
    "eval_job",
    "sparring_job",
    "data_gen_job",
    "archive_job",
    "submit_job",
]
