"""
Job Types - Types of jobs that can be dispatched to workers.

This module defines the job type system for distributed task execution.
Jobs can be routed to appropriate workers based on their type and requirements.

Usage:
    from guild.job_types import JobType, JobSpec, JobResult

    # Create a job spec
    spec = JobSpec(
        job_type=JobType.EVAL,
        payload={"skill_id": "bin", "level": 5, "batch_size": 100},
    )

    # Submit via dispatcher
    result = dispatcher.submit(spec)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class JobType(str, Enum):
    """
    Types of jobs that can be dispatched.

    Each job type maps to worker roles that can handle it:
    - EVAL → EVAL_WORKER role
    - DATA_GEN → DATA_FORGE role
    - SPARRING → EVAL_WORKER + INFERENCE (needs both)
    - ARCHIVE → VAULT_WORKER role
    - ANALYTICS → ANALYTICS role
    """
    # Inference-requiring jobs
    EVAL = "eval"                  # Skill evaluation (needs inference)
    SPARRING = "sparring"          # Self-correction sparring (needs inference)
    INFERENCE = "inference"        # Direct inference request

    # Data generation jobs (CPU-bound)
    DATA_GEN = "data_gen"          # Generate training data
    DATA_FILTER = "data_filter"    # Filter/validate data
    DATA_CONVERT = "data_convert"  # Convert data formats

    # Storage/archival jobs
    ARCHIVE = "archive"            # Archive checkpoints
    RETENTION = "retention"        # Apply retention policies
    SYNC = "sync"                  # Sync between zones

    # Reporting/analytics jobs
    ANALYTICS = "analytics"        # Run analytics/metrics
    REPORT = "report"              # Generate reports
    HEALTH_CHECK = "health_check"  # System health check

    # Model Archaeology jobs (interpretability)
    LAYER_STATS = "layer_stats"    # Per-layer weight/activation stats
    LAYER_DRIFT = "layer_drift"    # Compare weight drift between checkpoints

    @property
    def requires_gpu(self) -> bool:
        """Whether this job type requires GPU."""
        return self in {
            JobType.EVAL,
            JobType.SPARRING,
            JobType.INFERENCE,
            JobType.LAYER_STATS,  # Needs GPU for activation computation
        }

    @property
    def requires_inference(self) -> bool:
        """Whether this job type requires inference server access."""
        return self in {
            JobType.EVAL,
            JobType.SPARRING,
            JobType.INFERENCE,
        }

    @property
    def default_timeout(self) -> int:
        """Default timeout in seconds."""
        timeouts = {
            JobType.EVAL: 600,           # 10 min
            JobType.SPARRING: 1800,      # 30 min
            JobType.INFERENCE: 60,       # 1 min
            JobType.DATA_GEN: 300,       # 5 min
            JobType.DATA_FILTER: 300,
            JobType.DATA_CONVERT: 300,
            JobType.ARCHIVE: 1800,       # 30 min
            JobType.RETENTION: 600,
            JobType.SYNC: 3600,          # 1 hour
            JobType.ANALYTICS: 300,
            JobType.REPORT: 120,
            JobType.HEALTH_CHECK: 60,
            JobType.LAYER_STATS: 1800,   # 30 min (model loading is slow)
            JobType.LAYER_DRIFT: 600,    # 10 min (CPU-only)
        }
        return timeouts.get(self, 300)


class JobStatus(str, Enum):
    """Status of a job."""
    PENDING = "pending"        # Waiting to be picked up
    QUEUED = "queued"          # In worker queue (push model)
    CLAIMED = "claimed"        # Claimed by worker (pull model), not yet running
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Finished successfully
    FAILED = "failed"          # Failed with error
    CANCELLED = "cancelled"    # Cancelled by user
    TIMEOUT = "timeout"        # Timed out


class JobErrorCode(str, Enum):
    """
    Structured error categories for debuggable failures.

    Error codes enable analysis like:
    - "eval jobs failing with inference_error 60% of the time"
    - "sparring jobs hitting timeout"

    Codes are grouped by category for easier triage.
    """
    # Success (no error)
    NONE = "none"

    # Transport/Network errors - usually retryable
    TRANSPORT_ERROR = "transport_error"       # HTTP failures, timeouts to services
    CONNECTION_REFUSED = "connection_refused" # Service unreachable

    # Worker Setup errors - usually NOT retryable
    WORKER_SETUP = "worker_setup"             # Missing dependencies, config errors
    MODEL_NOT_FOUND = "model_not_found"       # Checkpoint/model doesn't exist
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # GPU OOM, disk full

    # Execution errors - varies
    GENERATOR_ERROR = "generator_error"       # Data generator raised exception
    INFERENCE_ERROR = "inference_error"       # Inference API returned error
    VALIDATION_ERROR = "validation_error"     # Output validation failed
    EXECUTION_ERROR = "execution_error"       # Generic execution failure

    # Job Contract errors - NOT retryable
    PAYLOAD_INVALID = "payload_invalid"       # Required fields missing/wrong type
    PAYLOAD_VERSION_MISMATCH = "payload_version"  # Old payload format

    # Timeout errors - sometimes retryable
    TIMEOUT = "timeout"                       # Exceeded job timeout
    LEASE_EXPIRED = "lease_expired"           # Worker died/lost lease

    # Cancellation - NOT retryable
    CANCELLED = "cancelled"                   # User cancelled
    SUPERSEDED = "superseded"                 # Replaced by newer job

    # Unknown - investigate these!
    UNKNOWN = "unknown"                       # Catch-all

    @property
    def is_retryable(self) -> bool:
        """Whether this error type is generally retryable."""
        return self in {
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.CONNECTION_REFUSED,
            JobErrorCode.INFERENCE_ERROR,
            JobErrorCode.TIMEOUT,
            JobErrorCode.LEASE_EXPIRED,
        }

    @property
    def category(self) -> str:
        """Get the error category for grouping."""
        categories = {
            JobErrorCode.NONE: "success",
            JobErrorCode.TRANSPORT_ERROR: "network",
            JobErrorCode.CONNECTION_REFUSED: "network",
            JobErrorCode.WORKER_SETUP: "setup",
            JobErrorCode.MODEL_NOT_FOUND: "setup",
            JobErrorCode.RESOURCE_UNAVAILABLE: "setup",
            JobErrorCode.GENERATOR_ERROR: "execution",
            JobErrorCode.INFERENCE_ERROR: "execution",
            JobErrorCode.VALIDATION_ERROR: "execution",
            JobErrorCode.EXECUTION_ERROR: "execution",
            JobErrorCode.PAYLOAD_INVALID: "contract",
            JobErrorCode.PAYLOAD_VERSION_MISMATCH: "contract",
            JobErrorCode.TIMEOUT: "timeout",
            JobErrorCode.LEASE_EXPIRED: "timeout",
            JobErrorCode.CANCELLED: "user",
            JobErrorCode.SUPERSEDED: "user",
            JobErrorCode.UNKNOWN: "unknown",
        }
        return categories.get(self, "unknown")


class JobEventType(str, Enum):
    """
    Event types for job audit trail.

    Every state transition is recorded as an event, enabling:
    - Full job history reconstruction
    - Debugging failed jobs
    - Performance analysis
    """
    # Lifecycle events
    CREATED = "created"           # Job submitted
    CLAIMED = "claimed"           # Worker claimed job
    STARTED = "started"           # Execution began
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"             # Failed with error

    # Status changes
    RETRIED = "retried"           # Moved back to pending for retry
    CANCELLED = "cancelled"       # User cancelled
    RELEASED = "released"         # Worker released without completing

    # Timeout events
    LEASE_EXPIRED = "lease_expired"  # Lease timeout, job returned to queue
    TIMEOUT = "timeout"              # Job exceeded time limit

    # Heartbeat/progress (optional, for long-running jobs)
    HEARTBEAT = "heartbeat"       # Worker sent heartbeat
    PROGRESS = "progress"         # Progress update


@dataclass
class JobEvent:
    """
    A single event in a job's history.

    Events are immutable once recorded.
    """
    event_id: int
    job_id: str
    timestamp: datetime
    event_type: JobEventType
    actor: str  # worker_id, "system", "user"
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "actor": self.actor,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class JobError:
    """
    Structured error information for failed jobs.

    Workers report errors as JobError objects, enabling:
    - Consistent error categorization
    - Retry decisions based on error type
    - Error rate analysis by code/category
    """
    code: JobErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: Optional[bool] = None  # Override default retryability

    def __post_init__(self):
        # Default retryable from error code if not specified
        if self.retryable is None:
            self.retryable = self.code.is_retryable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobError":
        return cls(
            code=JobErrorCode(data.get("code", "unknown")),
            message=data.get("message", "Unknown error"),
            details=data.get("details"),
            retryable=data.get("retryable"),
        )

    @classmethod
    def from_exception(cls, e: Exception, code: Optional[JobErrorCode] = None) -> "JobError":
        """Create JobError from an exception."""
        # Classify common exceptions
        if code is None:
            exc_type = type(e).__name__
            if "ConnectionRefused" in exc_type or "ConnectionError" in str(type(e).__mro__):
                code = JobErrorCode.CONNECTION_REFUSED
            elif "Timeout" in exc_type:
                code = JobErrorCode.TIMEOUT
            elif "requests" in str(type(e).__module__):
                code = JobErrorCode.TRANSPORT_ERROR
            elif "FileNotFoundError" in exc_type:
                code = JobErrorCode.MODEL_NOT_FOUND
            elif "MemoryError" in exc_type or "CUDA" in str(e):
                code = JobErrorCode.RESOURCE_UNAVAILABLE
            elif "ValidationError" in exc_type:
                code = JobErrorCode.VALIDATION_ERROR
            else:
                code = JobErrorCode.EXECUTION_ERROR

        return cls(
            code=code,
            message=str(e),
            details={"exception_type": type(e).__name__},
        )


class JobPriority(str, Enum):
    """Priority levels for jobs."""
    CRITICAL = "critical"      # Process immediately
    HIGH = "high"              # High priority
    NORMAL = "normal"          # Normal priority
    LOW = "low"                # Low priority (background)
    IDLE = "idle"              # Only when system is idle


@dataclass
class JobSpec:
    """
    Specification for a job to be submitted.

    Contains all information needed to route and execute a job.
    """
    job_type: JobType
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    timeout: Optional[int] = None  # Override default timeout
    tags: List[str] = field(default_factory=list)
    require_gpu: Optional[bool] = None  # Override auto-detection
    target_device: Optional[str] = None  # Specific device to run on

    def __post_init__(self):
        if self.timeout is None:
            self.timeout = self.job_type.default_timeout

    @property
    def needs_gpu(self) -> bool:
        """Whether this job needs GPU."""
        if self.require_gpu is not None:
            return self.require_gpu
        return self.job_type.requires_gpu

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "timeout": self.timeout,
            "tags": self.tags,
            "require_gpu": self.require_gpu,
            "target_device": self.target_device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        return cls(
            job_type=JobType(data["job_type"]),
            payload=data.get("payload", {}),
            priority=JobPriority(data.get("priority", "normal")),
            timeout=data.get("timeout"),
            tags=data.get("tags", []),
            require_gpu=data.get("require_gpu"),
            target_device=data.get("target_device"),
        )


@dataclass
class JobResult:
    """
    Result from executing a job.

    Returned by workers after job completion.
    """
    job_id: str
    status: JobStatus
    job_type: JobType
    worker_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == JobStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "job_type": self.job_type.value,
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "output": self.output,
            "error": self.error,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobResult":
        return cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            job_type=JobType(data["job_type"]),
            worker_id=data["worker_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds", 0),
            output=data.get("output", {}),
            error=data.get("error"),
            metrics=data.get("metrics", {}),
        )


@dataclass
class Job:
    """
    A job instance with full tracking information.

    Combines spec with runtime tracking data.
    """
    job_id: str
    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[JobResult] = None
    error_code: JobErrorCode = JobErrorCode.NONE

    @classmethod
    def create(cls, spec: JobSpec) -> "Job":
        """Create a new job from a spec."""
        return cls(
            job_id=str(uuid.uuid4())[:8],
            spec=spec,
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the job is in a terminal state."""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }

    @property
    def can_retry(self) -> bool:
        """Whether the job can be retried."""
        return (
            self.status == JobStatus.FAILED
            and self.attempts < self.max_attempts
        )

    def to_dict(self) -> Dict[str, Any]:
        # Support both JobResult and raw dict result (from job store)
        if self.result:
            result_data = self.result.to_dict()
        elif hasattr(self, '_raw_result'):
            result_data = self._raw_result
        else:
            result_data = None

        return {
            "job_id": self.job_id,
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "worker_id": self.worker_id,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "result": result_data,
            "error_code": self.error_code.value,
        }


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def eval_job(
    skill_id: str,
    level: int = 1,
    batch_size: int = 100,
    priority: JobPriority = JobPriority.HIGH,
    # Model identity fields (for run-independent evals)
    hero_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    context_hash: Optional[str] = None,
) -> JobSpec:
    """
    Create an eval job spec.

    If model identity fields are provided, the eval is anchored to
    that specific model version. Otherwise, workers may use the
    current active model (not recommended for reproducibility).
    """
    payload = {
        "skill_id": skill_id,
        "level": level,
        "batch_size": batch_size,
    }
    # Add model identity if provided
    if hero_id:
        payload["hero_id"] = hero_id
    if campaign_id:
        payload["campaign_id"] = campaign_id
    if checkpoint_id:
        payload["checkpoint_id"] = checkpoint_id
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    if context_hash:
        payload["context_hash"] = context_hash

    return JobSpec(
        job_type=JobType.EVAL,
        payload=payload,
        priority=priority,
        tags=["eval", skill_id],
    )


def sparring_job(
    skill_id: str,
    count: int = 100,
    checkpoint: Optional[str] = None,
    priority: JobPriority = JobPriority.HIGH,
    # Model identity fields
    hero_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    context_hash: Optional[str] = None,
) -> JobSpec:
    """
    Create a sparring job spec.

    If model identity fields are provided, the sparring is anchored to
    that specific model version.
    """
    payload = {
        "skill_id": skill_id,
        "count": count,
        "checkpoint": checkpoint,
    }
    # Add model identity if provided
    if hero_id:
        payload["hero_id"] = hero_id
    if campaign_id:
        payload["campaign_id"] = campaign_id
    if checkpoint_id:
        payload["checkpoint_id"] = checkpoint_id
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    if context_hash:
        payload["context_hash"] = context_hash

    return JobSpec(
        job_type=JobType.SPARRING,
        payload=payload,
        priority=priority,
        tags=["sparring", skill_id],
    )


def data_gen_job(
    generator: str,
    count: int = 1000,
    priority: JobPriority = JobPriority.NORMAL,
) -> JobSpec:
    """Create a data generation job spec."""
    return JobSpec(
        job_type=JobType.DATA_GEN,
        payload={
            "generator": generator,
            "count": count,
        },
        priority=priority,
        tags=["data_gen", generator],
    )


def archive_job(
    source_zone: str = "hot",
    target_zone: str = "warm",
    max_count: int = 10,
) -> JobSpec:
    """Create an archive job spec."""
    return JobSpec(
        job_type=JobType.ARCHIVE,
        payload={
            "source_zone": source_zone,
            "target_zone": target_zone,
            "max_count": max_count,
        },
        priority=JobPriority.LOW,
        tags=["archive", f"{source_zone}_to_{target_zone}"],
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("Job Types")
    print("=" * 50)

    for jt in JobType:
        print(f"\n{jt.value}:")
        print(f"  requires_gpu: {jt.requires_gpu}")
        print(f"  requires_inference: {jt.requires_inference}")
        print(f"  default_timeout: {jt.default_timeout}s")

    print("\n\nExample Job Specs:")
    print("-" * 50)

    specs = [
        eval_job("bin", level=5),
        sparring_job("binary", count=100),
        data_gen_job("binary_arithmetic", count=500),
        archive_job("hot", "warm"),
    ]

    for spec in specs:
        print(f"\n{spec.job_type.value}:")
        print(f"  payload: {spec.payload}")
        print(f"  priority: {spec.priority.value}")
        print(f"  needs_gpu: {spec.needs_gpu}")
