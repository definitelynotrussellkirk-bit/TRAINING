# Job System V2 Specification

**Status:** Draft
**Date:** 2025-11-28
**Author:** Claude + Russ

---

## Executive Summary

Evolve the existing job system into a production-grade distributed scheduler. The current foundation (SQLiteJobStore, ClaimingWorker, job types, device registry) is solid. This spec adds:

1. **Structured error taxonomy** - Debuggable failure modes
2. **Job type registry with validation** - Single source of truth for contracts
3. **Worker registration & heartbeats** - Explicit identity and health
4. **Backpressure controls** - Queue limits and guards
5. **Enhanced observability** - Health dashboard and audit trail

**Non-goals:** This is NOT a rewrite. We enhance incrementally.

---

## 1. Invariants (Design Principles)

These rules govern all decisions:

| # | Invariant | Implication |
|---|-----------|-------------|
| 1 | **Jobs are immutable contracts** | After creation, only `status`, `result`, `error_*` fields change. Payload never mutates. |
| 2 | **Single source of truth** | `jobs/registry.py` defines job types, payloads, roles. No duplicates elsewhere. |
| 3 | **Workers are dumb but honest** | Workers report capabilities; VaultKeeper decides routing. Workers don't choose jobs. |
| 4 | **All failure is structured** | Error codes, not strings. Every failure has a category. |
| 5 | **Backpressure is explicit** | System can say "no" or "not right now". Silent pile-up is forbidden. |
| 6 | **Observability by default** | Every state change is auditable. Health is always queryable. |

---

## 2. Error Taxonomy

### 2.1 Error Codes

Replace string errors with structured codes. Add to `guild/job_types.py`:

```python
class JobErrorCode(str, Enum):
    """Structured error categories for debuggable failures."""

    # Success (no error)
    NONE = "none"

    # Transport/Network
    TRANSPORT_ERROR = "transport_error"       # HTTP failures, timeouts to services
    CONNECTION_REFUSED = "connection_refused" # Service unreachable

    # Worker Setup
    WORKER_SETUP = "worker_setup"             # Missing dependencies, config errors
    MODEL_NOT_FOUND = "model_not_found"       # Checkpoint/model doesn't exist
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # GPU OOM, disk full

    # Execution
    GENERATOR_ERROR = "generator_error"       # Data generator raised exception
    INFERENCE_ERROR = "inference_error"       # Inference API returned error
    VALIDATION_ERROR = "validation_error"     # Output validation failed
    EXECUTION_ERROR = "execution_error"       # Generic execution failure

    # Job Contract
    PAYLOAD_INVALID = "payload_invalid"       # Required fields missing/wrong type
    PAYLOAD_VERSION_MISMATCH = "payload_version"  # Old payload format

    # Timeouts
    TIMEOUT = "timeout"                       # Exceeded job timeout
    LEASE_EXPIRED = "lease_expired"           # Worker died/lost lease

    # Cancellation
    CANCELLED = "cancelled"                   # User cancelled
    SUPERSEDED = "superseded"                 # Replaced by newer job

    # Unknown
    UNKNOWN = "unknown"                       # Catch-all (investigate these!)
```

### 2.2 Structured Error Response

Workers report errors as:

```python
@dataclass
class JobError:
    code: JobErrorCode
    message: str                    # Human-readable description
    details: Optional[Dict] = None  # Stack trace, context, etc.
    retryable: bool = False         # Hint for auto-retry logic
```

### 2.3 Retryable vs Terminal

| Error Code | Retryable | Reason |
|------------|-----------|--------|
| TRANSPORT_ERROR | Yes | Transient network issue |
| CONNECTION_REFUSED | Yes | Service may restart |
| INFERENCE_ERROR | Yes | Inference server may recover |
| TIMEOUT | Yes | May succeed with more time |
| WORKER_SETUP | No | Config/dependency issue |
| MODEL_NOT_FOUND | No | Model won't appear magically |
| PAYLOAD_INVALID | No | Fix the caller |
| GENERATOR_ERROR | No | Code bug |
| CANCELLED | No | User intent |

---

## 3. Job Type Registry

### 3.1 Location

New file: `jobs/registry.py`

This is the **single source of truth** for job contracts.

### 3.2 JobTypeConfig

```python
@dataclass
class JobTypeConfig:
    """Configuration for a job type."""

    # Identity
    name: str                       # e.g., "eval", "sparring", "data_gen"
    description: str                # Human-readable description

    # Payload contract
    required_fields: List[str]      # Must be present
    optional_fields: List[str]      # May be present
    payload_version: int            # Schema version (increment on breaking changes)

    # Execution
    default_timeout: int            # Seconds (from guild/job_types.py)
    max_attempts: int               # Retry limit
    retryable_errors: List[JobErrorCode]  # Which errors trigger retry

    # Routing
    allowed_roles: List[str]        # From devices.json roles
    requires_gpu: bool              # Must have GPU

    # Backpressure
    max_pending: int                # Queue limit for this type
    max_running: int                # Concurrent execution limit
    queue_full_policy: str          # "reject" | "warn" | "allow"
```

### 3.3 Registry Definition

```python
JOB_TYPE_REGISTRY: Dict[str, JobTypeConfig] = {

    "eval": JobTypeConfig(
        name="eval",
        description="Run skill evaluation suite on a model",
        required_fields=["skill_id", "level"],
        optional_fields=["batch_size", "model_ref", "checkpoint_step"],
        payload_version=1,
        default_timeout=600,
        max_attempts=2,
        retryable_errors=[
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.INFERENCE_ERROR,
            JobErrorCode.TIMEOUT,
        ],
        allowed_roles=["eval_worker"],
        requires_gpu=False,  # Uses remote inference
        max_pending=50,
        max_running=3,
        queue_full_policy="warn",
    ),

    "sparring": JobTypeConfig(
        name="sparring",
        description="Self-correction sparring session",
        required_fields=["skill_id"],
        optional_fields=["count", "checkpoint", "threshold"],
        payload_version=1,
        default_timeout=1800,
        max_attempts=1,  # Expensive, don't retry
        retryable_errors=[],
        allowed_roles=["eval_worker"],
        requires_gpu=False,
        max_pending=10,
        max_running=1,
        queue_full_policy="reject",
    ),

    "inference": JobTypeConfig(
        name="inference",
        description="Direct inference request",
        required_fields=["prompt"],
        optional_fields=["max_tokens", "temperature", "model_ref"],
        payload_version=1,
        default_timeout=60,
        max_attempts=2,
        retryable_errors=[
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.INFERENCE_ERROR,
        ],
        allowed_roles=["inference", "eval_worker"],
        requires_gpu=False,
        max_pending=100,
        max_running=10,
        queue_full_policy="reject",
    ),

    "data_gen": JobTypeConfig(
        name="data_gen",
        description="Generate training data",
        required_fields=["generator", "count"],
        optional_fields=["skill_id", "level", "output_path"],
        payload_version=1,
        default_timeout=300,
        max_attempts=2,
        retryable_errors=[JobErrorCode.EXECUTION_ERROR],
        allowed_roles=["data_forge"],
        requires_gpu=False,
        max_pending=20,
        max_running=2,
        queue_full_policy="warn",
    ),

    "archive": JobTypeConfig(
        name="archive",
        description="Archive checkpoints to cold storage",
        required_fields=["source_zone", "target_zone"],
        optional_fields=["checkpoint_pattern", "keep_last_n", "dry_run"],
        payload_version=1,
        default_timeout=3600,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["vault_worker"],
        requires_gpu=False,
        max_pending=5,
        max_running=1,
        queue_full_policy="reject",
    ),

    "retention": JobTypeConfig(
        name="retention",
        description="Apply retention policy to zone",
        required_fields=["zone"],
        optional_fields=["policy", "dry_run"],
        payload_version=1,
        default_timeout=600,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["vault_worker"],
        requires_gpu=False,
        max_pending=3,
        max_running=1,
        queue_full_policy="reject",
    ),

    "health_check": JobTypeConfig(
        name="health_check",
        description="System health check",
        required_fields=[],
        optional_fields=["components", "deep"],
        payload_version=1,
        default_timeout=60,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["eval_worker", "data_forge", "vault_worker"],
        requires_gpu=False,
        max_pending=10,
        max_running=5,
        queue_full_policy="allow",
    ),
}
```

### 3.4 Validation Functions

```python
def validate_job_type(job_type: str) -> JobTypeConfig:
    """Get config for job type, raise if unknown."""
    if job_type not in JOB_TYPE_REGISTRY:
        raise ValueError(f"Unknown job type: {job_type}. "
                        f"Valid types: {list(JOB_TYPE_REGISTRY.keys())}")
    return JOB_TYPE_REGISTRY[job_type]


def validate_payload(job_type: str, payload: Dict[str, Any]) -> None:
    """Validate payload against job type contract."""
    config = validate_job_type(job_type)

    missing = [f for f in config.required_fields if f not in payload]
    if missing:
        raise ValueError(
            f"Missing required fields for {job_type}: {missing}. "
            f"Required: {config.required_fields}"
        )

    # Warn about unknown fields (might be typos)
    known = set(config.required_fields + config.optional_fields)
    unknown = [f for f in payload if f not in known]
    if unknown:
        # Log warning but don't fail (forward compatibility)
        pass


def get_allowed_job_types(roles: List[str]) -> List[str]:
    """Get job types a worker with given roles can execute."""
    worker_roles = set(roles)
    return [
        config.name
        for config in JOB_TYPE_REGISTRY.values()
        if set(config.allowed_roles) & worker_roles
    ]
```

---

## 4. Worker Registration

### 4.1 Workers Table

Add to SQLite schema in `jobs/store.py`:

```sql
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,           -- device_id + "." + worker_kind
    device_id TEXT NOT NULL,       -- From devices.json
    worker_kind TEXT NOT NULL,     -- "claiming", "eval", "forge"
    roles_json TEXT NOT NULL,      -- JSON array of roles
    version TEXT,                  -- Worker code version
    hostname TEXT,                 -- Network hostname/IP
    registered_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    last_seen_ip TEXT,
    status TEXT NOT NULL DEFAULT 'online',  -- online, offline, draining
    active_jobs INTEGER DEFAULT 0,
    metadata_json TEXT             -- Extra worker info
);

CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_heartbeat ON workers(last_heartbeat_at);
```

### 4.2 Registration Endpoint

New endpoint in VaultKeeper:

```
POST /api/jobs/workers/register

Request:
{
    "device_id": "macmini_eval_1",
    "worker_kind": "claiming",
    "roles": ["eval_worker", "data_forge"],
    "version": "2025.11.28",
    "hostname": "macmini-eval-1.local"
}

Response (200):
{
    "worker_id": "macmini_eval_1.claiming",
    "registered": true,
    "allowed_job_types": ["eval", "sparring", "data_gen", "health_check"]
}

Response (400):
{
    "error": "Device macmini_eval_1 not in devices.json"
}

Response (403):
{
    "error": "Roles [admin] not allowed for device macmini_eval_1"
}
```

### 4.3 Heartbeat

Workers send heartbeats as part of claim or explicit:

```
POST /api/jobs/workers/heartbeat

Request:
{
    "worker_id": "macmini_eval_1.claiming",
    "active_jobs": 1,
    "status": "online"
}

Response (200):
{
    "acknowledged": true,
    "server_time": "2025-11-28T10:30:00Z"
}
```

### 4.4 Worker Status

```
GET /api/jobs/workers

Response:
{
    "workers": [
        {
            "worker_id": "macmini_eval_1.claiming",
            "device_id": "macmini_eval_1",
            "roles": ["eval_worker", "data_forge"],
            "version": "2025.11.28",
            "status": "online",
            "last_heartbeat": "2025-11-28T10:29:45Z",
            "heartbeat_age_sec": 15,
            "active_jobs": 1,
            "allowed_job_types": ["eval", "sparring", "data_gen", "health_check"]
        }
    ],
    "summary": {
        "total": 3,
        "online": 2,
        "offline": 1,
        "by_role": {
            "eval_worker": 2,
            "data_forge": 1,
            "vault_worker": 0
        }
    }
}
```

### 4.5 ClaimingWorker Changes

Update `workers/claiming_worker.py`:

```python
class ClaimingWorker:
    def __init__(self, config: ClaimingWorkerConfig):
        self.config = config
        self.worker_id = f"{config.device_id}.claiming"
        self.allowed_job_types: List[str] = []

    def startup(self):
        """Register with server on startup."""
        resp = self._register()
        if not resp.get("registered"):
            raise RuntimeError(f"Registration failed: {resp.get('error')}")
        self.allowed_job_types = resp.get("allowed_job_types", [])
        logger.info(f"Registered as {self.worker_id}, "
                   f"can run: {self.allowed_job_types}")

    def _register(self) -> dict:
        return self.client.post("/api/jobs/workers/register", {
            "device_id": self.config.device_id,
            "worker_kind": "claiming",
            "roles": self.config.roles,
            "version": VERSION,
            "hostname": socket.gethostname(),
        })

    def claim_job(self) -> Optional[Job]:
        """Claim next job, sending heartbeat implicitly."""
        # Heartbeat sent with every claim attempt
        return self.client.claim(
            device_id=self.config.device_id,
            roles=self.config.roles,
            lease_duration=self.config.lease_duration,
        )
```

---

## 5. Backpressure

### 5.1 Queue Limits

Enforce limits on job submission:

```python
def check_queue_limits(job_type: str) -> Tuple[bool, str]:
    """Check if queue can accept new job of this type.

    Returns: (can_accept, reason)
    """
    config = JOB_TYPE_REGISTRY[job_type]

    # Count current jobs
    stats = job_store.get_type_stats(job_type)
    pending = stats["pending"]
    running = stats["running"]

    if pending >= config.max_pending:
        if config.queue_full_policy == "reject":
            return False, f"Queue full: {pending}/{config.max_pending} pending"
        # "warn" or "allow" - proceed but note it

    if running >= config.max_running:
        if config.queue_full_policy == "reject":
            return False, f"At capacity: {running}/{config.max_running} running"

    return True, "ok"
```

### 5.2 "No Workers" Guard

Warn when no workers can handle a job type:

```python
def check_worker_availability(job_type: str) -> Tuple[bool, str]:
    """Check if any workers are online for this job type.

    Returns: (workers_available, warning_message)
    """
    config = JOB_TYPE_REGISTRY[job_type]
    required_roles = set(config.allowed_roles)

    # Check for online workers with required roles
    workers = job_store.get_online_workers(max_age_sec=120)
    matching = [
        w for w in workers
        if set(w.roles) & required_roles
    ]

    if not matching:
        return False, (
            f"No online workers for {job_type}. "
            f"Requires roles: {config.allowed_roles}"
        )

    return True, "ok"
```

### 5.3 Submit Response

Updated submit endpoint returns backpressure info:

```
POST /api/jobs

Request:
{
    "type": "eval",
    "payload": {"skill_id": "bin", "level": 5},
    "priority": "normal"
}

Response (201 - Accepted):
{
    "accepted": true,
    "job_id": "a1b2c3d4",
    "queue_position": 12,
    "estimated_wait_sec": 60
}

Response (200 - Accepted with warning):
{
    "accepted": true,
    "job_id": "a1b2c3d4",
    "warning": "no_active_workers",
    "warning_message": "No online workers for eval. Job queued but may not run."
}

Response (429 - Queue Full):
{
    "accepted": false,
    "reason": "queue_full",
    "message": "Eval queue full: 50/50 pending. Try again later.",
    "retry_after_sec": 30
}

Response (400 - Invalid):
{
    "accepted": false,
    "reason": "payload_invalid",
    "message": "Missing required fields: ['skill_id']"
}
```

---

## 6. Job Events (Audit Trail)

### 6.1 Events Table

```sql
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT,                    -- worker_id, "system", "user"
    message TEXT,
    details_json TEXT,

    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);
CREATE INDEX IF NOT EXISTS idx_job_events_time ON job_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_job_events_type ON job_events(event_type);
```

### 6.2 Event Types

```python
class JobEventType(str, Enum):
    CREATED = "created"           # Job submitted
    CLAIMED = "claimed"           # Worker claimed job
    STARTED = "started"           # Execution began
    HEARTBEAT = "heartbeat"       # Worker sent heartbeat
    PROGRESS = "progress"         # Progress update (optional)
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"             # Failed with error
    RETRIED = "retried"           # Moved back to pending for retry
    CANCELLED = "cancelled"       # User cancelled
    LEASE_EXPIRED = "lease_expired"  # Lease timeout
    TIMEOUT = "timeout"           # Job timeout
```

### 6.3 Recording Events

```python
def record_event(
    job_id: str,
    event_type: JobEventType,
    actor: str = "system",
    message: str = None,
    details: dict = None
):
    """Record a job event for audit trail."""
    job_store.insert_event(
        job_id=job_id,
        timestamp=utc_now(),
        event_type=event_type.value,
        actor=actor,
        message=message,
        details_json=json.dumps(details) if details else None,
    )
```

### 6.4 Event Query API

```
GET /api/jobs/{job_id}/events

Response:
{
    "job_id": "a1b2c3d4",
    "events": [
        {
            "timestamp": "2025-11-28T10:00:00Z",
            "event_type": "created",
            "actor": "user",
            "message": "Job submitted via API"
        },
        {
            "timestamp": "2025-11-28T10:00:05Z",
            "event_type": "claimed",
            "actor": "macmini_eval_1.claiming",
            "message": "Job claimed, lease until 10:05:05"
        },
        {
            "timestamp": "2025-11-28T10:00:06Z",
            "event_type": "started",
            "actor": "macmini_eval_1.claiming",
            "message": "Execution started"
        },
        {
            "timestamp": "2025-11-28T10:01:30Z",
            "event_type": "completed",
            "actor": "macmini_eval_1.claiming",
            "message": "Completed successfully",
            "details": {"duration_sec": 84, "result_size": 1234}
        }
    ]
}
```

---

## 7. Enhanced Observability

### 7.1 Health Endpoint

```
GET /api/jobs/health

Response:
{
    "status": "healthy",  // healthy, degraded, unhealthy
    "checks": {
        "database": {"ok": true, "latency_ms": 2},
        "workers": {"ok": true, "online": 2, "offline": 1},
        "queue": {"ok": true, "depth": 12, "oldest_pending_sec": 45}
    },
    "alerts": [],
    "timestamp": "2025-11-28T10:30:00Z"
}

Response (degraded):
{
    "status": "degraded",
    "checks": {
        "database": {"ok": true},
        "workers": {"ok": false, "online": 0, "offline": 3},
        "queue": {"ok": false, "depth": 150, "oldest_pending_sec": 3600}
    },
    "alerts": [
        {"level": "warning", "message": "No online workers"},
        {"level": "warning", "message": "Queue depth exceeds 100"}
    ]
}
```

### 7.2 Summary Endpoint (Enhanced)

```
GET /api/jobs/summary

Response:
{
    "queue": {
        "total_pending": 12,
        "total_running": 3,
        "by_type": {
            "eval": {"pending": 8, "running": 2, "max_pending": 50, "max_running": 3},
            "sparring": {"pending": 2, "running": 1, "max_pending": 10, "max_running": 1},
            "data_gen": {"pending": 2, "running": 0, "max_pending": 20, "max_running": 2}
        }
    },
    "workers": {
        "total": 3,
        "online": 2,
        "by_role": {
            "eval_worker": {"total": 2, "online": 2},
            "data_forge": {"total": 1, "online": 1},
            "vault_worker": {"total": 0, "online": 0}
        }
    },
    "performance_24h": {
        "submitted": 150,
        "completed": 142,
        "failed": 5,
        "cancelled": 3,
        "avg_duration_sec": 95,
        "error_rate": 0.033
    },
    "errors_24h": {
        "by_code": {
            "inference_error": 3,
            "timeout": 1,
            "generator_error": 1
        },
        "by_type": {
            "eval": 4,
            "sparring": 1
        }
    }
}
```

### 7.3 Dashboard Checks

The `/api/jobs/health` endpoint enables traffic-light status:

| Check | Green | Yellow | Red |
|-------|-------|--------|-----|
| Workers | All online | Some offline | None online |
| Queue depth | < 50% max | 50-90% max | > 90% max |
| Error rate (24h) | < 5% | 5-15% | > 15% |
| Oldest pending | < 5 min | 5-30 min | > 30 min |

---

## 8. Database Schema Changes

### 8.1 Jobs Table Updates

Add columns to existing `jobs` table:

```sql
-- Add error_code column
ALTER TABLE jobs ADD COLUMN error_code TEXT DEFAULT 'none';

-- Add payload_version column
ALTER TABLE jobs ADD COLUMN payload_version INTEGER DEFAULT 1;

-- Create index for error analysis
CREATE INDEX IF NOT EXISTS idx_jobs_error_code ON jobs(error_code);
```

### 8.2 Migration Strategy

1. Add new columns with defaults (non-breaking)
2. New code writes error_code on failures
3. Old jobs show error_code='unknown' (acceptable)
4. No data migration needed

---

## 9. API Changes Summary

### 9.1 New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/jobs/workers/register` | Worker registration |
| POST | `/api/jobs/workers/heartbeat` | Worker heartbeat |
| GET | `/api/jobs/workers` | List workers |
| GET | `/api/jobs/health` | Health check |
| GET | `/api/jobs/summary` | Enhanced summary |
| GET | `/api/jobs/{id}/events` | Job event history |

### 9.2 Modified Endpoints

| Endpoint | Change |
|----------|--------|
| POST `/api/jobs` | Returns backpressure info, validates payload |
| POST `/api/jobs/{id}/failed` | Accepts structured error object |
| GET `/api/jobs/stats` | Returns error breakdown |

### 9.3 Backward Compatibility

- All new fields are optional or have defaults
- Old workers continue to work (just won't register)
- Old error strings converted to `UNKNOWN` code
- New endpoints don't break existing clients

---

## 10. Implementation Phases

### Phase 1: Error Taxonomy (2-3 hours)

1. Add `JobErrorCode` enum to `guild/job_types.py`
2. Add `JobError` dataclass
3. Add `error_code` column to jobs table
4. Update `mark_failed()` to accept structured error
5. Update ClaimingWorker to report error codes

**Deliverables:**
- `guild/job_types.py` updated
- `jobs/store.py` updated
- `workers/claiming_worker.py` updated

### Phase 2: Job Registry (2-3 hours)

1. Create `jobs/registry.py` with `JobTypeConfig`
2. Define configs for all job types
3. Add `validate_payload()` function
4. Wire validation into submit endpoint
5. Update `get_allowed_job_types()` in router

**Deliverables:**
- `jobs/registry.py` (new)
- `vault/server.py` submit handler updated

### Phase 3: Worker Registration (3-4 hours)

1. Add `workers` table to schema
2. Add registration endpoint
3. Add heartbeat endpoint
4. Add workers list endpoint
5. Update ClaimingWorker to register on startup
6. Update maintenance worker to mark stale workers offline

**Deliverables:**
- `jobs/store.py` workers table
- `vault/server.py` worker endpoints
- `workers/claiming_worker.py` registration

### Phase 4: Backpressure (2-3 hours)

1. Add queue limit checks to submit
2. Add worker availability check
3. Return backpressure info in response
4. Add `max_pending`/`max_running` to registry

**Deliverables:**
- `jobs/registry.py` limits added
- `vault/server.py` submit updated

### Phase 5: Job Events (2-3 hours)

1. Add `job_events` table
2. Add `record_event()` function
3. Record events on state transitions
4. Add events query endpoint

**Deliverables:**
- `jobs/store.py` events table
- `vault/server.py` events endpoint

### Phase 6: Observability (2-3 hours)

1. Add `/api/jobs/health` endpoint
2. Enhance `/api/jobs/summary`
3. Add error rate calculations
4. Update UI with health indicators

**Deliverables:**
- `vault/server.py` new endpoints
- `tavern/templates/jobs.html` updated

---

## 11. UI Enhancements

### 11.1 Jobs Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  JOBS DASHBOARD                                    [Refresh ↻]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CLUSTER HEALTH: 🟢 Healthy                                     │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Pending  │ │ Running  │ │ Workers  │ │ Errors   │            │
│  │    12    │ │    3     │ │   2/3    │ │   3.3%   │            │
│  │ ▓▓▓▓░░░░ │ │ ▓▓▓░░░░░ │ │  online  │ │  24h     │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  WORKERS                                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ID             │ Roles        │ Status  │ Jobs │ Heartbeat │ │
│  ├────────────────┼──────────────┼─────────┼──────┼───────────┤ │
│  │ macmini.claim  │ eval, forge  │ 🟢 online│  1   │ 15s ago   │ │
│  │ trainer.claim  │ eval, vault  │ 🟢 online│  2   │ 5s ago    │ │
│  │ inference.eval │ eval, infer  │ 🔴 offline│ 0   │ 5m ago    │ │
│  └────────────────┴──────────────┴─────────┴──────┴───────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  JOBS                    [Type ▼] [Status ▼] [Show: 50 ▼]       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ID       │ Type     │ Status    │ Worker    │ Age   │ Err  │ │
│  ├──────────┼──────────┼───────────┼───────────┼───────┼──────┤ │
│  │ a1b2c3d4 │ eval     │ 🟢 running │ macmini   │ 45s   │      │ │
│  │ e5f6g7h8 │ sparring │ 🟡 pending │ -         │ 2m    │      │ │
│  │ i9j0k1l2 │ eval     │ 🔴 failed  │ trainer   │ 5m    │ INF  │ │
│  │ m3n4o5p6 │ eval     │ ✅ complete│ macmini   │ 10m   │      │ │
│  └──────────┴──────────┴───────────┴───────────┴───────┴──────┘ │
│                                                                 │
│  [Submit Eval] [Submit Sparring]                                │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Error Code Tooltips

Show error code meaning on hover:

| Code | Tooltip |
|------|---------|
| `INF` | Inference error - Inference API returned error |
| `TRN` | Transport error - Network failure |
| `GEN` | Generator error - Data generator crashed |
| `TMO` | Timeout - Job exceeded time limit |
| `SET` | Setup error - Worker couldn't initialize |

---

## 12. Future Considerations (Out of Scope)

These are explicitly NOT in this spec but noted for later:

1. **Job dependencies** - Job A waits for Job B
2. **Workflows** - Multi-step job chains
3. **Priority boost** - Aging jobs get priority bump
4. **Rate limiting** - Per-client submission limits
5. **Job deduplication** - Prevent duplicate submissions
6. **Result storage** - Persist results long-term
7. **PostgreSQL migration** - Scale beyond SQLite

---

## 13. Testing Strategy

### 13.1 Unit Tests

```python
# tests/test_job_registry.py
def test_validate_payload_required_fields():
    with pytest.raises(ValueError, match="Missing required"):
        validate_payload("eval", {})

def test_validate_payload_valid():
    validate_payload("eval", {"skill_id": "bin", "level": 5})

def test_get_allowed_job_types():
    types = get_allowed_job_types(["eval_worker"])
    assert "eval" in types
    assert "sparring" in types
    assert "archive" not in types
```

### 13.2 Integration Tests

```python
# tests/test_job_backpressure.py
def test_queue_full_rejection():
    # Fill queue to max
    for _ in range(50):
        client.submit({"type": "eval", "payload": {...}})

    # Next should be rejected
    resp = client.submit({"type": "eval", "payload": {...}})
    assert resp["accepted"] == False
    assert resp["reason"] == "queue_full"
```

---

## Appendix A: Error Code Quick Reference

| Code | Category | Retryable | Action |
|------|----------|-----------|--------|
| `transport_error` | Network | Yes | Check connectivity |
| `connection_refused` | Network | Yes | Check service status |
| `worker_setup` | Setup | No | Fix worker config |
| `model_not_found` | Setup | No | Sync checkpoints |
| `resource_unavailable` | Setup | No | Free resources |
| `generator_error` | Execution | No | Fix generator code |
| `inference_error` | Execution | Yes | Check inference server |
| `validation_error` | Execution | No | Check output format |
| `execution_error` | Execution | Maybe | Investigate |
| `payload_invalid` | Contract | No | Fix caller |
| `timeout` | Timeout | Yes | Increase timeout |
| `lease_expired` | Timeout | Yes | Worker died |
| `cancelled` | User | No | User intent |
| `unknown` | Unknown | No | Investigate! |

---

## Appendix B: Migration Checklist

- [ ] Add `JobErrorCode` enum
- [ ] Add `error_code` column to jobs table
- [ ] Create `jobs/registry.py`
- [ ] Add `workers` table
- [ ] Add `job_events` table
- [ ] Update ClaimingWorker registration
- [ ] Update VaultKeeper submit validation
- [ ] Update VaultKeeper failed handler
- [ ] Add worker endpoints
- [ ] Add events endpoint
- [ ] Add health endpoint
- [ ] Update jobs.html UI
- [ ] Write tests
- [ ] Update CLAUDE.md

---

*End of specification*
