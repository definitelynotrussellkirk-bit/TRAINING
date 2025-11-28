# Heterogeneous Cluster Implementation Plan

**Status:** Planning
**Created:** 2025-11-28

---

## Overview

Transform the job system from "any worker can do any matching role" to a smart routing system that treats the 4090, 3090, and Mac minis as distinct citizens with capabilities, priorities, and load awareness.

### Current State

- **Role-based routing only** - Jobs match workers by role strings
- **No resource awareness** - A Mac mini and RTX 4090 are treated equally for `eval_worker` jobs
- **No load balancing** - First worker to claim wins, regardless of capacity
- **No priority routing** - Critical training boxes do analytics when training is behind

### Target State

- **Resource-class routing** - Jobs prefer/require specific device tiers
- **Capability matching** - Fine-grained job→worker matching
- **Load-aware claiming** - Consider worker's current load in routing
- **Cluster modes** - "catch-up mode" protects training, "idle mode" enables analytics

---

## Implementation Phases

### HC-01: Device & Worker Capabilities

**Goal:** Extend device registry and worker registration to capture resource classes and capabilities.

#### 1.1 Extend `config/devices.json`

Add new fields to each device:

```json
{
  "trainer4090": {
    "hostname": "192.168.x.x",
    "description": "Main training box - RTX 4090",
    "roles": ["trainer", "eval_worker", "storage_hot", "control_plane"],
    "gpus": [{"name": "RTX 4090", "count": 1, "vram_gb": 24}],

    "resource_class": "gpu_heavy",
    "priority_class": "critical",
    "max_concurrent_jobs": 2,
    "capabilities": ["cuda_12", "flash_attn", "bf16", "training"],

    "cpu": {"cores": 16, "threads": 32},
    "memory_gb": 64,
    "enabled": true
  },
  "inference3090": {
    "hostname": "192.168.x.x",
    "resource_class": "gpu_medium",
    "priority_class": "support",
    "max_concurrent_jobs": 3,
    "capabilities": ["cuda_12", "inference", "analytics"],
    ...
  },
  "macmini_eval_1": {
    "hostname": "macmini-eval-1.local",
    "resource_class": "cpu_light",
    "priority_class": "auxiliary",
    "max_concurrent_jobs": 4,
    "capabilities": ["data_processing", "cpu_inference"],
    ...
  },
  "r730xd": {
    "hostname": "r730xd.local",
    "resource_class": "cpu_heavy",
    "priority_class": "support",
    "max_concurrent_jobs": 16,
    "capabilities": ["data_processing", "bulk_storage", "parallel_cpu"],
    "cpu": {"cores": 32, "threads": 64},
    "memory_gb": 188,
    "storage_tb": 40,
    ...
  }
}
```

**New Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `resource_class` | string | Device tier: `gpu_heavy`, `gpu_medium`, `cpu_light` |
| `priority_class` | string | Scheduling priority: `critical`, `support`, `auxiliary` |
| `max_concurrent_jobs` | int | Max jobs this device should run simultaneously |
| `capabilities` | string[] | Fine-grained capability tags |

#### 1.2 Extend Worker Registration

**File:** `workers/claiming_worker.py`

Update registration payload to include hardware info:

```python
registration_data = {
    "device_id": self.device_id,
    "worker_kind": self.worker_kind,
    "roles": self.roles,
    "version": self.version,
    "hostname": socket.gethostname(),
    # NEW FIELDS
    "reported_hardware": {
        "gpu_name": get_gpu_name(),      # "RTX 4090" or None
        "vram_gb": get_vram_gb(),        # 24 or None
        "cuda_version": get_cuda_version(),  # "12.1" or None
        "memory_gb": get_system_memory_gb(),
    },
    "max_concurrent_jobs": self.max_concurrent_jobs,  # from device config
}
```

#### 1.3 Extend Workers Table Schema

**File:** `jobs/store.py`

Add columns to workers table:

```sql
ALTER TABLE workers ADD COLUMN resource_class TEXT;
ALTER TABLE workers ADD COLUMN priority_class TEXT;
ALTER TABLE workers ADD COLUMN max_concurrent_jobs INTEGER DEFAULT 1;
ALTER TABLE workers ADD COLUMN capabilities_json TEXT;  -- JSON array
ALTER TABLE workers ADD COLUMN reported_hardware_json TEXT;  -- JSON object
```

Or for new installations, update CREATE TABLE:

```sql
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
    -- NEW COLUMNS
    resource_class TEXT,
    priority_class TEXT,
    max_concurrent_jobs INTEGER DEFAULT 1,
    capabilities_json TEXT,
    reported_hardware_json TEXT,
    metadata_json TEXT
);
```

#### 1.4 Update Worker Registration Endpoint

**File:** `vault/server.py` - `/api/jobs/workers/register`

On registration:
1. Validate device_id exists
2. Validate roles ⊆ device.roles
3. **NEW:** Copy `resource_class`, `priority_class`, `max_concurrent_jobs`, `capabilities` from device config
4. Store `reported_hardware` for observability
5. Return enriched worker info

#### 1.5 Update `/api/jobs/workers` Response

Include new fields:

```json
{
  "workers": [
    {
      "worker_id": "trainer4090.claiming",
      "device_id": "trainer4090",
      "resource_class": "gpu_heavy",
      "priority_class": "critical",
      "max_concurrent_jobs": 2,
      "active_jobs": 1,
      "capabilities": ["cuda_12", "flash_attn", "bf16", "training"],
      "reported_hardware": {"gpu_name": "RTX 4090", "vram_gb": 24},
      ...
    }
  ]
}
```

#### Deliverables

- [ ] Update `config/devices.json` schema with resource_class, priority_class, max_concurrent_jobs, capabilities
- [ ] Add hardware detection helpers in `workers/hardware.py`
- [ ] Update `workers/claiming_worker.py` registration payload
- [ ] Add migration for workers table (new columns)
- [ ] Update `jobs/store.py` - `register_worker()` to store new fields
- [ ] Update `vault/server.py` - `/api/jobs/workers/register` endpoint
- [ ] Update `vault/server.py` - `/api/jobs/workers` to return new fields

---

### HC-02: JobTypeConfig Routing Metadata

**Goal:** Extend job type definitions with resource requirements and preferences.

#### 2.1 Extend JobTypeConfig

**File:** `jobs/registry.py`

```python
@dataclass
class JobTypeConfig:
    # Existing fields...
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    default_timeout: int = 300
    max_attempts: int = 3
    retryable_errors: List[JobErrorCode] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    requires_gpu: bool = False
    max_pending: int = 100
    max_running: int = 10
    queue_full_policy: str = "warn"

    # NEW ROUTING FIELDS
    resource_intensity: str = "medium"  # "light", "medium", "heavy"
    job_priority_class: str = "normal"  # "critical", "high", "normal", "low"
    preferred_resource_classes: List[str] = field(default_factory=list)  # ["gpu_heavy", "gpu_medium"]
    required_resource_classes: List[str] = field(default_factory=list)   # Must be one of these
    forbidden_roles: List[str] = field(default_factory=list)             # Never route to these
    required_capabilities: List[str] = field(default_factory=list)       # Worker must have all
    min_vram_gb: Optional[int] = None                                    # Minimum VRAM required
```

#### 2.2 Update Job Type Definitions

**File:** `jobs/registry.py` - `_register_job_types()`

```python
# Training (critical, GPU-heavy only)
JobTypeConfig(
    name="train_step",
    description="Execute training step",
    allowed_roles=["trainer"],
    requires_gpu=True,
    resource_intensity="heavy",
    job_priority_class="critical",
    required_resource_classes=["gpu_heavy"],
    required_capabilities=["training", "cuda_12"],
    min_vram_gb=20,
    ...
)

# Quick eval (can run anywhere with GPU or strong CPU)
JobTypeConfig(
    name="eval",
    description="Run skill evaluation",
    allowed_roles=["eval_worker"],
    requires_gpu=False,
    resource_intensity="light",
    job_priority_class="high",
    preferred_resource_classes=["gpu_heavy", "gpu_medium", "cpu_light"],
    ...
)

# Heavy eval (needs GPU)
JobTypeConfig(
    name="eval_full",
    description="Full evaluation suite",
    allowed_roles=["eval_worker"],
    requires_gpu=True,
    resource_intensity="heavy",
    job_priority_class="normal",
    preferred_resource_classes=["gpu_heavy", "gpu_medium"],
    min_vram_gb=8,
    ...
)

# Layer stats (analytics, GPU-heavy but low priority)
JobTypeConfig(
    name="layer_stats",
    description="Model archaeology analysis",
    allowed_roles=["analytics"],
    requires_gpu=True,
    resource_intensity="heavy",
    job_priority_class="low",
    preferred_resource_classes=["gpu_medium"],  # Prefer 3090
    forbidden_roles=["trainer"],  # Don't use training box
    min_vram_gb=16,
    ...
)

# Data validation (CPU is fine)
JobTypeConfig(
    name="data_validate",
    description="Validate data shards",
    allowed_roles=["data_forge"],
    requires_gpu=False,
    resource_intensity="light",
    job_priority_class="normal",
    preferred_resource_classes=["cpu_light", "gpu_medium"],
    ...
)
```

#### 2.3 Add Helper Functions

**File:** `jobs/registry.py`

```python
def get_job_types_for_resource_class(resource_class: str) -> List[str]:
    """Return job types that prefer or accept this resource class."""
    result = []
    for jt in JOB_TYPE_CONFIGS.values():
        if resource_class in jt.required_resource_classes:
            result.append(jt.name)
        elif resource_class in jt.preferred_resource_classes:
            result.append(jt.name)
        elif not jt.required_resource_classes and not jt.preferred_resource_classes:
            # No preference = accept all
            result.append(jt.name)
    return result

def worker_can_run_job_type(worker: dict, job_type: str) -> tuple[bool, str]:
    """Check if worker can run job type. Returns (can_run, reason)."""
    config = JOB_TYPE_CONFIGS.get(job_type)
    if not config:
        return False, "unknown_job_type"

    # Check roles
    worker_roles = set(worker.get("roles", []))
    if not worker_roles.intersection(config.allowed_roles):
        return False, "role_mismatch"

    # Check forbidden roles
    if worker_roles.intersection(config.forbidden_roles):
        return False, "forbidden_role"

    # Check required resource class
    if config.required_resource_classes:
        if worker.get("resource_class") not in config.required_resource_classes:
            return False, "resource_class_required"

    # Check capabilities
    worker_caps = set(worker.get("capabilities", []))
    required_caps = set(config.required_capabilities)
    if not required_caps.issubset(worker_caps):
        missing = required_caps - worker_caps
        return False, f"missing_capabilities:{','.join(missing)}"

    # Check VRAM
    if config.min_vram_gb:
        worker_vram = worker.get("reported_hardware", {}).get("vram_gb", 0)
        if worker_vram < config.min_vram_gb:
            return False, f"insufficient_vram:{worker_vram}<{config.min_vram_gb}"

    return True, "ok"
```

#### Deliverables

- [ ] Extend `JobTypeConfig` dataclass with new routing fields
- [ ] Update all 17 job type definitions with appropriate routing metadata
- [ ] Add `worker_can_run_job_type()` helper
- [ ] Add `get_job_types_for_resource_class()` helper
- [ ] Update `/api/jobs/types` endpoint to return new fields

---

### HC-03: Smart Routing Logic

**Goal:** Replace simple role-matching with resource-aware, load-balanced routing.

#### 3.1 Implement `ordered_job_types_for_worker()`

**File:** `jobs/routing.py` (new file)

```python
from typing import List, Dict, Optional
from jobs.registry import JOB_TYPE_CONFIGS, worker_can_run_job_type

def ordered_job_types_for_worker(
    worker: dict,
    cluster_mode: str = "normal",
    queue_depths: Optional[Dict[str, int]] = None
) -> List[str]:
    """
    Return job types this worker should try to claim, in priority order.

    Considers:
    - Worker's resource_class and capabilities
    - Worker's priority_class (critical workers prioritize critical jobs)
    - Cluster mode (catch_up, normal, idle)
    - Current queue depths (prefer jobs with backlog)
    """
    candidate_types = []

    for job_type, config in JOB_TYPE_CONFIGS.items():
        can_run, reason = worker_can_run_job_type(worker, job_type)
        if can_run:
            candidate_types.append(job_type)

    # Score each job type for this worker
    def score_job_type(job_type: str) -> tuple:
        config = JOB_TYPE_CONFIGS[job_type]

        # Priority score (lower = higher priority)
        priority_map = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        priority_score = priority_map.get(config.job_priority_class, 2)

        # Preference score (lower = better fit)
        worker_rc = worker.get("resource_class", "")
        if worker_rc in config.required_resource_classes:
            preference_score = 0  # Perfect match
        elif worker_rc in config.preferred_resource_classes:
            idx = config.preferred_resource_classes.index(worker_rc)
            preference_score = idx + 1  # Preference order matters
        else:
            preference_score = 10  # Fallback

        # Cluster mode adjustments
        if cluster_mode == "catch_up":
            # In catch-up mode, deprioritize low-priority jobs for critical workers
            if worker.get("priority_class") == "critical" and priority_score >= 2:
                priority_score += 10  # Push to end
        elif cluster_mode == "idle":
            # In idle mode, all jobs are fair game
            pass

        # Queue depth bonus (prefer jobs with backlog)
        backlog_bonus = 0
        if queue_depths:
            depth = queue_depths.get(job_type, 0)
            if depth > 10:
                backlog_bonus = -1  # Slight priority boost

        return (priority_score + backlog_bonus, preference_score, job_type)

    # Sort by score
    candidate_types.sort(key=score_job_type)

    return candidate_types
```

#### 3.2 Implement Cluster Mode Detection

**File:** `jobs/routing.py`

```python
def compute_cluster_mode(store: "SQLiteJobStore") -> str:
    """
    Determine current cluster mode based on queue state.

    Returns:
        "catch_up" - Training backlog is high, protect critical workers
        "idle" - No training jobs, analytics can use all resources
        "normal" - Balanced operation
    """
    stats = store.get_queue_stats()

    # Check training backlog
    train_pending = stats.get("train_step", {}).get("pending", 0)
    train_running = stats.get("train_step", {}).get("running", 0)

    # Estimate backlog in seconds (assume ~1 sec per step)
    train_backlog_seconds = train_pending

    if train_backlog_seconds > 60:  # More than 1 minute of training queued
        return "catch_up"

    if train_pending == 0 and train_running == 0:
        # Check if any critical jobs pending
        critical_pending = sum(
            stats.get(jt, {}).get("pending", 0)
            for jt, cfg in JOB_TYPE_CONFIGS.items()
            if cfg.job_priority_class == "critical"
        )
        if critical_pending == 0:
            return "idle"

    return "normal"
```

#### 3.3 Update `claim_next()` Logic

**File:** `jobs/store.py`

```python
def claim_next(
    self,
    worker_id: str,
    worker: dict,  # Full worker info including resource_class, capabilities
    lease_duration: int = 300,
    cluster_mode: Optional[str] = None,
) -> Optional[Job]:
    """
    Claim the next appropriate job for this worker.

    Uses ordered_job_types_for_worker() to determine claim priority.
    """
    from jobs.routing import ordered_job_types_for_worker, compute_cluster_mode

    if cluster_mode is None:
        cluster_mode = compute_cluster_mode(self)

    # Get queue depths for routing decisions
    queue_depths = {
        jt: stats.get("pending", 0)
        for jt, stats in self.get_queue_stats().items()
    }

    # Get ordered job types for this worker
    ordered_types = ordered_job_types_for_worker(
        worker=worker,
        cluster_mode=cluster_mode,
        queue_depths=queue_depths,
    )

    # Try to claim in order
    for job_type in ordered_types:
        job = self._claim_one_job_type(
            worker_id=worker_id,
            job_type=job_type,
            lease_duration=lease_duration,
        )
        if job:
            # Log routing decision
            self._log_event(
                job.job_id,
                "routing_decision",
                {
                    "worker_id": worker_id,
                    "resource_class": worker.get("resource_class"),
                    "cluster_mode": cluster_mode,
                    "job_type": job_type,
                    "ordered_types": ordered_types[:5],  # First 5 for brevity
                }
            )
            return job

    return None
```

#### 3.4 Update `/api/jobs/claim` Endpoint

**File:** `vault/server.py`

```python
@app.route("/api/jobs/claim", methods=["POST"])
def claim_job():
    data = request.json or {}
    worker_id = data.get("worker_id")

    if not worker_id:
        return jsonify({"error": "worker_id required"}), 400

    # Get full worker info
    worker = job_store.get_worker(worker_id)
    if not worker:
        return jsonify({"error": "worker not registered"}), 404

    # Check load limits
    if worker["active_jobs"] >= worker.get("max_concurrent_jobs", 1):
        return jsonify({
            "claimed": False,
            "message": "worker at capacity",
            "active_jobs": worker["active_jobs"],
            "max_concurrent_jobs": worker.get("max_concurrent_jobs", 1),
        })

    lease_duration = data.get("lease_duration", 300)

    job = job_store.claim_next(
        worker_id=worker_id,
        worker=worker,
        lease_duration=lease_duration,
    )

    if job:
        return jsonify({"claimed": True, "job": job.to_dict()})
    else:
        return jsonify({"claimed": False, "message": "no suitable jobs available"})
```

#### Deliverables

- [ ] Create `jobs/routing.py` with `ordered_job_types_for_worker()`
- [ ] Implement `compute_cluster_mode()`
- [ ] Update `jobs/store.py` - `claim_next()` to use smart routing
- [ ] Add load checking in `/api/jobs/claim`
- [ ] Add routing decision logging to job_events

---

### HC-04: Worker Insights & Cluster Dashboard

**Goal:** Provide visibility into cluster state, worker performance, and routing decisions.

#### 4.1 Add Worker Stats Table

**File:** `jobs/store.py`

```sql
CREATE TABLE IF NOT EXISTS worker_stats (
    worker_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    window_start TEXT NOT NULL,  -- ISO timestamp, hourly buckets
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0,
    total_duration_sec REAL DEFAULT 0,
    avg_duration_sec REAL DEFAULT 0,
    PRIMARY KEY (worker_id, job_type, window_start)
);
```

#### 4.2 Add Stats Aggregation

**File:** `jobs/store.py`

```python
def aggregate_worker_stats(self, window_hours: int = 24):
    """Aggregate job completion stats per worker into worker_stats table."""
    # ... implementation
```

#### 4.3 Extend `/api/jobs/workers` with Stats

```json
{
  "workers": [
    {
      "worker_id": "trainer4090.claiming",
      "device_id": "trainer4090",
      "resource_class": "gpu_heavy",
      "priority_class": "critical",
      "max_concurrent_jobs": 2,
      "active_jobs": 1,
      "status": "online",
      "job_stats_24h": {
        "train_step": {"completed": 5000, "failed": 2, "avg_duration_sec": 0.9},
        "eval": {"completed": 120, "failed": 1, "avg_duration_sec": 10.0}
      }
    }
  ]
}
```

#### 4.4 Add `/api/jobs/cluster` Endpoint

**File:** `vault/server.py`

```python
@app.route("/api/jobs/cluster", methods=["GET"])
def get_cluster_status():
    """Return cluster-wide status and queue depths."""
    stats = job_store.get_queue_stats()
    cluster_mode = compute_cluster_mode(job_store)
    workers = job_store.get_all_workers()

    return jsonify({
        "mode": cluster_mode,
        "queue_depths": stats,
        "workers_summary": {
            "total": len(workers),
            "online": sum(1 for w in workers if w["status"] == "online"),
            "by_resource_class": count_by_key(workers, "resource_class"),
            "total_capacity": sum(w.get("max_concurrent_jobs", 1) for w in workers),
            "total_active": sum(w.get("active_jobs", 0) for w in workers),
        },
        "health": {
            "training_backlog_steps": stats.get("train_step", {}).get("pending", 0),
            "eval_backlog": stats.get("eval", {}).get("pending", 0),
            "analytics_paused": cluster_mode == "catch_up",
        }
    })
```

#### 4.5 Tavern Cluster Dashboard

**File:** `tavern/templates/cluster.html` (new)

Layout:
```
┌─────────────────────────────────────────────────────────────┐
│  CLUSTER STATUS                                    [normal] │
├─────────────────────────────────────────────────────────────┤
│  Training: 0 pending, 2 running     [████████░░] 80%       │
│  Eval:     8 pending, 1 running     [██░░░░░░░░] 20%       │
│  Analytics: 3 pending, 1 running    [█░░░░░░░░░] 10%       │
│  Data:     20 pending, 4 running    [████░░░░░░] 40%       │
├─────────────────────────────────────────────────────────────┤
│  WORKERS                                                    │
├──────────────┬────────────┬────────┬──────────┬────────────┤
│  Device      │  Class     │ Status │ Load     │ 24h Jobs   │
├──────────────┼────────────┼────────┼──────────┼────────────┤
│  trainer4090 │  gpu_heavy │ online │ 2/2      │ 5,122      │
│  inference3090│ gpu_medium│ online │ 1/3      │ 847        │
│  macmini_1   │  cpu_light │ online │ 3/4      │ 1,203      │
└──────────────┴────────────┴────────┴──────────┴────────────┘
```

#### Deliverables

- [ ] Add `worker_stats` table and aggregation
- [ ] Extend `/api/jobs/workers` with per-worker job stats
- [ ] Add `/api/jobs/cluster` endpoint
- [ ] Create `tavern/templates/cluster.html`
- [ ] Add route in `tavern/server.py`
- [ ] Add link in Tavern navigation

---

### HC-05: Battle Log Integration

**Goal:** Make cluster events visible in the Battle Log.

#### 5.1 Cluster Mode Events

When cluster mode changes:

```python
battle_log.log_event(
    event_type="cluster_mode",
    message=f"Cluster entered {new_mode} mode",
    data={
        "previous_mode": old_mode,
        "new_mode": new_mode,
        "reason": reason,  # e.g., "training_backlog_high"
    }
)
```

**Battle Log display:**

```
⚙️ Cluster entered catch-up mode (training backlog: 120 steps)
🌙 Cluster idle; analytics jobs enabled on GPUs
⚡ Cluster back to normal mode
```

#### 5.2 Notable Routing Events

```python
# When a non-primary worker picks up a job it normally wouldn't
battle_log.log_event(
    event_type="routing_assist",
    message=f"3090 assisting with eval (4090 busy)",
    data={
        "worker_id": "inference3090.claiming",
        "job_type": "eval",
        "reason": "primary_at_capacity",
    }
)
```

#### Deliverables

- [ ] Add cluster mode change detection and logging
- [ ] Add notable routing event logging
- [ ] Update Battle Log UI to render cluster events

---

## File Changes Summary

| File | Changes |
|------|---------|
| `config/devices.json` | Add resource_class, priority_class, max_concurrent_jobs, capabilities |
| `workers/hardware.py` | New file: hardware detection helpers |
| `workers/claiming_worker.py` | Send hardware info during registration |
| `jobs/registry.py` | Extend JobTypeConfig, update all job type definitions |
| `jobs/routing.py` | New file: smart routing logic |
| `jobs/store.py` | Update workers table schema, claim_next(), add worker_stats |
| `vault/server.py` | Update /api/jobs/claim, /api/jobs/workers, add /api/jobs/cluster |
| `tavern/templates/cluster.html` | New file: cluster dashboard |
| `tavern/server.py` | Add /cluster route |
| `core/battle_log.py` | Add cluster event types |

---

## Implementation Order

```
HC-01 (Device & Worker Capabilities)
  ↓
HC-02 (JobTypeConfig Routing Metadata)
  ↓
HC-03 (Smart Routing Logic)
  ↓
HC-04 (Worker Insights & Dashboard)
  ↓
HC-05 (Battle Log Integration)
```

Each phase can be shipped independently. HC-01 and HC-02 are pure schema/config changes with no behavior change. HC-03 activates the smart routing. HC-04 and HC-05 are visibility features.

---

## Testing Strategy

1. **Unit tests for routing logic**
   - `test_ordered_job_types_for_worker()` with different worker configs
   - `test_compute_cluster_mode()` with various queue states

2. **Integration tests**
   - Register workers with different resource classes
   - Submit jobs with different requirements
   - Verify routing decisions

3. **Manual testing**
   - Start workers on each machine
   - Submit mixed job load
   - Verify cluster dashboard shows correct state
   - Verify Battle Log shows mode changes

---

## Rollback Plan

All changes are additive:
- New columns use defaults (existing data works)
- New JobTypeConfig fields use defaults (existing job types work)
- claim_next() falls back to role-based if worker lacks new fields

To rollback: revert code, new columns become unused but harmless.
