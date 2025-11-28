# Job System Upgrade Plan

**Status:** IMPLEMENTED
**Date:** 2025-11-28
**Goal:** Central job store with lease-based claiming and crash recovery

---

## Executive Summary

The design doc proposes a **pull model** where workers claim jobs from a central server. The current system uses a **push model** where the dispatcher sends jobs directly to workers.

**Recommendation:** Implement a **hybrid approach** that:
1. Adds a central SQLite job store (for persistence and visibility)
2. Supports both push (existing) and pull (new Mac mini workers) models
3. Adds lease-based claiming for crash recovery
4. Integrates with Tavern for UI visibility

---

## Current State Analysis

### What Already Exists (Solid Foundation)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Job Types | `guild/job_types.py` | Complete | 12 job types, priorities, specs, results |
| Job Router | `guild/job_router.py` | Complete | Maps JobType → DeviceRole, tracks status |
| Job Dispatcher | `guild/job_dispatcher.py` | Complete | Push-based submission, HTTP to workers |
| Device Registry | `core/devices.py` | Complete | Device roles, capabilities, config |
| Base Worker | `workers/base_worker.py` | Complete | HTTP server, job handling |
| Eval Worker | `workers/eval_worker.py` | Complete | Handles eval/sparring via SkillEngine |
| Skill Engine | `guild/skills/engine.py` | Complete | Local eval generation, scoring |

### What's Missing

| Gap | Impact | Priority |
|-----|--------|----------|
| Central job store | No persistence on server side | HIGH |
| Lease-based claiming | No crash recovery for workers | HIGH |
| Job visibility API | Can't see all jobs in Tavern | MEDIUM |
| "Eval Now" UI button | Manual CLI workflow | MEDIUM |
| Job history/analytics | No historical data | LOW |

---

## Architecture Decision

### Option A: Full Pull Model (Design Doc)

```
         ┌─────────────┐
         │  Job Store  │
         │  (SQLite)   │
         │  Port 8850  │
         └──────┬──────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐
│ 4090  │  │ 3090  │  │ Mac   │
│Worker │  │Worker │  │ Mini  │
└───────┘  └───────┘  └───────┘
  (poll)     (poll)     (poll)
```

**Pros:** Clean model, good crash recovery
**Cons:** Requires new server, changes existing flow, polling overhead

### Option B: Hybrid Model (Recommended)

```
                    ┌─────────────┐
          ┌────────▶│  Job Store  │◀────────┐
          │         │  (SQLite)   │         │
          │         │  (in Vault) │         │
          │         └──────┬──────┘         │
          │                │                │
    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
    │ Dispatcher │    │  Tavern   │    │   Mac     │
    │  (push)    │    │  (view)   │    │  Workers  │
    └─────┬──────┘    └───────────┘    └────┬──────┘
          │                                  │
          ▼                                  ▼
    ┌───────────┐                      ┌───────────┐
    │   4090    │                      │  claim    │
    │   3090    │ (direct HTTP push)   │  (poll)   │
    └───────────┘                      └───────────┘
```

**Key insight:** Your 4090 and 3090 are always on and reliable. Mac minis may come and go.

**Design:**
1. **Job Store** lives in VaultKeeper server (already running on :8767)
2. **Dispatcher** continues to push to 4090/3090 workers
3. **Mac minis** pull/claim from job store (lease-based)
4. **All jobs** are recorded in the store for visibility

---

## Implementation Plan

### Phase 1: Job Store (Core)

**Files to create:**

```
jobs/
├── __init__.py        # Module exports
├── store.py           # JobStore ABC + SQLiteJobStore
├── schema.sql         # SQLite schema
└── migrations/        # Future schema migrations
```

**Schema:**

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,      -- JSON
    status TEXT NOT NULL,
    result TEXT,                -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    priority TEXT DEFAULT 'normal',
    requested_roles TEXT NOT NULL,  -- JSON array
    target_device_id TEXT,
    claimed_by TEXT,
    lease_expires_at TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type ON jobs(type);
CREATE INDEX idx_jobs_claimed ON jobs(claimed_by);
```

**SQLiteJobStore methods:**

```python
class SQLiteJobStore:
    def submit(self, job: Job) -> Job
    def get(self, job_id: str) -> Optional[Job]
    def update(self, job: Job) -> None

    # Pull model - atomic claim with lease
    def claim_next(
        self,
        device_id: str,
        roles: list[str],
        lease_duration_seconds: int = 300
    ) -> Optional[Job]

    # Status updates
    def mark_running(self, job_id: str, device_id: str) -> None
    def mark_complete(self, job_id: str, result: dict) -> None
    def mark_failed(self, job_id: str, error: str) -> None

    # Queries
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100
    ) -> list[Job]

    # Cleanup
    def expire_stale_leases(self) -> int
    def cleanup_old_jobs(self, max_age_days: int = 7) -> int
```

### Phase 2: VaultKeeper Integration

Add job endpoints to existing VaultKeeper server (`:8767`):

```
/api/jobs                GET    List jobs
/api/jobs                POST   Submit job
/api/jobs/{id}           GET    Get job
/api/jobs/{id}/status    PUT    Update status
/api/jobs/claim          POST   Claim next (for pull model)
/api/jobs/stats          GET    Job statistics
```

**Why VaultKeeper?**
- Already running on 4090
- Already has SQLite experience (catalog.db)
- Already has HTTP API patterns
- No new service to manage

### Phase 3: Dispatcher Integration

Modify `guild/job_dispatcher.py`:

```python
class JobDispatcher:
    def __init__(self, ..., store: Optional[JobStore] = None):
        # Use store for persistence
        self.store = store or get_job_store()

    def submit(self, spec: JobSpec, ...) -> str:
        # 1. Record in store
        job = self.store.submit(Job.from_spec(spec))

        # 2. Route and push (existing logic)
        decision = self.router.route(spec)
        if decision.success:
            self._submit_to_worker(job, decision.worker)

        return job.job_id
```

### Phase 4: Pull Model Worker

Create `workers/claiming_worker.py`:

```python
class ClaimingWorker(BaseWorker):
    """
    Worker that claims jobs from central store (pull model).

    Used for Mac minis that may come and go.
    """

    def __init__(self, store_url: str, ...):
        super().__init__(...)
        self.store_url = store_url  # http://trainer.local:8767
        self.claim_interval = 5     # seconds
        self.lease_duration = 300   # 5 min

    def run(self, ...):
        # Don't start HTTP server - just claim and execute
        while True:
            job = self._claim_next()
            if job:
                self._execute_job(job)
            else:
                time.sleep(self.claim_interval)

    def _claim_next(self) -> Optional[Job]:
        """Claim from central store."""
        response = requests.post(
            f"{self.store_url}/api/jobs/claim",
            json={
                "device_id": self.config.device_id,
                "roles": self.get_supported_roles(),
                "lease_duration": self.lease_duration,
            }
        )
        if response.status_code == 200:
            return Job.from_dict(response.json())
        return None
```

### Phase 5: Tavern UI

Add to Tavern (`tavern/server.py`):

```python
@app.route('/jobs')
def jobs_page():
    """Job queue visibility page."""
    return render_template('jobs.html')

@app.route('/api/jobs')
def api_jobs():
    """Proxy to VaultKeeper jobs API."""
    return requests.get(f"{vault_url}/api/jobs").json()

@app.route('/guild/skills/<skill_id>/eval', methods=['POST'])
def eval_skill_now(skill_id):
    """Submit eval job from UI."""
    level = request.json.get('level', 1)
    batch_size = request.json.get('batch_size', 100)

    spec = eval_job(skill_id, level, batch_size)
    job_id = dispatcher.submit(spec)

    return jsonify({"job_id": job_id})
```

Add `tavern/templates/jobs.html`:

```html
<!-- Job queue table with status, type, worker, etc. -->
<!-- "Eval Now" button for each skill -->
<!-- Auto-refresh every 5 seconds -->
```

---

## Migration Path

### Step 1: Add Store (No Behavior Change)

1. Implement `SQLiteJobStore`
2. Add VaultKeeper endpoints
3. Dispatcher logs jobs to store but doesn't require it
4. Verify jobs appear in Tavern

### Step 2: Add Claiming (Optional Path)

1. Implement `ClaimingWorker`
2. Deploy on one Mac mini
3. Verify claim/execute cycle works
4. Monitor for issues

### Step 3: Full Integration

1. All jobs go through store
2. Mac minis use ClaimingWorker
3. 4090/3090 continue with push model
4. Lease expiration runs periodically

---

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `jobs/__init__.py` | Module exports |
| `jobs/store.py` | JobStore ABC + SQLiteJobStore |
| `jobs/schema.sql` | SQLite schema |
| `workers/claiming_worker.py` | Pull-model worker |
| `tavern/templates/jobs.html` | Jobs UI page |

### Modified Files

| File | Changes |
|------|---------|
| `vault/server.py` | Add /api/jobs endpoints |
| `guild/job_dispatcher.py` | Use JobStore for persistence |
| `tavern/server.py` | Add /jobs route + /guild/skills/.../eval |

### Config Files

| File | Changes |
|------|---------|
| `config/devices.json` | Add Mac mini devices |

---

## Testing Strategy

1. **Unit tests** for SQLiteJobStore
   - Submit, claim, update, expire
   - Concurrent claim safety

2. **Integration tests**
   - Dispatcher → Store → Worker cycle
   - Lease expiration and re-claim

3. **Manual tests**
   - Submit eval job from Tavern
   - Watch Mac mini claim and execute
   - Kill Mac mini, verify job re-claimed

---

## Estimated Work

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Job Store | 2-3 hours | None |
| Phase 2: VaultKeeper | 1-2 hours | Phase 1 |
| Phase 3: Dispatcher | 1 hour | Phase 2 |
| Phase 4: ClaimingWorker | 2 hours | Phase 2 |
| Phase 5: Tavern UI | 2-3 hours | Phase 2 |

**Total:** ~10 hours

---

## Open Questions

1. **Where should jobs.db live?**
   - Option A: With catalog.db in vault/
   - Option B: Separate status/jobs.db
   - **Recommendation:** With catalog.db (both are VaultKeeper's domain)

2. **Should the dispatcher require the store?**
   - Option A: Fail if store unavailable (strict)
   - Option B: Work without store (graceful degradation)
   - **Recommendation:** Option B for now, Option A later

3. **Lease duration?**
   - Eval jobs: 5-10 minutes
   - Sparring jobs: 30-60 minutes
   - **Recommendation:** Make it configurable per job type

4. **Job history retention?**
   - Keep forever? 7 days? 30 days?
   - **Recommendation:** 7 days for completed, 30 days for failed

---

## Next Steps

1. **Review this plan** - are there concerns or changes needed?
2. **Decide on Phase 1 approach** - start with store implementation
3. **Add Mac mini to devices.json** - prepare for testing
