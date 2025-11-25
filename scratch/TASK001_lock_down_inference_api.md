# TASK001: Lock Down Inference API

**Priority:** CRITICAL
**Effort:** 2 hours
**Dependencies:** None
**Files:** `inference/main.py`

---

## Problem

The inference API (`inference/main.py`) exposes powerful endpoints without authentication:

- `/models/register`, `/models/set_active`, `/models/reload` - can change running model
- `/data_gen/jobs`, `/eval/jobs`, `/jobs` - control workload queue
- `/gpu`, `/system` - full GPU + system telemetry
- `/settings/power_profile` - runs `sudo nvidia-smi -pl` via subprocess

If exposed beyond localhost, anyone could:
- Change power limits on GPU
- Flip active models
- Queue arbitrary heavy jobs
- Enumerate system specs

## Solution

Add token-based authentication with role separation.

## Implementation Steps

### Step 1: Add auth middleware
```python
# inference/auth.py
import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
ADMIN_KEY = os.environ.get("INFERENCE_ADMIN_KEY", "")
READ_KEY = os.environ.get("INFERENCE_READ_KEY", "")

def require_admin(api_key: str = Security(API_KEY_HEADER)):
    if not ADMIN_KEY:
        raise HTTPException(500, "INFERENCE_ADMIN_KEY not configured")
    if api_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid or missing admin API key")
    return api_key

def require_read(api_key: str = Security(API_KEY_HEADER)):
    if not READ_KEY and not ADMIN_KEY:
        raise HTTPException(500, "No API keys configured")
    if api_key not in [READ_KEY, ADMIN_KEY]:
        raise HTTPException(401, "Invalid or missing API key")
    return api_key
```

### Step 2: Protect endpoints by role

| Endpoint | Auth Level | Rationale |
|----------|------------|-----------|
| `/health` | None | Health checks need to work |
| `/v1/chat/completions` | `require_read` | Main inference endpoint |
| `/models/info` | `require_read` | Info only |
| `/models/*` (write) | `require_admin` | Model management |
| `/gpu`, `/system` | `require_admin` | System info |
| `/settings/*` | `require_admin` | Power management |
| `/jobs/*` | `require_admin` | Job queue |
| `/eval/*`, `/data_gen/*` | `require_admin` | Heavy operations |

### Step 3: Update main.py

```python
from auth import require_admin, require_read

# Example: protect /models/reload
@app.post("/models/reload", dependencies=[Security(require_admin)])
async def reload_model():
    ...

# Example: protect /v1/chat/completions
@app.post("/v1/chat/completions", dependencies=[Security(require_read)])
async def chat_completions(req: ChatCompletionRequest):
    ...
```

### Step 4: Document in README

Add to `inference/README.md`:
```markdown
## Authentication

Set environment variables before starting:

```bash
export INFERENCE_ADMIN_KEY="your-secret-admin-key"
export INFERENCE_READ_KEY="your-secret-read-key"
```

- Admin key: Required for model management, settings, jobs
- Read key: Required for inference endpoints

**WARNING:** Never expose this API to the public internet without auth.
```

### Step 5: Update prediction_client.py

```python
class PredictionClient:
    def __init__(self, base_url, api_key=None, ...):
        self.api_key = api_key or os.environ.get("INFERENCE_READ_KEY", "")

    def _request_with_retry(self, method, url, **kwargs):
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        ...
```

## Checkpoints

- [x] Create `inference/auth.py` with `require_admin` and `require_read`
- [x] Add `dependencies=[Security(...)]` to all sensitive endpoints
- [x] Update `prediction_client.py` to send API key
- [x] Update `inference/README.md` with auth docs
- [ ] Test: start server without keys → 500 on protected endpoints
- [ ] Test: start server with keys → 401 without header, 200 with header
- [x] Update `deployment_orchestrator.py` to use API key

## Verification

```bash
# Without key - should fail
curl http://192.168.x.x:8765/models/reload -X POST
# Expected: 401 Unauthorized

# With key - should work
curl -H "X-API-Key: $INFERENCE_ADMIN_KEY" http://192.168.x.x:8765/models/reload -X POST
# Expected: 200 OK

# Health check - should always work
curl http://192.168.x.x:8765/health
# Expected: 200 OK
```

## Rollback

If auth breaks things:
1. Remove `dependencies=[...]` from endpoints
2. Restart server
3. Debug auth module separately
