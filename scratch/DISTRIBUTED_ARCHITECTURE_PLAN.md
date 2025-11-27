# Distributed Architecture Plan

**Goal:** Abstract all location dependencies so any component can run anywhere.

## Current State (Tightly Coupled)

```
4090 (Training Machine) - Everything runs here
├── Tavern UI (localhost:8888)
├── VaultKeeper (localhost:8767)
├── Training Daemon
├── Ledger (local file: status/checkpoint_ledger.json)
├── Checkpoints (local: current_model/)
└── All status files (local: status/)

3090 (Inference) - Only inference
├── Inference Server (8765)
└── Checkpoints (synced copies)
```

**Problems:**
- Ledger read via local file, not API
- Training status read via local file
- Tavern assumes it's on the training machine
- Hardcoded paths throughout

## Target State (Loosely Coupled)

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROLLER                                 │
│              (Can run on ANY machine)                           │
│                                                                 │
│  Tavern UI - Queries everything via HTTP APIs                   │
│  No local file access except its own templates/static           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   config/hosts.json       │
                    │   (Service Discovery)     │
                    └─────────────┬─────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
     ▼                            ▼                            ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│  TRAINER    │          │  ORACLE     │          │  ARCHIVE    │
│  (4090)     │          │  (3090)     │          │  (NAS)      │
│             │          │             │          │             │
│ VaultKeeper │          │ Inference   │          │ Branch      │
│  :8767      │          │  :8765      │          │ Officer     │
│             │          │             │          │  :8768      │
│ Endpoints:  │          │ Endpoints:  │          │             │
│ /api/ledger │          │ /v1/chat    │          │ /assets     │
│ /api/vault  │          │ /models     │          │ /status     │
│ /api/train  │          │             │          │             │
│ /api/zones  │          │ Branch      │          │             │
│             │          │ Officer     │          │             │
│ Branch      │          │  :8768      │          │             │
│ Officer     │          │             │          │             │
│  :8768      │          │             │          │             │
└─────────────┘          └─────────────┘          └─────────────┘
```

## Implementation Phases

### Phase 1: Host Registry & Service Discovery
**Files:** `core/hosts.py`, `config/hosts.json`

Create a proper service registry that:
- Defines all known hosts and their capabilities
- Provides discovery for services
- Supports health checks
- Is the SINGLE source of truth for "where is X?"

```python
from core.hosts import get_host, get_service_url

# Get training host
trainer = get_host("trainer")  # Returns host config

# Get service URL
ledger_url = get_service_url("ledger")  # "http://192.168.x.x:8767/api/ledger"
inference_url = get_service_url("inference")  # "http://192.168.x.x:8765"
```

### Phase 2: Ledger API on VaultKeeper
**Files:** `vault/server.py` (add endpoints)

Add ledger endpoints to VaultKeeper so ledger is queryable via API:
- `GET /api/ledger` - List checkpoints
- `GET /api/ledger/{step}` - Get checkpoint
- `GET /api/ledger/best` - Best by metric
- `GET /api/ledger/summary` - Summary stats
- `POST /api/ledger/record` - Record new checkpoint (internal)

### Phase 3: Remote Ledger Client
**Files:** `core/checkpoint_ledger.py` (extend)

Add `RemoteLedgerClient` that implements same interface as `CheckpointLedger`:

```python
# Local (when on training host)
ledger = get_ledger()

# Remote (when querying from elsewhere)
ledger = RemoteLedgerClient("http://192.168.x.x:8767")

# Auto-detect (check if local, fallback to remote)
ledger = get_ledger(auto_remote=True)
```

### Phase 4: Training Status API
**Files:** `vault/server.py` or new `core/training_api.py`

Expose training status via API:
- `GET /api/training/status` - Current training state
- `GET /api/training/queue` - Queue status
- `POST /api/training/control` - Pause/resume/stop

### Phase 5: Update Tavern to Use APIs
**Files:** `tavern/server.py`

Change Tavern from local file reads to API calls:
- Ledger: Query VaultKeeper API
- Training status: Query training API
- Vault assets: Query VaultKeeper API
- Zones: Already uses API

### Phase 6: Branch Officer Ledger Sync
**Files:** `vault/branch_officer.py`, `vault/zones.py`

Branch Officers sync ledger data:
- Pull ledger from central (4090)
- Cache locally for fast queries
- Report local checkpoints back to central

## Detailed Task Breakdown

### Task 1: Create core/hosts.py
- [ ] Define HostConfig dataclass
- [ ] Define ServiceType enum (TRAINER, INFERENCE, STORAGE, VAULT, LEDGER)
- [ ] Create HostRegistry class
- [ ] Load from config/hosts.json
- [ ] Provide get_host(), get_service_url() helpers
- [ ] Add health check methods

### Task 2: Update config/hosts.json
- [ ] Add service definitions per host
- [ ] Add capabilities list
- [ ] Add API endpoints
- [ ] Add default ports

### Task 3: Add Ledger API to VaultKeeper
- [ ] Add /api/ledger endpoint
- [ ] Add /api/ledger/{step} endpoint
- [ ] Add /api/ledger/best endpoint
- [ ] Add /api/ledger/summary endpoint
- [ ] Wire up to CheckpointLedger

### Task 4: Create RemoteLedgerClient
- [ ] Implement same interface as CheckpointLedger
- [ ] HTTP client for remote queries
- [ ] Caching for performance
- [ ] Fallback logic (local -> remote)

### Task 5: Add Training Status API
- [ ] Create /api/training/status endpoint
- [ ] Read from status/training_status.json
- [ ] Add queue info
- [ ] Add control endpoints (pause/resume)

### Task 6: Update Tavern for Remote Operation
- [ ] Use host registry for URLs
- [ ] Query APIs instead of reading files
- [ ] Handle connection errors gracefully
- [ ] Show "remote mode" indicator in UI

### Task 7: Branch Officer Ledger Sync
- [ ] Add /ledger endpoint to Branch Officer
- [ ] Sync mechanism (pull from central)
- [ ] Report local checkpoints
- [ ] Conflict resolution

## Config File Structure

```json
{
  "hosts": {
    "4090": {
      "name": "Training Server",
      "host": "192.168.x.x",
      "role": "trainer",
      "services": {
        "vault": {"port": 8767, "path": "/api"},
        "ledger": {"port": 8767, "path": "/api/ledger"},
        "training": {"port": 8767, "path": "/api/training"},
        "branch": {"port": 8768}
      }
    },
    "3090": {
      "name": "Inference Server",
      "host": "192.168.x.x",
      "role": "inference",
      "services": {
        "inference": {"port": 8765},
        "branch": {"port": 8768}
      }
    }
  },
  "local_host": "4090",
  "default_trainer": "4090",
  "default_inference": "3090"
}
```

## Migration Path

1. **Phase 1-2:** Add APIs, keep local file reads working
2. **Phase 3-4:** Add remote clients with fallback to local
3. **Phase 5:** Switch Tavern to APIs (can still run on 4090)
4. **Phase 6:** Enable running Tavern elsewhere
5. **Final:** All components location-independent

## Testing Strategy

1. Run everything on 4090 (current state) - should work
2. Run Tavern on laptop, point to 4090 APIs - should work
3. Run Tavern on 3090, point to 4090 APIs - should work
4. Multiple inference hosts - should work

## Success Criteria

- [ ] Tavern can run on any machine with network access
- [ ] No hardcoded IPs in application code (only in config)
- [ ] All data accessed via APIs, not file system
- [ ] Host registry is single source of truth
- [ ] Adding a new host = edit config, start services
