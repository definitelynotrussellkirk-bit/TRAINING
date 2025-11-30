# Event Bus Architecture Diagrams

**Companion document to:** `PLAN_EVENT_BUS_CONSOLIDATION.md`

---

## Current State (Messy - 3 Overlapping Systems)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVENT PRODUCERS                                 │
│  (training_daemon, eval_runner, jobs, workers, vault, guild, etc.)     │
└────────────┬────────────────┬────────────────┬──────────────────────────┘
             │                │                │
             │                │                │
     ┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────────┐
     │ core/events  │  │core/battle  │  │ core/realm     │
     │              │  │    _log     │  │   _store       │
     │ JSONL write  │  │ SQLite      │  │ HTTP → SQLite  │
     └───────┬──────┘  └──────┬──────┘  └─────┬──────────┘
             │                │                │
             │                │                │
        ┌────▼──────┐    ┌────▼──────┐   ┌────▼────────┐
        │status/    │    │vault/     │   │data/realm_  │
        │events.jsonl│    │jobs.db    │   │state.db     │
        └────┬──────┘    └──────┬────┘   └────┬────────┘
             │                  │              │
             │                  │              │
             ▼                  ▼              ▼
      ❌ Some UIs read    ✅ Battle Log    ✅ Game UI
         this file         UI reads         reads this
         (inconsistent)    this             (RealmState)

PROBLEMS:
  - Three separate "canonical" event streams
  - Different producers pick different systems
  - Different consumers read different sources
  - Some events logged to one but not others
  - Naming collision: core/battle_log vs arena/battle_log
```

---

## Target State (Clean - 1 Producer, 2 Consumers)

```
┌────────────────────────────────────────────────────────────────────────┐
│                         EVENT PRODUCERS                                │
│  (training_daemon, eval_runner, jobs, workers, vault, guild, etc.)    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
                               │  ALL producers call:
                               │  emit_realm_event()
                               │
                    ┌──────────▼────────────┐
                    │  core/event_bus.py    │
                    │  (CANONICAL PRODUCER) │
                    │                       │
                    │  emit_realm_event()   │
                    │  job_started()        │
                    │  checkpoint_saved()   │
                    │  etc.                 │
                    └──────┬────────┬───────┘
                           │        │
              ┌────────────┤        └────────────┐
              │                                  │
     ┌────────▼─────────┐              ┌────────▼─────────┐
     │ core/realm_store │              │ core/battle_log  │
     │                  │              │                  │
     │ Live events      │              │ Persistent log   │
     │ (in-memory deque)│              │ (SQLite)         │
     │ + SSE broadcast  │              │ + queryable      │
     └────────┬─────────┘              └────────┬─────────┘
              │                                 │
              │                                 │
        ┌─────▼────────┐                  ┌─────▼────────┐
        │RealmService  │                  │vault/jobs.db │
        │port 8866     │                  │battle_log    │
        │data/realm_   │                  │   table      │
        │ state.db     │                  └─────┬────────┘
        └─────┬────────┘                        │
              │                                 │
              │                                 │
    ┌─────────▼─────────────┐         ┌─────────▼────────────┐
    │ CONSUMER MODE 1:      │         │ CONSUMER MODE 2:     │
    │ LIVE VIEW             │         │ HISTORIC VIEW        │
    │                       │         │                      │
    │ - Last 50 events      │         │ - Filterable queries │
    │ - Real-time SSE       │         │ - 24+ hours history  │
    │ - War Room (game.js)  │         │ - Channel filters    │
    │ - RealmState API      │         │ - Battle Log UI      │
    │                       │         │ - Analytics          │
    └───────────────────────┘         └──────────────────────┘

BENEFITS:
  ✅ Single producer API (core/event_bus)
  ✅ Events appear in BOTH modes automatically
  ✅ Clear consumption patterns
  ✅ Backward compatible
  ✅ No naming collisions
```

---

## Data Flow Details

### Producer Path (Event Emission)

```
Event happens (e.g., checkpoint saved)
    │
    ▼
from core.event_bus import checkpoint_saved
checkpoint_saved(step=183000, loss=0.234, path="/path/to/ckpt")
    │
    ▼
core/event_bus.emit_realm_event(
    kind="checkpoint_saved",
    message="Checkpoint 183,000 saved (loss: 0.2340)",
    channel="checkpoint",  # auto-detected from kind
    severity="success",    # auto-detected from kind
    details={"step": 183000, "loss": 0.234, "path": "..."},
)
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      ▼
RealmStore           BattleLogger         (optional future)
.emit_event()        .log()                - Webhooks
                                           - External APIs
    │                      │                - Event replay
    ▼                      ▼
HTTP POST             INSERT INTO
→ RealmService        battle_log
→ SQLite write        (vault/jobs.db)
→ SSE broadcast
    │                      │
    ▼                      ▼
Browser receives      Available for
via /api/events/     /api/battle_log
stream                queries
```

### Consumer Path 1: Live View

```
User opens http://localhost:8888 (Tavern/War Room)
    │
    ▼
Frontend JS connects to SSE: /api/events/stream
    │
    ▼
window.eventStream.on('training.step', (event) => {
    // Real-time event arrives
    updateGraph(event.data.loss, event.data.step);
});
    │
    ▼
Graph updates immediately as training progresses

CHARACTERISTICS:
- Near-instant (< 1 second latency)
- Last ~50 events in memory
- Perfect for dashboards
- Stateless (reconnect = fresh stream)
```

### Consumer Path 2: Historic View

```
User opens http://localhost:8888/battle-log
    │
    ▼
Frontend polls: GET /api/battle_log?channels=training,jobs&limit=100
    │
    ▼
vault/server.py calls BattleLogger.get_events(channels=[...])
    │
    ▼
SELECT * FROM battle_log
WHERE channel IN ('training', 'jobs')
ORDER BY timestamp DESC
LIMIT 100
    │
    ▼
Returns events with full details, filterable, paginated

CHARACTERISTICS:
- Queryable (filter by channel, time, severity)
- 24+ hour retention (configurable)
- Full event details preserved
- Perfect for debugging, audit trails
```

---

## Channel Architecture

### Channel Hierarchy

```
EVENT
  ├─ kind: "checkpoint_saved"  (what happened)
  ├─ channel: "checkpoint"     (where to show it)
  ├─ severity: "success"       (how important)
  └─ message: "Checkpoint 183,000 saved..."

CHANNELS (where events appear):

system       ⚙️  - Server start/stop, config changes, critical errors
jobs         ⚔️  - Job lifecycle (claimed, started, completed, failed)
training     📈  - Training progress, LR changes, campaign milestones
eval         📊  - Evaluation results, accuracy metrics, regressions
vault        🗃️  - Archive/retention/sync operations
guild        🏰  - Titles earned, lore events, hero progression
debug        🔧  - Dev-only internal assertions, edge cases
data         📦  - NEW - Forge, curriculum, dataset generation
checkpoint   💾  - NEW - Checkpoint saves, ledger updates

SEVERITIES:

info         ℹ️  - Normal events (90% of events)
success      ✅  - Positive outcomes (checkpoint saved, job completed)
warning      ⚠️  - Attention needed (queue backlog, stale worker)
error        ❌  - Something went wrong (job failed, training crashed)
```

### Kind → Channel Mapping

```python
KIND_METADATA = {
    # Jobs
    "job_submitted":    ("jobs", "info"),
    "job_started":      ("jobs", "info"),
    "job_completed":    ("jobs", "success"),
    "job_failed":       ("jobs", "error"),

    # Training
    "training_started":     ("training", "info"),
    "training_completed":   ("training", "success"),
    "training_paused":      ("training", "warning"),
    "training_failed":      ("training", "error"),

    # Checkpoints
    "checkpoint_saved":     ("checkpoint", "success"),
    "checkpoint_promoted":  ("checkpoint", "success"),
    "checkpoint_deleted":   ("checkpoint", "info"),

    # Evaluations
    "eval_started":     ("eval", "info"),
    "eval_completed":   ("eval", "success"),
    "eval_regression":  ("eval", "warning"),

    # Data/Curriculum
    "dataset_generated":  ("data", "success"),
    "curriculum_updated": ("data", "info"),
    "forge_batch_ready":  ("data", "info"),

    # System
    "server_started":   ("system", "success"),
    "server_stopped":   ("system", "info"),
    "config_reloaded":  ("system", "info"),
    "worker_joined":    ("system", "success"),
    "worker_left":      ("system", "warning"),
    "warning_raised":   ("system", "warning"),

    # Guild
    "title_earned":     ("guild", "success"),
    "skill_level_up":   ("guild", "success"),
    "achievement":      ("guild", "success"),
}
```

---

## Migration Strategy Visualization

### Phase 1: Add event_bus (backward compatible)

```
BEFORE:
  Producer → core/events.emit_job_started() → events.jsonl

AFTER (Phase 1):
  Producer → core/events.emit_job_started() → WRAPPER
                                                  │
                                                  ▼
                                            event_bus.job_started()
                                                  │
                                                  ├──→ RealmStore
                                                  └──→ BattleLogger

  Old code still works! Now writes to BOTH places.
```

### Phase 2: Direct migration (optional, gradual)

```
PHASE 1 (compatible):
  from core.events import emit_job_started
  emit_job_started(job_id, job_type, worker_id)

PHASE 2 (direct):
  from core.event_bus import job_started
  job_started(job_id, job_type, worker_id)

Same result, but clearer intent. Migration is OPTIONAL.
```

### Phase 3: Deprecation warnings

```
PHASE 3:
  from core.events import emit_job_started  # ⚠️ DeprecationWarning
  emit_job_started(...)

  User sees:
    DeprecationWarning: core.events is deprecated.
    Use core.event_bus instead.
```

---

## Testing Strategy

### Unit Tests

```python
def test_event_bus_dual_write():
    """Events appear in both RealmStore and BattleLogger."""
    from core.event_bus import checkpoint_saved

    # Emit event
    checkpoint_saved(step=1000, loss=0.5, path="/tmp/ckpt")

    # Check RealmStore (live)
    events = get_events(limit=10)
    assert any(e['kind'] == 'checkpoint_saved' for e in events)

    # Check BattleLogger (persistent)
    logger = get_battle_logger()
    events = logger.get_events(channels=['checkpoint'], limit=10)
    assert any(e.message.startswith('Checkpoint 1,000') for e in events)


def test_channel_auto_detection():
    """Channels auto-detected from kind."""
    from core.event_bus import emit_realm_event

    event = emit_realm_event(
        kind="job_started",
        message="Job eval started",
    )

    assert event['channel'] == 'jobs'
    assert event['severity'] == 'info'
```

### Integration Tests

```bash
# Start training
python3 core/training_daemon.py &

# Check live events
curl http://localhost:8866/api/events | jq '.events[0]'

# Check persistent log
curl http://localhost:8767/api/battle_log?limit=10 | jq '.[0]'

# Both should show recent training events
```

### UI Tests

1. **War Room (Live View)**
   - Open http://localhost:8888
   - Verify events appear in real-time
   - Verify graph updates

2. **Battle Log (Historic View)**
   - Open http://localhost:8888/battle-log
   - Verify channel filters work
   - Verify 24h history visible

---

## Performance Considerations

### Write Performance

```
Single event emission:

RealmStore write:
  - HTTP POST → ~5ms
  - SQLite write → ~1ms
  - SSE broadcast → ~0.5ms
  - TOTAL: ~6.5ms per event

BattleLogger write:
  - SQLite INSERT → ~1ms
  - TOTAL: ~1ms per event

COMBINED: ~7.5ms per event
```

**Impact:** Negligible. Training loop emits ~1 event per 4 seconds (heartbeat).

### Read Performance

```
Live view (RealmState):
  - In-memory deque → O(1) access
  - Last 50 events → instant

Historic view (BattleLogger):
  - SQLite SELECT with index → ~2ms
  - 100 events → ~5ms
  - Pagination supported
```

**Impact:** No user-facing latency.

---

## Rollback Plan

### Phase 1 Rollback

```python
# If event_bus causes issues, revert to direct writes:

# BEFORE (event_bus)
from core.event_bus import job_started
job_started(job_id, job_type, worker_id)

# ROLLBACK (direct)
from core.events import emit_job_started
emit_job_started(job_id, job_type, worker_id)

# Both work identically after Phase 1!
```

### Phase 2 Rollback

- UI changes independent of backend
- Can revert battle_log.html changes without affecting event flow

### Phase 3 Rollback

```bash
# Restore JSONL storage if needed:
git revert <commit-hash>

# arena/battle_log.py rename:
git mv arena/training_status.py arena/battle_log.py
```

---

## Future Extensions

### 1. Event Replay

```python
# Backfill events from SQLite into RealmState
from core.event_bus import replay_events
replay_events(since="2025-11-29T00:00:00Z")
```

### 2. Event Webhooks

```python
# Send events to external systems
emit_realm_event(
    kind="checkpoint_saved",
    message="...",
    webhooks=["https://discord.com/webhooks/..."]
)
```

### 3. Event Aggregation

```sql
-- Daily summaries
SELECT channel, COUNT(*) as count, date(timestamp)
FROM battle_log
WHERE timestamp > datetime('now', '-7 days')
GROUP BY channel, date(timestamp)
```

### 4. Event Export

```bash
# Export to JSONL for analysis
python3 -m core.event_bus export --since 2025-11-29 --output events.jsonl
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Event** | Something that happened (checkpoint saved, job started) |
| **Kind** | Event type identifier (e.g., `checkpoint_saved`) |
| **Channel** | Where to display the event (e.g., `training`, `jobs`) |
| **Severity** | Importance level (`info`, `success`, `warning`, `error`) |
| **Producer** | Code that emits events |
| **Consumer** | Code/UI that reads events |
| **Live View** | Recent events from RealmState (in-memory) |
| **Historic View** | Queryable events from BattleLogger (SQLite) |
| **event_bus** | Canonical producer API (`core/event_bus.py`) |

---

**Related Documents:**
- `PLAN_EVENT_BUS_CONSOLIDATION.md` - Full implementation plan
- `ARCHITECTURE.md` - System architecture overview
- `core/event_bus.py` - Canonical producer API (to be created)
