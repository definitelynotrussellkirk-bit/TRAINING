# Evaluation System Upgrade Plan

**Goal**: Two-tier eval system (QUICK/FULL), campaign-aware, prioritizing recent checkpoints.

---

## Current State Analysis

### What Exists
| Component | Location | Status |
|-----------|----------|--------|
| Eval Runner | `core/eval_runner.py` | Works, uses old validation format |
| Eval Ledger | `core/evaluation_ledger.py` | Works, queue + storage |
| Task Master | `guild/task_master.py` | Works, monitors 3090 |
| Task Registry | `guild/task_registry.py` | Has `process_eval_queue` task |
| New Eval Sets | `data/validation/{skill}/level_XX.json` | Just pulled via Summoner |
| Old Eval Sets | `data/validation/{skill}_validation.json` | Legacy format |

### Current Flow
```
Checkpoint Save → queue_evaluation() → eval_queue.json → Task Master → EvalRunner → ledger
```

### Problems
1. **Single eval type** - Only 5 problems, no comprehensive eval
2. **Old eval format** - Runner uses `{skill}_validation.json`, not new per-level files
3. **FIFO queue** - No priority for recent checkpoints
4. **No backfill** - Old checkpoints never evaluated if missed
5. **Campaign isolation incomplete** - Ledger symlinks work, but eval runner not campaign-aware

---

## Proposed Architecture

### Two Eval Tiers

| Tier | Problems | Trigger | Priority | Purpose |
|------|----------|---------|----------|---------|
| **QUICK** | 5/level | Checkpoint save | HIGH (10) | Fast signal, progression gates |
| **FULL** | All levels | Scheduled, manual | MED (6) | Comprehensive assessment |

### QUICK Eval
- **When**: Every checkpoint save (current behavior)
- **What**: 5 problems at CURRENT skill level only
- **Time**: ~30 seconds per skill
- **Storage**: `status/evaluation_ledger.json` + sidecar

### FULL Eval
- **When**:
  - Every N checkpoints (e.g., every 5000 steps)
  - On demand (`--full`)
  - Scheduled task when GPU idle
- **What**: All 5 problems × ALL levels (up to current)
- **Time**: ~2-5 minutes per skill
- **Storage**: Same ledger, marked as `eval_type: "full"`

### Priority Queue (Newest First)

```python
# Instead of FIFO, sort by:
# 1. Eval type (QUICK before FULL for same checkpoint)
# 2. Checkpoint step (descending - newest first)
# 3. Queue time (ascending - older queued items first)

queue.sort(key=lambda e: (-e.checkpoint_step, e.eval_type == "full", e.queued_at))
```

### Backfill System

```
status/eval_backlog.json:
{
  "last_scanned": 183000,
  "missing_evals": [
    {"step": 180000, "skill": "bin", "levels": [1,2,3,4,5]},
    {"step": 175000, "skill": "sy", "levels": [1,2,3]}
  ]
}
```

**Backfill Task** (priority 3):
- Scans checkpoint ledger for gaps
- Queues missing evals at LOW priority
- Runs when nothing else is pending

---

## Implementation Plan

### Phase 1: Update Eval Runner for New Format

**File**: `core/eval_runner.py`

1. Change `load_validation_set()` to read from `data/validation/{skill}/level_{NN}.json`
2. Add `eval_type` field to queue entries ("quick" or "full")
3. Add `--full` flag for comprehensive evaluation
4. Support campaign-aware paths

```python
def load_validation_set(skill: str, level: int) -> List[dict]:
    """Load eval set for specific skill and level."""
    eval_file = BASE_DIR / "data" / "validation" / skill / f"level_{level:02d}.json"
    if not eval_file.exists():
        raise FileNotFoundError(f"No eval set for {skill} level {level}")

    with open(eval_file) as f:
        data = json.load(f)

    return data.get("problems", [])
```

### Phase 2: Priority Queue Implementation

**File**: `core/evaluation_ledger.py`

1. Add priority sorting to `get_pending_evaluations()`
2. Add `eval_type` and `priority` fields to queue schema
3. Add backlog scanning

```python
@dataclass
class EvalQueueEntry:
    checkpoint_step: int
    skill: str
    level: int
    queued_at: str
    eval_type: str = "quick"  # "quick" or "full"
    priority: int = 10        # 1-10, higher = sooner

def get_pending_evaluations(self, sorted: bool = True) -> List[EvalQueueEntry]:
    """Get pending evals, optionally sorted by priority."""
    entries = self._load_queue()
    if sorted:
        # Newest checkpoints first, quick before full
        entries.sort(key=lambda e: (
            -e.priority,
            -e.checkpoint_step,
            e.eval_type == "full",
            e.queued_at
        ))
    return entries
```

### Phase 3: Campaign-Aware Storage

**File**: `core/evaluation_ledger.py`

1. Get active campaign from `control/active_campaign.json`
2. Store results in campaign-specific ledger
3. Keep top-level as symlink (current behavior works)

```python
def _get_ledger_path(self) -> Path:
    """Get path to evaluation ledger (campaign-aware)."""
    # Top-level status/ is symlinked to active campaign
    # So this should "just work" if symlinks are maintained
    return BASE_DIR / "status" / "evaluation_ledger.json"
```

### Phase 4: Task Registry Updates

**File**: `guild/task_registry.py`

Add new tasks:

```python
TASKS = [
    # Existing
    Task(
        id="process_eval_queue_quick",
        name="Quick Evaluations",
        description="Process pending QUICK evaluations (5 problems/level)",
        gpu="3090",
        priority=10,  # HIGHEST
        command=["python3", "core/eval_runner.py", "--once", "--type", "quick"],
        cooldown=60,
        estimated_duration=120,
    ),

    # NEW
    Task(
        id="process_eval_queue_full",
        name="Full Evaluations",
        description="Process pending FULL evaluations (all levels)",
        gpu="3090",
        priority=6,  # Medium
        command=["python3", "core/eval_runner.py", "--once", "--type", "full"],
        cooldown=300,
        estimated_duration=600,
    ),

    Task(
        id="eval_backfill",
        name="Backfill Missing Evals",
        description="Queue evaluations for checkpoints with missing evals",
        gpu="none",  # Just queuing, no GPU needed
        priority=3,  # Low
        command=["python3", "core/eval_runner.py", "--backfill"],
        cooldown=3600,
        estimated_duration=30,
    ),
]
```

### Phase 5: train.py Integration

**File**: `core/train.py`

Update checkpoint save callback:

```python
def on_checkpoint_save(step: int, skill: str, level: int):
    """Queue evaluations when checkpoint is saved."""
    ledger = get_eval_ledger()

    # Always queue QUICK eval at current level
    ledger.queue_evaluation(
        checkpoint_step=step,
        skill=skill,
        level=level,
        eval_type="quick",
        priority=10
    )

    # Queue FULL eval every 5000 steps
    if step % 5000 == 0:
        ledger.queue_evaluation(
            checkpoint_step=step,
            skill=skill,
            level=None,  # All levels
            eval_type="full",
            priority=6
        )
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `core/eval_runner.py` | New format loader, eval_type support, --full flag, campaign paths |
| `core/evaluation_ledger.py` | Priority queue, eval_type field, backfill scanner |
| `guild/task_registry.py` | Add full eval + backfill tasks |
| `core/train.py` | Queue full evals every N steps |
| `guild/summoner.py` | Already done - pulls eval sets |

---

## New CLI Commands

```bash
# Run quick evals for pending queue
python3 core/eval_runner.py --once --type quick

# Run full eval for specific checkpoint
python3 core/eval_runner.py --checkpoint 183000 --full

# Run full eval for all skills at current level
python3 core/eval_runner.py --full

# Scan and queue missing evals (backfill)
python3 core/eval_runner.py --backfill

# Show eval status
python3 core/eval_runner.py --status
```

---

## Task Master Flow (Updated)

```
Every 60 seconds:
  1. Check 3090 status (idle?)
  2. Get available tasks sorted by priority:
     - process_eval_queue_quick (10) - ALWAYS FIRST
     - process_eval_queue_full (6)
     - sparring_* (8-9)
     - eval_backfill (3)
  3. Run highest priority task that's ready (cooldown passed)
  4. Record result
```

---

## Verification Checklist

- [ ] New eval format loads correctly
- [ ] Quick evals run on checkpoint save
- [ ] Full evals run every 5000 steps
- [ ] Priority queue sorts newest first
- [ ] Results stored in campaign ledger
- [ ] Backfill finds missing evals
- [ ] Task Master runs both eval types
- [ ] Tavern shows eval progress

---

## Migration Notes

1. **Existing evals**: Keep in ledger, they're valid
2. **Old format files**: Can delete `{skill}_validation.json` after testing
3. **Symlinks**: Already working, no changes needed
4. **Backwards compatibility**: eval_type defaults to "quick" for old entries
