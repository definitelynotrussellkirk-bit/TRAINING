# Evolution Tracker Debugging Findings

**Date:** 2025-11-16
**Status:** Problem identified, solution in progress

## Problem Statement

1. **User reports:** "learning evolution viewer isn't working"
2. **Training live monitor:** WORKING ✅
3. **Evolution viewer:** NOT showing data ❌

## System Diagnosis

### What's Working ✅
- Training daemon running
- Live monitor on port 8080 serving data
- Memory API on port 8081 responding
- Training progressing (step 10/11)
- Evaluations triggering every 10 steps
- Evolution tracker initialized: `✅ Evolution tracker initialized for: evolution_demo`

### What's Broken ❌
1. **No snapshots captured** - directories exist but are empty
2. **Evolution API returns empty** - `{"datasets": []}`
3. **Evolution viewer shows no data** - because no snapshots exist

## Root Cause Analysis

### Evidence Trail

1. **Directory structure exists:**
   ```
   data/evolution_snapshots/
   ├── evolution_demo/    (EMPTY - 0 files)
   ├── evolution_test/    (EMPTY)
   └── proper_test/       (EMPTY)
   ```

2. **Evolution tracker initialized:**
   ```
   training_output.log line 80:
   ✅ Evolution tracker initialized for: evolution_demo
   ```

3. **Evaluation triggered:**
   ```
   training_output.log:
   DEBUG: Eval triggered at step 10
   ```

4. **But NO snapshot message:**
   ```
   Expected: "📸 Capturing evolution snapshot at step 10"
   Found: NOTHING
   ```

### Code Analysis

**In evolution_tracker.py:**
```python
def capture_snapshot(self, ...):
    if not self.should_snapshot(current_step):
        return None  # ← Returning here silently!

    print(f"📸 Capturing evolution snapshot at step {current_step}")
    # ... rest of snapshot code
```

**Snapshot schedule includes step 10:**
```python
schedule = [
    0,   # Baseline
    10,  # ← Step 10 IS in schedule
    25, 50, 100, 150, ...
]
```

**In train.py (line 882):**
```python
if self.evolution_tracker:
    try:
        self.evolution_tracker.capture_snapshot(
            model=self.model_ref,
            tokenizer=self.tokenizer,
            examples=self.raw_train_examples,
            current_step=state.global_step,  # ← What is this value?
            model_version="training",
            max_examples=100
        )
    except Exception as e:
        print(f"⚠️ Evolution tracker failed at step {state.global_step}: {e}")
```

## Hypotheses (in order of likelihood)

### Hypothesis 1: global_step mismatch ⭐ MOST LIKELY
- `state.global_step` might not equal the "training step" shown in logs
- If checkpoint exists, global_step might be offset
- Log says "step 10" but global_step might be different (e.g., 13, 14, etc.)

**Evidence:**
- Checkpoint warning: `⚠️ Checkpoint checkpoint-3 is already completed (step 3/3)`
- This suggests resuming from a checkpoint
- Global step could be higher than local step

### Hypothesis 2: capture_snapshot() not being called
- The callback isn't executing
- Exception caught silently

### Hypothesis 3: should_snapshot() logic issue
- Something wrong with the schedule check
- `self.last_snapshot_step` already set to 10

## Next Steps

### 1. Add Debug Logging
Add to train.py before calling capture_snapshot():
```python
print(f"🔍 DEBUG: About to call capture_snapshot(current_step={state.global_step})")
```

Add to evolution_tracker.py in should_snapshot():
```python
def should_snapshot(self, current_step: int) -> bool:
    print(f"🔍 should_snapshot({current_step}): schedule={self.snapshot_schedule[:5]}..., last={self.last_snapshot_step}")
    result = current_step in self.snapshot_schedule and current_step != self.last_snapshot_step
    print(f"🔍 should_snapshot({current_step}) → {result}")
    return result
```

### 2. Create Test Script
Create standalone test to verify evolution tracker works in isolation.

### 3. Check Checkpoint State
Verify what global_step actually is vs what we expect.

## Quick Fixes to Try

### Fix 1: Force snapshot at every eval
Change schedule to include ALL multiples of 10:
```python
schedule = list(range(0, 100, 10)) + [100, 150, 200, ...]
```

### Fix 2: Log when capture_snapshot returns None
```python
result = self.evolution_tracker.capture_snapshot(...)
if result is None:
    print(f"⚠️  Evolution tracker skipped snapshot at step {state.global_step}")
```

### Fix 3: Reset last_snapshot_step
If it's stuck, manually reset:
```python
self.evolution_tracker.last_snapshot_step = -1
```

## Files Involved

- `/path/to/training/evolution_tracker.py` - Snapshot capture logic
- `/path/to/training/train.py` - Calls capture_snapshot() at line 882
- `/path/to/training/launch_live_monitor.py` - Serves evolution API
- `/path/to/training/evolution_viewer.html` - UI (works if data exists)
- `/path/to/training/data/evolution_snapshots/` - Empty directories

## Commands for User

### Check if snapshots exist:
```bash
find data/evolution_snapshots -name "*.json" -type f
```

### Watch for snapshot messages in real-time:
```bash
tail -f training_output.log | grep -E "(📸|should_snapshot|capture_snapshot)"
```

### Check current global_step:
```bash
cat status/training_status.json | jq '{current_step, total_steps, dataset_name}'
```

### Test evolution tracker directly:
```bash
python3 -c "
from evolution_tracker import EvolutionTracker
from pathlib import Path
tracker = EvolutionTracker(Path('.'), 'test_debug')
print('Step 0 should snapshot:', tracker.should_snapshot(0))
print('Step 10 should snapshot:', tracker.should_snapshot(10))
print('Schedule:', tracker.snapshot_schedule[:10])
"
```

## Status

**Current:** Identified that capture_snapshot() is returning early from should_snapshot()
**Next:** Add debug logging to see why should_snapshot() returns False at step 10
**ETA:** 15 minutes to debug + fix + validate
