# Training System Edge Case Analysis
**Generated:** 2025-11-11
**Disk Space:** 1.5TB free (18% used) - HEALTHY

---

## SUMMARY

The system has **good basic error handling** but **lacks robustness** for critical edge cases. Most concerning are:
1. ❌ **No atomic operations** for checkpoint saves (partial state possible)
2. ❌ **No validation** of checkpoint integrity on load
3. ❌ **No recovery** from corrupted state files
4. ⚠️  **3 AM consolidation** only waits for empty inbox (doesn't check if training actively running)
5. ✅ **Batch transitions** work correctly (cumulative global_step)

---

## 1. BATCH TRANSITION EDGE CASES

### Current Behavior (train.py:565-654)
```python
# Read current global_step from checkpoint
current_global_step = 0
if Path(self.args.output_dir).exists():
    trainer_state_file = Path(self.args.output_dir) / "trainer_state.json"
    if trainer_state_file.exists():
        trainer_state = json.load(f)
        current_global_step = trainer_state.get("global_step", 0)

# Calculate cumulative total
total_steps = current_global_step + steps_this_batch

# After training: Copy optimizer state from checkpoint to root
checkpoint_paths = sorted(Path(self.args.output_dir).glob("checkpoint-*"))
if checkpoint_paths:
    latest_checkpoint = checkpoint_paths[-1]
    for filename in ['optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth']:
        shutil.copy2(latest_checkpoint / filename, output_dir / filename)

    # Delete checkpoints
    for checkpoint_path in checkpoint_paths:
        shutil.rmtree(checkpoint_path)
```

### ✅ WORKING CORRECTLY
- **Cumulative global_step**: Correctly reads and accumulates across batches
- **Optimizer state preservation**: Copies optimizer.pt to root for next batch
- **Sequential processing**: Only one file trained at a time (no race conditions)

### ❌ CRITICAL EDGE CASES

**1. Corrupted trainer_state.json**
- **What happens**: `json.load()` throws exception
- **Current handling**: ⚠️  Exception bubbles up, training aborts
- **Impact**: Entire batch queue stalls
- **Fix needed**: Try/except with fallback to global_step=0

**2. Partial checkpoint copy failure**
- **What happens**: Disk full or I/O error mid-copy
- **Current handling**: ❌ No rollback - partial files left behind
- **Impact**: Next batch loads corrupted optimizer state
- **Fix needed**: Atomic copy (write to temp, then rename)

**3. Checkpoint deletion fails after successful copy**
- **What happens**: Disk fills up with old checkpoints
- **Current handling**: ❌ Exception ignored, checkpoints accumulate
- **Impact**: Eventual disk full
- **Fix needed**: Log warning, continue (checkpoints are redundant after copy)

**4. global_step mismatch (manual editing)**
- **What happens**: User manually edits trainer_state.json
- **Current handling**: ❌ Blindly trusts the value
- **Impact**: Progress tracking incorrect, but training continues
- **Fix needed**: Validation against expected range

**5. Training crashes DURING final checkpoint save**
- **What happens**: Optimizer state not copied to root
- **Current handling**: ❌ Next batch starts from stale optimizer state
- **Impact**: Loss may spike (stale momentum)
- **Fix needed**: Write completion marker after successful save

---

## 2. 3 AM CONSOLIDATION EDGE CASES

### Current Behavior (training_daemon.py:198-270)
```python
def should_consolidate(self) -> bool:
    today = datetime.now().date()
    if self.last_consolidation_date == today:
        return False

    consolidation_time = datetime.strptime("03:00", "%H:%M").time()
    current_time = datetime.now().time()

    if current_time >= consolidation_time:
        return True
    return False

# In main loop:
inbox_files_check = self.get_inbox_files()
if not inbox_files_check and self.should_consolidate():
    self.consolidate_model()
```

### ✅ GOOD DESIGN CHOICES
- **Waits for empty inbox**: Only consolidates when idle (line 460)
- **Daily marker file**: `.last_consolidation` prevents duplicate runs
- **Subprocess with timeout**: 30-minute timeout prevents hangs
- **Backup before merge**: Adapter backed up to `consolidated_backups/`

### ❌ CRITICAL EDGE CASES

**1. Training actively running at 3 AM**
- **What happens**: Daemon checks inbox (empty), but training in progress
- **Current handling**: ✅ SAFE - Training is synchronous, blocks daemon loop
- **Impact**: Consolidation delayed until training completes
- **Status**: No fix needed - working as designed

**2. Consolidation crashes mid-merge**
- **What happens**: Partial merged model in `consolidated_models/`
- **Current handling**: ❌ Partial model left behind, config.json unchanged
- **Impact**: Wasted disk space, but training continues on old base
- **Fix needed**: Atomic commit (write to temp dir, rename on success)

**3. Consolidation deletes adapter before saving merged model**
- **What happens**: Power loss between delete and save
- **Current handling**: ❌ CATASTROPHIC - All learning lost!
- **Impact**: Must restore from daily snapshot
- **Fix needed**: Change order: save first, delete last

**4. Config.json update fails after merge**
- **What happens**: Merged model saved but config still points to old base
- **Current handling**: ❌ Next training uses old base (ignores merged model)
- **Impact**: Wasted consolidation effort
- **Fix needed**: Atomic config update (write temp, rename)

**5. Disk full during merge**
- **What happens**: Merge fails partway through
- **Current handling**: ⚠️  Exception caught, logged (line 266-267)
- **Impact**: Consolidation fails but training continues
- **Status**: Acceptable - needs disk space monitoring

**6. System restart during consolidation**
- **What happens**: Partial state everywhere
- **Current handling**: ❌ No cleanup on startup
- **Impact**: Wasted disk space, possible corruption
- **Fix needed**: Startup health check to clean incomplete consolidations

---

## 3. TRAINING CRASH EDGE CASES

### Current Behavior (train.py:665-680)
```python
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ Training error: {error_msg}")
    traceback.print_exc()

    self.status_writer.mark_crashed(error_msg, type(e).__name__)

    if "out of memory" in error_msg.lower():
        self.notifier.oom_error()
    else:
        self.notifier.training_crashed(error_msg)

    return False
```

### ❌ CRITICAL ISSUES

**1. No cleanup of partial state**
- **What happens**: Crash leaves partial checkpoints
- **Current handling**: ❌ Files left in output_dir
- **Impact**: Next batch may try to resume from corrupted checkpoint
- **Fix needed**: Delete partial checkpoints on crash

**2. Status JSON left in "training" state**
- **What happens**: Crash before mark_crashed() is called
- **Current handling**: ❌ Status shows "training" forever
- **Impact**: Monitors show stale status
- **Fix needed**: Startup health check to reset stale status

**3. Daemon keeps failed file**
- **What happens**: Training fails, daemon doesn't delete file
- **Current handling**: ✅ GOOD - Keeps file (line 480)
- **Impact**: File can be fixed manually
- **Status**: Working as designed

**4. OOM crash leaves GPU memory allocated**
- **What happens**: CUDA context not cleaned up
- **Current handling**: ⚠️  Python exit cleans up automatically
- **Impact**: Usually fine, but rarely requires reboot
- **Fix needed**: Explicit `torch.cuda.empty_cache()` on crash

**5. No retry logic**
- **What happens**: Transient error (network, GPU hiccup) aborts training
- **Current handling**: ❌ No retries
- **Impact**: Failed batch requires manual restart
- **Fix needed**: Retry logic with exponential backoff

---

## 4. DAEMON RESTART EDGE CASES

### Current Behavior (training_daemon.py:133-155)
```python
def initialize_model(self):
    # Check if current_model exists
    if (self.current_model_dir / "adapter_config.json").exists():
        self.logger.info(f"Current model already exists")
        return

    # Check for latest snapshot
    snapshots = sorted(self.snapshots_dir.glob("20*"))
    if snapshots:
        latest_snapshot = snapshots[-1]
        shutil.copytree(latest_snapshot, self.current_model_dir, dirs_exist_ok=True)
        return

    # No snapshot, need base model
    if not self.config.get("model_path"):
        self.logger.error("No model_path in config!")
        sys.exit(1)
```

### ❌ CRITICAL EDGE CASES

**1. Daemon crashes mid-training**
- **What happens**: current_model/ has partial state
- **Current handling**: ❌ Resumes using partial model (assumes valid)
- **Impact**: May continue training on corrupted model
- **Fix needed**: Validate model integrity on startup

**2. Corrupt current_model/ on startup**
- **What happens**: adapter_config.json exists but files corrupted
- **Current handling**: ❌ Assumes valid, tries to resume
- **Impact**: Training fails immediately
- **Fix needed**: Health check on startup (try loading model)

**3. Status JSON shows "training" but no process**
- **What happens**: Previous daemon crashed
- **Current handling**: ❌ Status not reset
- **Impact**: Monitors show stale data
- **Fix needed**: Reset status to "idle" on startup

**4. `.stop` file left from previous run**
- **What happens**: Daemon starts but immediately stops
- **Current handling**: ✅ GOOD - Deletes .stop on stop (line 449)
- **Impact**: None
- **Status**: Working correctly

---

## 5. DISK SPACE EDGE CASES

### Current Behavior
❌ **NO DISK SPACE MONITORING**

### CRITICAL MISSING FEATURES

**1. Training starts with insufficient space**
- **What happens**: Crash mid-training when disk fills
- **Current handling**: ❌ No pre-check
- **Impact**: Wasted GPU time, corrupted checkpoints
- **Fix needed**: Check available space before training

**2. Snapshots fill disk**
- **What happens**: Daily snapshots accumulate indefinitely
- **Current handling**: ❌ No cleanup policy
- **Impact**: Eventual disk full
- **Fix needed**: Keep only N most recent snapshots

**3. Consolidated models fill disk**
- **What happens**: Each consolidation creates ~30GB merged model
- **Current handling**: ❌ Never deleted
- **Impact**: Disk fills over weeks
- **Fix needed**: Keep only N most recent consolidated models

**4. Disk fills during training**
- **What happens**: Checkpoint save fails
- **Current handling**: ⚠️  Exception logged, training continues
- **Impact**: No checkpoints saved (risky!)
- **Fix needed**: Alert user, pause training

---

## 6. CHECKPOINT SAVE FAILURE EDGE CASES

### Current Behavior (train.py:635-654)
```python
# Copy optimizer state from checkpoint to root
for filename in ['optimizer.pt', 'scheduler.pt', ...]:
    src = latest_checkpoint / filename
    dst = Path(self.args.output_dir) / filename
    if src.exists():
        shutil.copy2(src, dst)

# Delete checkpoints
for checkpoint_path in checkpoint_paths:
    shutil.rmtree(checkpoint_path)
```

### ❌ NON-ATOMIC OPERATIONS

**1. Copy succeeds, delete fails**
- **What happens**: Checkpoints accumulate on disk
- **Current handling**: ❌ No error handling
- **Impact**: Disk space wasted
- **Fix needed**: Log warning, continue

**2. Partial copy (disk full mid-copy)**
- **What happens**: optimizer.pt copied, scheduler.pt fails
- **Current handling**: ❌ Partial state in root
- **Impact**: Next batch loads mismatched optimizer/scheduler
- **Fix needed**: Atomic copy (all or nothing)

**3. Delete partially succeeds**
- **What happens**: Some checkpoints deleted, others remain
- **Current handling**: ❌ No verification
- **Impact**: Orphaned checkpoint files
- **Fix needed**: Verify all deletes succeeded

---

## 7. CORRUPTED DATA FILE EDGE CASES

### Current Behavior (training_daemon.py:323-403)
```python
def train_on_file(self, data_file: Path):
    # Count lines
    with open(data_file) as f:
        num_examples = sum(1 for _ in f)

    # Train
    try:
        trainer = UltimateTrainer(args)
        success = trainer.run()

        if success:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False
```

### ❌ MISSING VALIDATION

**1. Invalid JSON in JSONL**
- **What happens**: Training starts, crashes on first bad line
- **Current handling**: ❌ No pre-validation
- **Impact**: Wasted GPU time
- **Fix needed**: Quick validation pass before training

**2. Missing required fields**
- **What happens**: Training starts, crashes during data prep
- **Current handling**: ❌ No schema validation
- **Impact**: Wasted time
- **Fix needed**: Validate schema before training

**3. Empty file**
- **What happens**: num_examples = 0, training starts anyway
- **Current handling**: ❌ No check
- **Impact**: Immediate crash or empty training
- **Fix needed**: Check num_examples > 0

---

## 8. RACE CONDITIONS

### Current Behavior
✅ **NO RACE CONDITIONS** - System is entirely synchronous

**Why it's safe:**
1. Daemon runs training in-process (not subprocess)
2. Only one file processed at a time
3. Daemon loop blocks during training
4. No concurrent file access

---

## RECOMMENDED FIXES (Priority Order)

### 🔴 CRITICAL (Fix Now)
1. **Atomic consolidation** (save before delete)
2. **Corrupted checkpoint handling** (validate on load)
3. **Disk space checks** (before training starts)
4. **Cleanup on crash** (delete partial checkpoints)

### 🟡 IMPORTANT (Fix Soon)
5. **Retry logic** (for transient errors)
6. **Snapshot cleanup** (keep only N recent)
7. **Data validation** (before training)
8. **Startup health check** (reset stale status)

### 🟢 NICE TO HAVE
9. **Atomic checkpoint saves** (all or nothing)
10. **Config validation** (on load)
11. **GPU memory cleanup** (on crash)
12. **Better error messages** (actionable guidance)

---

## 3 AM CONSOLIDATION FLOW DIAGRAM

```
03:00:00 - should_consolidate() returns True
03:00:00 - Check inbox empty?
           ├─ YES → Proceed to consolidation
           └─ NO  → Wait for next poll cycle

03:00:05 - Consolidation starts (if inbox empty)
           ├─ Backup adapter → consolidated_backups/TIMESTAMP/
           ├─ Load base model (30 min timeout)
           ├─ Merge adapter into base
           ├─ Save merged model → consolidated_models/TIMESTAMP/
           ├─ Update config.json → point to new base
           └─ Delete current_model/ (adapter removed)

03:15:00 - Consolidation completes
           ├─ Write .last_consolidation marker
           └─ Next training uses merged base (fresh adapter)
```

**EDGE CASE**: If training finishes at 03:10 (after 3 AM but inbox empty), consolidation runs at next poll cycle (03:10:30).

---

## CURRENT STATUS: SYSTEM IS WORKING BUT FRAGILE

**Strengths:**
- ✅ Batch transitions work correctly
- ✅ Optimizer state preserved across batches
- ✅ Consolidation waits for idle time
- ✅ Failed files kept for inspection

**Weaknesses:**
- ❌ No atomic operations (partial state possible)
- ❌ No validation of loaded state
- ❌ No disk space monitoring
- ❌ No cleanup on crash

**Overall Assessment:**
System will work reliably under **normal conditions** but may **fail catastrophically** under stress (disk full, crashes, corruption). Recommend implementing critical fixes before production use.
