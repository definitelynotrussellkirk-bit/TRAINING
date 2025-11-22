# COMPREHENSIVE EDGE CASE ANALYSIS
**Last Updated:** 2025-11-16
**Purpose:** Track all potential failure modes and safeguards

---

## ✅ ALREADY IMPLEMENTED SAFEGUARDS

### 1. **PID Lock File** (`training_daemon.py:812-835`)
- **Prevents:** Multiple daemons running simultaneously
- **How:** Writes PID to `.daemon.pid`, checks if process still running
- **Status:** ✅ COMPLETE

### 2. **Disk Space Pre-Flight** (`training_daemon.py:861-873`)
- **Prevents:** Training failure due to disk full
- **How:** Checks free space before training, aborts if < 20GB
- **Status:** ✅ COMPLETE

### 3. **Atomic File Writes** (`atomic_ops.py`)
- **Prevents:** Corruption from partial writes (crashes mid-write)
- **How:** Write to .tmp file, then atomic rename
- **Status:** ✅ COMPLETE (for JSON, text, directory copies)

### 4. **Checkpoint Auto-Resume** (`train.py:963-974`)
- **Prevents:** Losing progress on crash
- **How:** Automatically resumes from latest checkpoint
- **Status:** ✅ COMPLETE

### 5. **Data Validation** (`validate_data.py`)
- **Prevents:** Training on data that exceeds max_length
- **How:** Samples 100 examples, checks token lengths, aborts if >95% truncated
- **Status:** ✅ COMPLETE

### 6. **Backup Before Consolidation** (`consolidate_model.py`)
- **Prevents:** Losing model during merge operations
- **How:** Creates verified backup before any deletion
- **Status:** ✅ COMPLETE

### 7. **Version System** (`model_versioner.py`)
- **Prevents:** Overwriting previous models
- **How:** Numbered versions (v001, v002, etc.) with metadata
- **Status:** ✅ COMPLETE

### 8. **Queue System** (`training_queue.py`)
- **Prevents:** Race conditions on file processing
- **How:** Atomic queue operations with priority support
- **Status:** ✅ COMPLETE

### 9. **Graceful Control** (`training_controller.py`)
- **Prevents:** Data loss from forced kills
- **How:** Signal-based pause/stop, finishes current batch
- **Status:** ✅ COMPLETE

---

## ❌ MISSING CRITICAL SAFEGUARDS

### 10. **Dataset Hash Tracking in Checkpoints** ✅ IMPLEMENTED (2025-11-16)
- **Problem:** Switching datasets with existing checkpoints causes step counter mismatch
- **Example:** Step 4000 checkpoint + 2487-step dataset → thinks already done
- **Impact:** Just encountered this bug - training completed in 0.9min without doing anything
- **Fix:** Store dataset hash in checkpoint metadata, auto-clear if mismatch
- **Priority:** 🔥 **CRITICAL** - This breaks continuous training on new datasets
- **Status:** ✅ COMPLETE - `dataset_hash.py` + integration in `train.py`
- **Testing:** ✅ All test scenarios pass (see `test_dataset_hash.py`)

### 11. **Checkpoint Integrity Verification** ⚠️ HIGH
- **Problem:** Checkpoint corruption (crash during save) not detected
- **Impact:** Load corrupt checkpoint → NaN losses → waste hours of GPU time
- **Fix:** Add sha256 hash to checkpoint metadata, verify on load
- **Priority:** ⚠️ **HIGH** - Rare but catastrophic

### 12. **LoRA Config Validation** ⚠️ HIGH
- **Problem:** Checkpoint from r=64 LoRA used with r=128 config
- **Impact:** Crash or silent failure after hours of loading
- **Fix:** Store LoRA config in checkpoint, validate on load
- **Priority:** ⚠️ **HIGH** - Easy to misconfigure

### 13. **Consolidation Disk Space Check** ⚠️ HIGH
- **Problem:** Consolidation requires 3x model size (merge creates new copy)
- **Impact:** Disk full mid-consolidation → corrupt model
- **Fix:** Pre-flight check: free_space > 3 * model_size
- **Priority:** ⚠️ **HIGH** - Models are 30GB+, easy to run out

### 14. **Data Diversity Validation** 🟡 MEDIUM
- **Problem:** All examples identical (degenerate dataset)
- **Impact:** Waste GPU time on useless training
- **Fix:** Check unique count > 0.5 * total count
- **Priority:** 🟡 **MEDIUM** - Unlikely but wasteful

### 15. **Contradictory Labels Detection** 🟡 MEDIUM
- **Problem:** Same input, different outputs (contradictory training signal)
- **Impact:** Confuses model, poor quality
- **Fix:** Hash prompts, detect duplicates with different answers, warn user
- **Priority:** 🟡 **MEDIUM** - Data quality issue

### 16. **OOM Recovery** 🟡 MEDIUM
- **Problem:** CUDA OOM causes infinite crash loop if repeatable
- **Impact:** Training stuck, can't proceed
- **Fix:** Detect OOM, reduce batch_size by 50%, retry
- **Priority:** 🟡 **MEDIUM** - Already at batch_size=1, limited value

### 17. **Validation Loss NaN Alert** 🟡 MEDIUM
- **Problem:** Validation loss NaN means model diverged
- **Impact:** Hard to detect model failure
- **Fix:** Alert user, suggest stopping training
- **Priority:** 🟡 **MEDIUM** - Easy to spot manually

### 18. **Queue Deduplication** 🟢 LOW
- **Problem:** Same file added multiple times
- **Impact:** Train on it multiple times (wasteful)
- **Fix:** Deduplicate by filename hash
- **Priority:** 🟢 **LOW** - User can avoid this

### 19. **UTC Scheduling for Snapshots** 🟢 LOW
- **Problem:** Daylight saving time at 3am causes double/missed snapshot
- **Impact:** Minor - extra or missed backup
- **Fix:** Use UTC for all scheduling
- **Priority:** 🟢 **LOW** - Happens twice a year

### 20. **System RAM Pre-Flight** 🟢 LOW
- **Problem:** System OOM (not CUDA) kills daemon
- **Impact:** Training stops silently
- **Fix:** Check RAM < 85% before training
- **Priority:** 🟢 **LOW** - Rare on dedicated training machine

---

## 🔥 IMMEDIATE ACTION ITEMS (Top 3)

### Priority 1: Dataset Hash Tracking (Bug we just hit!)
**File:** `train.py`
**Changes:**
```python
# In checkpoint metadata, add:
{
  "dataset_hash": hashlib.md5(dataset_path.read_bytes()).hexdigest(),
  "dataset_name": dataset_path.name,
  "lora_config": {"r": 128, "alpha": 128, ...}
}

# On resume:
if checkpoint_dataset_hash != current_dataset_hash:
    logger.warning("Dataset changed - clearing checkpoints")
    clear_checkpoints()
```

### Priority 2: LoRA Config Validation
**File:** `train.py`
**Changes:**
```python
# On checkpoint load:
checkpoint_lora = load_checkpoint_metadata()["lora_config"]
if checkpoint_lora != current_lora_config:
    raise ValueError(f"LoRA mismatch: checkpoint has r={checkpoint_lora['r']}, config has r={current_lora_config['r']}")
```

### Priority 3: Checkpoint Integrity Hash
**File:** `train.py`
**Changes:**
```python
# After saving checkpoint:
checkpoint_hash = hash_directory(checkpoint_path)
save_metadata({"integrity_hash": checkpoint_hash})

# Before loading:
if not verify_checkpoint_integrity(checkpoint_path):
    logger.error("Checkpoint corrupted - starting fresh")
    remove_checkpoint()
```

---

## 📊 EDGE CASE CATEGORIES

### Category: Checkpoint/State Issues
| Edge Case | Status | Priority | Impact |
|-----------|--------|----------|--------|
| Checkpoint corruption | ❌ Missing | HIGH | NaN losses, wasted GPU time |
| LoRA config mismatch | ❌ Missing | HIGH | Crash after loading |
| **Dataset hash mismatch** | ❌ Missing | **CRITICAL** | **Silent training failure** |
| Disk full during save | ✅ Atomic writes | - | Prevented |
| Multiple daemon instances | ✅ PID lock | - | Prevented |

### Category: Data Quality Issues
| Edge Case | Status | Priority | Impact |
|-----------|--------|----------|--------|
| Empty files | ✅ Validation | - | Caught early |
| Malformed JSON | ✅ Validation | - | Caught early |
| Exceeds max_length | ✅ Validation | - | Warned/aborted |
| All examples identical | ❌ Missing | MEDIUM | Waste GPU time |
| Contradictory labels | ❌ Missing | MEDIUM | Poor model quality |

### Category: Resource Issues
| Edge Case | Status | Priority | Impact |
|-----------|--------|----------|--------|
| Disk space low | ✅ Pre-flight check | - | Prevented |
| GPU OOM | ❌ Partial (crash→resume) | MEDIUM | Crash loop possible |
| System OOM | ❌ Missing | LOW | Silent failure |
| Consolidation disk full | ❌ Missing | HIGH | Model corruption |

### Category: Process/Concurrency Issues
| Edge Case | Status | Priority | Impact |
|-----------|--------|----------|--------|
| Multiple daemons | ✅ PID lock | - | Prevented |
| Race on queue files | ✅ Atomic queue | - | Prevented |
| Partial file writes | ✅ Atomic writes | - | Prevented |
| Signal handling | ✅ Graceful control | - | Prevented |

---

## 🧪 TESTING PROTOCOL

### For Each New Safeguard:
1. **Unit test** - Test the specific safeguard in isolation
2. **Integration test** - Test with real training pipeline
3. **Chaos test** - Deliberately trigger the failure mode
4. **Recovery test** - Verify system recovers gracefully

### Example: Dataset Hash Tracking Test
```python
def test_dataset_hash_mismatch():
    # Train on dataset A
    train("dataset_a.jsonl")

    # Verify checkpoint exists
    assert checkpoint_exists()

    # Switch to dataset B (different size)
    train("dataset_b.jsonl")

    # Verify: should clear checkpoints and start fresh
    assert checkpoint_step() == 0
    assert training_completed_successfully()
```

---

## 📝 LESSONS LEARNED

### 2025-11-16: Dataset Switching Bug
- **What happened:** User switched from LEO data to SYLLO data, training "completed" in 0.9 minutes without doing anything
- **Root cause:** Checkpoint at step 4000 + new dataset with 2487 steps → system thought already done
- **Why missed:** Tested same-dataset continuation, but not dataset switching
- **Fix:** Manually cleared checkpoints (temporary), need automatic dataset hash tracking
- **Prevention:** Add dataset hash to checkpoint metadata, auto-clear on mismatch

### Key Insight:
**"Continuous training" doesn't just mean "add more batches of same data"**
- Users will switch between different datasets (SYLLO, LEO, math, reasoning, etc.)
- Each dataset is a separate "task" the model should learn
- Checkpoints must be dataset-aware

---

## 🎯 DESIGN PRINCIPLES FOR SAFEGUARDS

1. **Fail Fast** - Detect problems early, before wasting GPU time
2. **Fail Safe** - When failure occurs, preserve data (never delete without backup)
3. **Fail Loud** - Log errors clearly, don't fail silently
4. **Atomic Operations** - Use tmp + rename for all critical writes
5. **Verification** - Check integrity before and after operations
6. **Graceful Degradation** - Training can continue even if monitoring fails
7. **Recovery Paths** - Always provide a way to recover from failure

---

## 📋 IMPLEMENTATION CHECKLIST

- [x] PID lock file
- [x] Disk space pre-flight
- [x] Atomic file writes
- [x] Data validation (max_length)
- [x] Backup before consolidation
- [x] Version system
- [x] Queue system
- [x] Graceful control
- [x] **Dataset hash tracking** 🔥 **IMPLEMENTED 2025-11-16**
- [x] **LoRA config validation** (part of dataset hash tracking)
- [ ] Checkpoint integrity verification
- [ ] Consolidation disk space check
- [ ] Data diversity validation
- [ ] Contradictory labels detection
- [ ] OOM recovery
- [ ] Validation loss NaN alert
- [ ] Queue deduplication
- [ ] UTC scheduling

---

## 🚨 IF YOU'RE A FUTURE CLAUDE READING THIS

**Before making changes:**
1. Read this document completely
2. Check if the safeguard already exists
3. If implementing new safeguard, update this document
4. Add tests to edge case test suite
5. Document in session summary

**If user reports a bug:**
1. Add it to this document immediately
2. Categorize it (checkpoint/data/resource/process)
3. Determine if it's already supposed to be prevented
4. If safeguard exists but failed, investigate why
5. Update safeguard AND add specific test for this case

**Remember:**
- The user lost 3+ weeks of training to preventable mistakes
- Every safeguard here exists because of a real failure
- Don't assume you've thought of everything - you haven't
- Test edge cases explicitly, don't assume they work


## CRITICAL BUG DISCOVERED (2025-11-16 19:15)

**Problem:** Training consistently hangs at step 10 (and 20, 25, etc.)

**Symptoms:**
- Steps 1-9: Normal (~9s each)
- Step 10: Hangs indefinitely (tested with log_steps=10 AND log_steps=200)
- Same behavior repeats at steps 20, 25

**Not caused by:**
- ❌ log_steps callback (still hangs with log_steps=200)
- ❌ Dataset hash tracking (code runs before training loop)
- ❌ Validation loss (only runs every 200 steps)

**Possible causes to investigate:**
1. Detail collector (updates every 50 steps)
2. Gradient checkpointing warning triggers issue
3. CUDA memory/synchronization problem
4. Status file write blocking
5. Evolution tracker

**Workaround:**
- Use last working checkpoint (has step 25 progress)
- Need full debugging session to fix root cause


