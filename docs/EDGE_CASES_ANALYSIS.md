# Edge Cases & Failure Modes Analysis
**Date:** 2025-11-16
**Status:** Analysis of integrated queue/control system

---

## 🔴 CRITICAL ISSUES (Could cause data loss or corruption)

### 1. **Multiple Daemon Instances**
**Problem:** No detection if multiple daemons are running simultaneously
**Impact:**
- Two daemons could process same file from queue
- Race conditions on checkpoint writes
- Corrupted model state

**Current Behavior:** Silently runs multiple instances
**Mitigation Needed:** PID file lock + startup check

**Test:**
```bash
# Start two daemons - both will run!
python3 training_daemon.py --base-dir /path/to/training &
python3 training_daemon.py --base-dir /path/to/training &
```

**Fix Priority:** 🔴 HIGH

---

### 2. **Daemon Crash Leaves Files Stuck in Processing**
**Problem:** If daemon crashes mid-training, file stays in `queue/processing/` forever
**Impact:**
- File never gets retrained
- Queue thinks it's still processing
- No automatic recovery

**Current Behavior:** File orphaned in processing/
**Mitigation Needed:** Startup recovery - move processing/* back to queue

**Test:**
```bash
# Kill daemon mid-training
kill -9 <daemon_pid>
ls queue/processing/  # File stuck here forever
```

**Fix Priority:** 🔴 HIGH

---

### 3. **No Disk Space Checks**
**Problem:** Training continues even when disk is nearly full
**Impact:**
- Checkpoint save fails → training lost
- Model corruption if save partially completes
- System-wide issues if root partition fills

**Current Behavior:** Continues until write fails
**Mitigation Needed:** Pre-flight disk space check, abort if <10GB free

**Fix Priority:** 🔴 HIGH

---

## 🟡 SERIOUS ISSUES (Could cause training failures)

### 4. **Unhandled Exceptions Crash Daemon**
**Problem:** No try/except around main loop
**Impact:**
- Any unhandled error kills entire daemon
- Training stops until manual restart
- No error recovery

**Current Code:**
```python
while True:
    # No try/except here!
    self.queue.process_inbox()  # Could throw
    queue_status = self.queue.get_queue_status()  # Could throw
    # ... etc
```

**Mitigation Needed:** Wrap main loop in try/except with logging

**Fix Priority:** 🟡 MEDIUM-HIGH

---

### 5. **Failed Files Retry Infinitely**
**Problem:** `mark_failed()` keeps file but no retry limit
**Impact:**
- Same bad file retried forever
- Blocks other training
- Wastes GPU time

**Current Behavior:**
```python
self.queue.mark_failed(data_file, error="Training failed", keep_file=True)
# File goes... where? Back to queue? How many retries?
```

**Mitigation Needed:**
- Max retry counter (3 attempts)
- Move to `queue/failed/` after max retries
- Alert user

**Fix Priority:** 🟡 MEDIUM

---

### 6. **No Signal Handling (SIGTERM/SIGINT)**
**Problem:** Only checks `.stop` file, ignores system signals
**Impact:**
- `systemctl stop` doesn't work gracefully
- Ctrl+C in terminal doesn't stop cleanly
- System shutdown kills daemon hard

**Current Behavior:** Daemon ignores SIGTERM/SIGINT
**Mitigation Needed:** Signal handlers that set stop flag

**Fix Priority:** 🟡 MEDIUM

---

### 7. **Stale Controller State After Crash**
**Problem:** If daemon crashes, `control/state.json` shows "training" forever
**Impact:**
- Confusing status output
- Scripts waiting for "idle" hang forever
- No way to detect crash vs. active training

**Current Behavior:** State file never cleaned up
**Mitigation Needed:** Startup checks state.json, resets if stale (PID check)

**Fix Priority:** 🟡 MEDIUM

---

### 8. **Checkpoint Corruption During Crash**
**Problem:** If daemon crashes during checkpoint save, file may be corrupt
**Impact:**
- Training can't resume
- Model state lost
- No detection of corruption

**Current Behavior:** Uses corrupt checkpoint if it exists
**Mitigation Needed:** Atomic checkpoint writes (temp file + rename)

**Fix Priority:** 🟡 MEDIUM

---

## 🟢 MINOR ISSUES (Edge cases, unlikely but possible)

### 9. **Race Condition: File Deleted Between Queue Operations**
**Problem:**
```python
queue_status = self.queue.get_queue_status()  # File exists
# ... other code ...
data_file = self.queue.get_next_file()  # File deleted? Returns None?
```

**Impact:** Loop breaks, skips remaining queue
**Mitigation:** Already handled (get_next_file returns None → break)

**Fix Priority:** 🟢 LOW (Already handled)

---

### 10. **Conflicting Control Signals**
**Problem:** What if both pause AND stop signals exist?
**Current Code:**
```python
if self.controller.should_stop_after_batch():
    # Stop takes priority
    break
if self.controller.should_pause_after_batch():
    # This never runs if stop was set
```

**Impact:** Stop takes priority (reasonable)
**Mitigation:** Document behavior

**Fix Priority:** 🟢 LOW (Acceptable behavior)

---

### 11. **Skip Signal Set When Not Training**
**Problem:** User sets skip signal but no file is being processed
**Impact:** Signal ignored, stays set, cleared on next file

**Current Behavior:** Signal checked only after training fails
**Mitigation:** Clear stale signals on startup

**Fix Priority:** 🟢 LOW

---

### 12. **Malformed JSONL Files**
**Problem:** File exists but JSON is invalid
**Impact:** Training fails, file marked as failed

**Current Behavior:** Validation detects this, training aborts
**Mitigation:** Already handled by validation

**Fix Priority:** 🟢 LOW (Already handled)

---

### 13. **Empty Training Files**
**Problem:** File exists but has 0 lines
**Impact:** Training fails with "no examples"

**Current Behavior:** Likely crashes with division by zero
**Mitigation Needed:** Check num_examples > 0 before training

**Fix Priority:** 🟢 LOW

---

### 14. **Clock Skew Issues**
**Problem:** System clock jumps (NTP sync, manual change)
**Impact:**
- Daily snapshot time check fails
- Consolidation time check fails

**Current Behavior:** Uses `datetime.now()` comparisons
**Mitigation:** Unlikely in practice

**Fix Priority:** 🟢 LOW

---

### 15. **GPU Memory Leak**
**Problem:** Gradual memory accumulation over many batches
**Impact:** Eventually OOM crash

**Current Behavior:** May already exist (hard to detect)
**Mitigation:** Monitor VRAM over time, restart daemon daily?

**Fix Priority:** 🟢 LOW (Monitoring would detect)

---

### 16. **Permission Errors on Queue Operations**
**Problem:** File permissions change, can't move files
**Impact:** Queue operation fails

**Current Behavior:** Exception → daemon crashes (see #4)
**Mitigation:** Fix #4 (exception handling)

**Fix Priority:** 🟢 LOW

---

### 17. **Consolidation Fails During Training**
**Problem:** Consolidation starts while file is in processing (race)
**Impact:** Depends on timing

**Current Behavior:**
```python
inbox_files_check = self.get_inbox_files()
if not inbox_files_check and self.should_consolidate():
```

**Issue:** Checks inbox, not queue processing status!
**Mitigation Needed:** Check queue.processing is also empty

**Fix Priority:** 🟡 MEDIUM

---

## 🧪 RECOMMENDED TESTS

### Test 1: Multiple Daemon Detection
```bash
./test_multiple_daemons.sh
```

### Test 2: Crash Recovery
```bash
# Start training, kill -9 mid-batch, restart daemon
# Verify: File moved from processing/ back to queue
```

### Test 3: Disk Full
```bash
# Fill disk to 1GB free, start training
# Verify: Daemon aborts with clear error
```

### Test 4: Infinite Retry
```bash
# Add corrupted file, verify max 3 retries then moved to failed/
```

### Test 5: Signal Handling
```bash
kill -TERM <daemon_pid>
# Verify: Graceful shutdown after current batch
```

---

## 📋 PRIORITY FIX LIST

1. 🔴 **Multiple daemon detection** (data corruption risk)
2. 🔴 **Crash recovery for processing/ files**
3. 🔴 **Disk space checks**
4. 🟡 **Exception handling in main loop**
5. 🟡 **Failed file retry limits**
6. 🟡 **Signal handlers (SIGTERM/SIGINT)**
7. 🟡 **Stale state cleanup on startup**
8. 🟡 **Consolidation queue check bug**

---

## 📝 NOTES

- Most edge cases are **detectable but not handled**
- Main risk: **Daemon crash = silent failure**
- Best mitigation: **Monitoring + auto-restart**
- Consider: **Systemd service with auto-restart**
