# Critical Fixes Implementation Plan
**Priority:** Address top 3 critical issues immediately
**Date:** 2025-11-16

---

## 🔴 FIX #1: Multiple Daemon Detection

### Problem
No PID file locking - multiple daemons could run simultaneously and corrupt model

### Solution
```python
# In TrainingDaemon.__init__():
self.pid_file = self.base_dir / ".daemon.pid"

def acquire_lock(self):
    """Acquire PID file lock - prevents multiple daemons"""
    if self.pid_file.exists():
        try:
            old_pid = int(self.pid_file.read_text().strip())
            # Check if process is still running
            os.kill(old_pid, 0)  # Doesn't kill, just checks
            self.logger.error(f"❌ Another daemon is running (PID {old_pid})")
            self.logger.error("   Stop it first or remove .daemon.pid if stale")
            sys.exit(1)
        except (OSError, ValueError):
            # Process not running, clean up stale PID file
            self.logger.warning(f"⚠️  Removing stale PID file (old PID: {old_pid})")
            self.pid_file.unlink()

    # Write our PID
    self.pid_file.write_text(str(os.getpid()))
    self.logger.info(f"✅ Acquired daemon lock (PID: {os.getpid()})")

def release_lock(self):
    """Release PID file lock"""
    if self.pid_file.exists():
        self.pid_file.unlink()
    self.logger.info("✅ Released daemon lock")

# In run() method:
try:
    self.acquire_lock()
    # ... existing daemon loop ...
finally:
    self.release_lock()
```

### Testing
```bash
# Should fail with error message
python3 training_daemon.py &
sleep 2
python3 training_daemon.py  # Should exit with "Another daemon running"
```

---

## 🔴 FIX #2: Crash Recovery - Orphaned Processing Files

### Problem
If daemon crashes, files stuck in `queue/processing/` forever

### Solution
```python
def recover_orphaned_files(self):
    """Move orphaned files from processing/ back to normal queue on startup"""
    processing_files = list(self.queue.processing.glob("*.jsonl"))

    if processing_files:
        self.logger.warning(f"⚠️  Found {len(processing_files)} orphaned files from previous crash")
        for file_path in processing_files:
            target = self.queue.normal_priority / file_path.name
            shutil.move(str(file_path), str(target))
            self.logger.info(f"   Recovered: {file_path.name}")
        self.logger.info("✅ Crash recovery complete")

# Call in run() before main loop:
def run(self):
    self.setup_directories()
    self.initialize_model()
    self.acquire_lock()

    # NEW: Recover from previous crash
    self.recover_orphaned_files()

    # ... rest of daemon loop ...
```

### Testing
```bash
# Simulate crash
kill -9 <daemon_pid>
ls queue/processing/  # Should show orphaned file
python3 training_daemon.py  # Should recover file to queue/normal/
```

---

## 🔴 FIX #3: Disk Space Checks

### Problem
No checking if disk is almost full - could corrupt checkpoints

### Solution
```python
def check_disk_space(self) -> bool:
    """Check if enough disk space available"""
    stat = os.statvfs(self.base_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

    if free_gb < 10:
        self.logger.error(f"❌ CRITICAL: Only {free_gb:.1f}GB free disk space")
        self.logger.error("   Need at least 10GB for safe checkpoint saves")
        return False

    if free_gb < 50:
        self.logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free")

    return True

# Call before each training operation:
def train_on_file(self, data_file, ...):
    # Check disk space before training
    if not self.check_disk_space():
        self.logger.error(f"⚠️  Skipping {data_file.name} - insufficient disk space")
        self.queue.mark_failed(data_file, error="Insufficient disk space")
        return False

    # ... rest of training ...
```

### Testing
```bash
# Artificially fill disk (be careful!)
df -h /path/to/training
# Verify daemon refuses to train when <10GB free
```

---

## 🟡 FIX #4: Exception Handling in Main Loop

### Problem
Any unhandled exception crashes entire daemon

### Solution
```python
def run(self):
    # ... setup ...

    while True:
        try:
            iteration += 1

            # All existing daemon logic here...
            # (Check signals, process queue, etc.)

        except KeyboardInterrupt:
            self.logger.info("⚠️  Keyboard interrupt received")
            break
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in daemon loop: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            self.logger.error("   Daemon will retry in 60 seconds...")
            self.controller.update_state("error", reason=str(e))
            time.sleep(60)  # Wait before retrying
            # Loop continues - daemon doesn't die!
```

---

## 🟡 FIX #5: Failed File Retry Limits

### Problem
Failed files retry forever

### Solution
```python
# In training_queue.py metadata:
{
    "failed": [
        {
            "file": "bad.jsonl",
            "attempts": 3,  # NEW: Track attempts
            "last_error": "Training failed",
            "final_failure": "2025-11-16T..."  # NEW: When gave up
        }
    ]
}

# In mark_failed():
def mark_failed(self, file_path: Path, error: str = "Training failed"):
    metadata = self._get_metadata()

    # Check existing failures
    attempts = 1
    for fail in metadata.get("failed", []):
        if fail["file"] == file_path.name:
            attempts = fail.get("attempts", 0) + 1

    if attempts >= 3:
        # Give up - move to permanent failure
        failed_dir = self.queue_dir / "failed"
        failed_dir.mkdir(exist_ok=True)
        target = failed_dir / file_path.name
        shutil.move(str(file_path), str(target))

        metadata["failed"].append({
            "file": file_path.name,
            "attempts": attempts,
            "last_error": error,
            "final_failure": datetime.now().isoformat()
        })
        logger.error(f"❌ PERMANENTLY FAILED: {file_path.name} ({attempts} attempts)")
    else:
        # Retry - move back to low priority
        target = self.low_priority / file_path.name
        shutil.move(str(file_path), str(target))

        metadata["failed"].append({
            "file": file_path.name,
            "attempts": attempts,
            "last_error": error,
            "retry_at": datetime.now().isoformat()
        })
        logger.warning(f"⚠️  Failed: {file_path.name} (attempt {attempts}/3, will retry)")

    self._save_metadata(metadata)
```

---

## 🟡 FIX #6: Signal Handlers

### Problem
SIGTERM/SIGINT ignored - can't stop gracefully via system signals

### Solution
```python
import signal

class TrainingDaemon:
    def __init__(self, base_dir):
        # ... existing init ...
        self.shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        sig_name = signal.Signals(signum).name
        self.logger.info(f"⚠️  Received {sig_name} - will stop after current batch")
        self.shutdown_requested = True

    def should_stop(self):
        """Check if daemon should stop"""
        return (self.stop_file.exists() or
                self.shutdown_requested or
                self.controller.should_stop_after_batch())
```

Now `Ctrl+C` and `systemctl stop` work gracefully!

---

## 🟡 FIX #7: Stale State Cleanup

### Problem
Controller state shows "training" after crash

### Solution
```python
def cleanup_stale_state(self):
    """Clean up stale state from previous crash"""
    state = self.controller._load_state()

    if state.get("status") == "training":
        self.logger.warning("⚠️  Previous daemon crashed while training")
        self.controller.update_state("idle", reason="Recovered from crash")

    # Clear any stale signals
    self.controller.clear_signals()
    self.logger.info("✅ State cleanup complete")

# Call in run() at startup:
def run(self):
    self.setup_directories()
    self.initialize_model()
    self.acquire_lock()
    self.recover_orphaned_files()
    self.cleanup_stale_state()  # NEW
    # ... rest of daemon ...
```

---

## 🟡 FIX #8: Consolidation Queue Bug

### Problem
Checks inbox empty but not queue processing status

### Solution
```python
# In run() loop:
# Check if we should consolidate (when idle after 3 AM)
# FIXED: Check both inbox AND queue are empty
inbox_files_check = self.get_inbox_files()
queue_status = self.queue.get_queue_status()

if (not inbox_files_check and
    queue_status["total_queued"] == 0 and
    queue_status["processing"] == 0 and  # NEW: Check processing too!
    self.should_consolidate()):
    self.consolidate_model()
```

---

## Implementation Priority

1. ✅ **FIX #1** - Multiple daemon detection (5 min)
2. ✅ **FIX #2** - Crash recovery (10 min)
3. ✅ **FIX #3** - Disk space checks (5 min)
4. ✅ **FIX #4** - Exception handling (5 min)
5. **FIX #5** - Retry limits (15 min)
6. **FIX #6** - Signal handlers (10 min)
7. **FIX #7** - State cleanup (5 min)
8. **FIX #8** - Consolidation bug (2 min)

**Total Time: ~1 hour**

---

## Testing Checklist

After implementing:
- [ ] Test multiple daemon prevention
- [ ] Test crash recovery (kill -9)
- [ ] Test disk full scenario
- [ ] Test SIGTERM graceful shutdown
- [ ] Test failed file retry limits
- [ ] Test stale state cleanup
- [ ] Monitor for exceptions in logs

**Do these fixes now or wait?**
- Current training: Step 1054/2487 - will finish in ~2 hours
- Option A: Fix now (requires restart, training resumes from checkpoint)
- Option B: Fix after current batch completes
