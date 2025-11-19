# CRASH PREVENTION & RECOVERY GUIDE

**Created:** 2025-11-17
**Context:** After multiple daemon crashes, orphaned processes, and training hangs

This guide documents the complete crash prevention and auto-recovery system built to prevent future failures.

---

## 🚨 WHAT HAPPENED (2025-11-17)

### The Crash
- **Time:** ~9:46 PM EST
- **Symptom:** Training stopped at step 2500 during evaluation
- **Root Cause:** Daemon had crashed earlier (~30 hours before), training was running as orphaned process
- **Impact:** Lost ~93 steps of progress (2600-2693), had to restart from checkpoint-2600

### The Pattern
This was NOT an isolated incident. The system has repeatedly experienced:
1. Daemon crashes leaving orphaned training processes
2. Orphaned processes running unstable without oversight
3. No auto-restart or recovery mechanism
4. Silent failures with no alerts
5. Disk space bloat from accumulating checkpoints

---

## ✅ SOLUTIONS IMPLEMENTED

### 1. Daemon Watchdog (`daemon_watchdog.py`)

**Purpose:** Monitor daemon health and auto-restart on crash

**Features:**
- Checks daemon every 30 seconds
- Detects crashes and hung training
- Auto-restarts daemon with rate limiting
- Kills orphaned processes before restart
- Comprehensive logging

**Usage:**
```bash
# Start watchdog in background
nohup python3 daemon_watchdog.py > logs/watchdog.log 2>&1 &

# Or run in foreground for monitoring
python3 daemon_watchdog.py
```

**Configuration:**
- `CHECK_INTERVAL = 30` - Check every 30 seconds
- `PROGRESS_TIMEOUT = 600` - 10 min without progress = hung
- `MAX_RESTART_ATTEMPTS = 3` - Max restarts in 5 minutes
- `RESTART_WINDOW = 300` - 5 minute window

**What it prevents:**
- ✅ Daemon crashes going unnoticed
- ✅ Orphaned processes running indefinitely
- ✅ Training hangs without detection
- ✅ Lost progress from crashed processes

---

### 2. Crash Detector (`crash_detector.py`)

**Purpose:** Analyze logs to detect and categorize crashes

**Features:**
- Scans all logs for crash patterns
- Categorizes crashes (OOM, CUDA, timeout, etc.)
- Provides recovery suggestions
- Tracks crash history

**Usage:**
```bash
# Analyze recent logs
python3 crash_detector.py

# Only check last 1000 lines
python3 crash_detector.py --last-n-lines 1000

# Quiet mode (summary only)
python3 crash_detector.py --quiet
```

**Detected crash types:**
- CUDA OOM
- CUDA errors
- Process killed (OOM killer)
- Timeouts
- Import errors
- File not found
- Permission errors
- Multiprocessing errors
- Segfaults
- Disk full
- And more...

**What it provides:**
- ✅ Crash type identification
- ✅ Specific recovery suggestions
- ✅ Crash frequency tracking
- ✅ Pattern detection

---

### 3. Comprehensive Health Check (`comprehensive_health_check.py`)

**Purpose:** Test all system components for common failures

**Features:**
- 10+ health checks
- Auto-fix mode for simple issues
- Detailed reporting
- Exit codes for automation

**Usage:**
```bash
# Run all health checks
python3 comprehensive_health_check.py

# Auto-fix issues where possible
python3 comprehensive_health_check.py --fix
```

**Checks performed:**
1. Directory structure (inbox, logs, status, etc.)
2. Required files (config.json, train.py, etc.)
3. Disk space (warns if <50GB)
4. System memory (warns if <4GB)
5. GPU availability and status
6. Daemon process health
7. Training progress
8. Config validity
9. Checkpoint integrity
10. Orphaned processes

**What it prevents:**
- ✅ Missing directories causing crashes
- ✅ Disk full errors
- ✅ Out of memory issues
- ✅ Invalid config causing failures
- ✅ Corrupted checkpoints

---

### 4. Edge Case Testing (`test_edge_cases.py`)

**Purpose:** Simulate and test failure scenarios

**Features:**
- Tests common failure modes
- Validates recovery mechanisms
- Can run specific tests
- Dry-run mode for safety

**Usage:**
```bash
# Run all edge case tests
python3 test_edge_cases.py

# Run specific test
python3 test_edge_cases.py --test "gpu availability"

# Dry run (show without executing)
python3 test_edge_cases.py --dry-run
```

**Tests performed:**
1. Config corruption detection
2. Missing base model detection
3. Disk space monitoring
4. Checkpoint recovery
5. Orphaned process cleanup
6. GPU availability
7. Queue system functionality
8. Status file updates
9. Daemon crash recovery (optional)

**What it validates:**
- ✅ System detects failures correctly
- ✅ Recovery mechanisms work
- ✅ Edge cases handled gracefully

---

## 🛠️ RECOMMENDED WORKFLOW

### Daily Startup
```bash
# 1. Run health check
python3 comprehensive_health_check.py

# 2. If issues found, fix them
python3 comprehensive_health_check.py --fix

# 3. Start watchdog (if not already running)
nohup python3 daemon_watchdog.py > logs/watchdog.log 2>&1 &

# 4. Verify training is running
ps aux | grep training_daemon | grep -v grep
```

### After a Crash
```bash
# 1. Detect crash type
python3 crash_detector.py

# 2. Read recovery suggestions
# (crash_detector will print them)

# 3. Clean up orphaned processes
pkill -f train.py

# 4. Restart daemon (watchdog will do this automatically)
# Or manually:
rm .daemon.pid
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# 5. Verify recovery
python3 comprehensive_health_check.py
```

### Weekly Maintenance
```bash
# 1. Check crash history
python3 crash_detector.py

# 2. Clean old checkpoints if needed
# (Manual for now - auto-cleanup coming soon)

# 3. Review watchdog logs
tail -100 logs/watchdog.log

# 4. Run edge case tests
python3 test_edge_cases.py
```

---

## 📊 MONITORING BEST PRACTICES

### What to Monitor

1. **Daemon Health** (via watchdog)
   - Check `logs/watchdog.log` for restart events
   - Alert if >3 restarts in 5 minutes

2. **Training Progress** (via status file)
   - Verify step count increasing
   - Check loss decreasing
   - Monitor for hangs (no progress for 10+ min)

3. **System Resources**
   - Disk space: Should have >50GB free
   - Memory: Should have >4GB available
   - GPU: Should be <85°C, <95% memory

4. **Checkpoints**
   - Latest checkpoint should be <24h old
   - Should have valid files (adapter_model.safetensors, etc.)
   - Don't accumulate >30 checkpoints (cleanup)

### Automated Monitoring

**Option 1: Cron jobs**
```bash
# Add to crontab (crontab -e)

# Health check every hour
0 * * * * cd /path/to/training && python3 comprehensive_health_check.py --quiet >> logs/health_check.log 2>&1

# Crash detection every 6 hours
0 */6 * * * cd /path/to/training && python3 crash_detector.py --quiet >> logs/crash_history.log 2>&1

# Edge case tests daily
0 3 * * * cd /path/to/training && python3 test_edge_cases.py >> logs/edge_case_tests.log 2>&1
```

**Option 2: Systemd service** (for watchdog)
```ini
# /etc/systemd/system/training-watchdog.service
[Unit]
Description=Training Daemon Watchdog
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/path/to/training
ExecStart=/usr/bin/python3 /path/to/training/daemon_watchdog.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 🔍 COMMON FAILURE SCENARIOS

### Scenario 1: Daemon Crashes
**Detection:** Watchdog detects missing PID or dead process
**Recovery:** Watchdog auto-restarts daemon
**Prevention:** Monitor watchdog logs for crash patterns

### Scenario 2: Training Hangs
**Detection:** Watchdog detects no progress for 10+ minutes
**Recovery:** Watchdog restarts daemon
**Prevention:** Identify slow operations in code, optimize eval frequency

### Scenario 3: Out of Memory (GPU)
**Detection:** Crash detector finds "CUDA out of memory" in logs
**Recovery:** Reduce batch_size or max_length in config
**Prevention:** Monitor GPU memory usage, set appropriate limits

### Scenario 4: Out of Memory (System RAM)
**Detection:** Health check warns of low memory, or process killed
**Recovery:** Close other programs, add swap space
**Prevention:** Monitor system memory in health checks

### Scenario 5: Disk Full
**Detection:** Health check warns of low disk space
**Recovery:** Clean old checkpoints, free up space
**Prevention:** Monitor disk usage, implement auto-cleanup

### Scenario 6: Orphaned Processes
**Detection:** Health check or watchdog finds train.py without daemon
**Recovery:** Watchdog kills orphans before restart
**Prevention:** Always use control system, never kill -9 daemon

### Scenario 7: Config Corruption
**Detection:** Health check fails to parse config.json
**Recovery:** Restore from backup or rebuild config
**Prevention:** Validate config before writing changes

### Scenario 8: Missing Base Model
**Detection:** Health check verifies base_model path
**Recovery:** Update config to correct path
**Prevention:** Verify paths before consolidation

---

## 📋 TESTING CHECKLIST

Before deploying to production, test these scenarios:

- [ ] Daemon crashes and auto-restarts (watchdog)
- [ ] Training hangs and gets restarted (watchdog)
- [ ] Orphaned processes are cleaned up
- [ ] Crashes are detected and categorized
- [ ] Recovery suggestions are helpful
- [ ] Health checks detect issues
- [ ] Auto-fix mode works for simple issues
- [ ] Checkpoints are valid after crash
- [ ] Training resumes from correct step
- [ ] Disk space warnings trigger
- [ ] Memory warnings trigger
- [ ] GPU errors are detected
- [ ] Config corruption is detected
- [ ] Queue system handles crashes gracefully

---

## 🎯 SUCCESS METRICS

### Before (Old System)
- ❌ Crashes went unnoticed for hours/days
- ❌ No auto-restart (manual intervention required)
- ❌ Orphaned processes ran unstably
- ❌ Lost progress from crashes
- ❌ No crash type identification
- ❌ No automated health checks

### After (New System)
- ✅ Crashes detected within 30 seconds
- ✅ Auto-restart within 1 minute
- ✅ Orphaned processes cleaned up automatically
- ✅ Minimal progress loss (only to last checkpoint)
- ✅ Crash types identified with recovery suggestions
- ✅ Continuous health monitoring

---

## 🚀 FUTURE IMPROVEMENTS

### High Priority
1. **Automatic checkpoint cleanup** - Delete old checkpoints, keep only last 3-5
2. **Alert system** - Send notifications on critical failures
3. **Metrics dashboard** - Visualize system health over time
4. **Save total limit fix** - Prevent checkpoint bloat at source

### Medium Priority
5. **Smart crash recovery** - Adjust config based on crash type (e.g., reduce batch size on OOM)
6. **Checkpoint verification** - Validate checkpoints before resuming
7. **Resource usage tracking** - Historical data on memory/GPU/disk
8. **Auto-scaling** - Adjust batch size based on available resources

### Low Priority
9. **Distributed training support** - Handle multi-GPU failures
10. **Cloud backup** - Automatic checkpoint backup to cloud storage
11. **Web UI for monitoring** - Real-time dashboard
12. **Anomaly detection** - ML-based unusual pattern detection

---

## 📞 TROUBLESHOOTING

### Watchdog isn't restarting daemon
- Check if watchdog is running: `ps aux | grep daemon_watchdog`
- Check watchdog logs: `tail logs/watchdog.log`
- Verify PID file: `cat .daemon.pid` matches actual process
- Check restart limits (max 3 in 5 min)

### Crash detector finds no crashes
- Check if logs exist: `ls logs/`
- Try with more lines: `--last-n-lines 5000`
- Check for permission issues
- Verify log rotation isn't hiding errors

### Health check shows false positives
- Review check thresholds in code
- Verify paths are correct
- Check if services started but slow to respond

### Edge case tests fail
- Read test output carefully
- Some tests may be expected to "pass detection" not "pass operation"
- Use dry-run mode to see what would be tested

---

## 📚 RELATED DOCUMENTATION

- `SYSTEM_HEALTH_REPORT_2025-11-17.md` - Original crash analysis
- `CLAUDE.md` - Updated with new tools
- Watchdog logs: `logs/watchdog.log`
- Crash history: `.crash_history.json`
- Health check results: stdout from comprehensive_health_check.py

---

## ✅ SUMMARY

The crash prevention system consists of **4 complementary tools**:

1. **Daemon Watchdog** - Auto-restart on crash (proactive)
2. **Crash Detector** - Identify crash types (reactive)
3. **Health Check** - Prevent issues before they occur (preventive)
4. **Edge Case Tests** - Validate system behavior (verification)

Together, these tools provide **comprehensive protection** against the crash patterns that have plagued this system.

**Key takeaway:** Run the watchdog in production, health checks daily, crash detector after incidents, and edge case tests weekly.
