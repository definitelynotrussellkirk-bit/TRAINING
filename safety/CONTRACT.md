# safety/ - Manual Diagnostic & Recovery Tools

**Purpose:** Standalone tools for diagnosing, validating, and recovering from training system failures.

**Status:** Not actively running. These are on-demand diagnostic tools for manual troubleshooting.

**Architecture Role:** Safety net layer - manually invoked when issues occur.

---

## 🔧 Tools Provided

### 1. daemon_watchdog.py

**What it does:**
- Monitors training_daemon.py health continuously
- Auto-restarts daemon on crash or hang
- Tracks restart attempts (max 3 in 5 min window)
- Logs all events to logs/watchdog.log

**What it expects:**
- `BASE_DIR`: /path/to/training
- `.daemon.pid`: PID file for daemon
- `status/training_status.json`: Progress tracking
- `training_output.log`: Daemon output log

**Thresholds:**
- Check interval: 30 seconds
- Progress timeout: 10 minutes (no progress = hung)
- Max restarts: 3 per 5 minutes

**Usage:**
```bash
python3 safety/daemon_watchdog.py
```

**Current status:** Not running (monitoring/3090_watchdog.py may have replaced this)

---

### 2. anti_stuck_monitor.py

**What it does:**
- Detects training hangs (stuck eval steps, CUDA hangs, I/O hangs)
- Monitors step progress every 60 seconds
- Kills and restarts training if stuck > 15 minutes
- Special handling for eval timeouts (30 min limit)

**What it expects:**
- `status/training_status.json`: current_step field
- `.daemon.pid`: Process to kill if stuck
- Writes to: `logs/anti_stuck.log`

**Thresholds:**
- Check interval: 60 seconds
- Stuck timeout: 15 minutes (no step progress)
- Eval timeout: 30 minutes (stuck at eval step)
- Max kill attempts: 3

**Usage:**
```bash
python3 safety/anti_stuck_monitor.py [--timeout 900]
```

**Current status:** Not running

---

### 3. crash_detector.py

**What it does:**
- Scans logs for crash patterns (OOM, CUDA errors, timeouts, etc.)
- Categorizes crash type
- Suggests recovery actions
- Tracks crash history in .crash_history.json

**What it expects:**
- `logs/`: Training log files
- Patterns detected: CUDA OOM, CUDA errors, SIGKILL, timeouts, import errors

**Crash categories:**
- `cuda_oom`: Out of memory errors
- `cuda_error`: General CUDA failures
- `process_killed`: SIGKILL (system OOM killer)
- `timeout`: Timeout errors
- `import_error`: Missing dependencies

**Usage:**
```bash
python3 safety/crash_detector.py [--last-n-lines 1000]
```

**Output:** Crash analysis + suggested fixes

**Current status:** Manual diagnostic tool

---

### 4. comprehensive_health_check.py

**What it does:**
- Runs 10+ health checks on system
- Tests: daemon health, training progress, disk space, GPU, config, permissions, memory, checkpoints
- Optional auto-fix mode
- Reports issues/warnings/passed checks

**What it expects:**
- `config.json`: Config file
- `status/training_status.json`: Status file
- `.daemon.pid`: Daemon PID
- Directories: inbox, logs, status, queue, current_model
- Files: config.json, train.py, training_daemon.py

**Thresholds:**
- Min disk space: 50GB
- Min memory: 4GB
- Max checkpoint age: 24 hours

**Usage:**
```bash
# Check only
python3 safety/comprehensive_health_check.py

# Check and auto-fix
python3 safety/comprehensive_health_check.py --fix
```

**Current status:** Available for manual checks

---

### 5. config_validator.py

**What it does:**
- Validates config.json for dangerous changes
- Prevents base_model changes when checkpoints exist
- Checks for invalid values and missing paths
- Creates .config_lock.json to track training start config

**What it expects:**
- `config.json`: Main config
- `current_model/`: Checkpoint directory
- `.config_lock.json`: Locked config (created when training starts)

**Prevents:**
- Changing base_model during training (would train on wrong base)
- Invalid config values
- Path references to non-existent files

**Usage:**
```bash
# Validate and suggest fixes
python3 safety/config_validator.py

# Validate only (no fixes)
python3 safety/config_validator.py --validate-only
```

**Current status:** Manual validation tool

---

### 6. verify_checkpoint_resume.py

**What it does:**
- Verifies training resumes from correct checkpoint
- Checks checkpoint integrity
- Prevents starting from scratch when checkpoints exist

**What it expects:**
- `config.json`: Config file
- `status/training_status.json`: Training status
- `current_model/`: Checkpoint directory
- `current_model/checkpoint-*/`: Individual checkpoints

**Checks:**
- Latest checkpoint exists
- Checkpoint files present (model.safetensors, optimizer.pt, scheduler.pt, trainer_state.json)
- Training will resume from correct step

**⚠️  NOTE:** Contains legacy LoRA references (adapter_model.safetensors) - needs update for full model training.

**Usage:**
```bash
python3 safety/verify_checkpoint_resume.py
```

**Current status:** May need updating for full model architecture

---

## 📋 File Dependencies

All tools expect this directory structure:

```
/path/to/training/
├── config.json                  # Main config
├── .daemon.pid                  # Daemon process ID
├── .config_lock.json            # Config at training start
├── .crash_history.json          # Crash history
├── core/
│   ├── train.py
│   └── training_daemon.py
├── status/
│   └── training_status.json     # Current training state
├── logs/
│   ├── watchdog.log
│   ├── anti_stuck.log
│   └── *.log                    # Training logs
├── current_model/               # Checkpoint directory
│   └── checkpoint-*/
└── queue/, inbox/, etc.
```

---

## 🔄 Relationship to monitoring/

**Key difference:**
- **safety/**: Manual diagnostic tools (run on-demand)
- **monitoring/**: Autonomous 24/7 daemons (auto-run in background)

**Possible overlap:**
- `safety/daemon_watchdog.py` vs `monitoring/3090_watchdog.py`
- `safety/anti_stuck_monitor.py` vs monitoring regression detection
- `safety/comprehensive_health_check.py` vs `monitoring/3090_health_dashboard.py`

**Current state:** monitoring/ daemons handle 24/7 monitoring, safety/ tools for manual intervention.

---

## 🚀 When to Use

**Use daemon_watchdog.py when:**
- training_daemon.py keeps crashing
- Need auto-restart functionality
- Monitoring daemon health manually

**Use anti_stuck_monitor.py when:**
- Training hangs at eval steps
- Suspect CUDA/I/O deadlock
- Need aggressive hang detection

**Use crash_detector.py when:**
- Training crashed and you don't know why
- Need to analyze log patterns
- Tracking crash history

**Use comprehensive_health_check.py when:**
- System acting weird
- Before starting new training
- After recovering from crash
- Diagnosing multi-component issues

**Use config_validator.py when:**
- About to change config.json
- Verifying config safety
- Checking for dangerous changes

**Use verify_checkpoint_resume.py when:**
- Training restarted from scratch (shouldn't happen)
- Verifying checkpoint loading
- Debugging resume issues
- **⚠️  Currently may report false positives due to LoRA references**

---

## ⚠️  Known Issues

1. **verify_checkpoint_resume.py**: Contains legacy LoRA code (checks for adapter_model.safetensors), but system now does full model training (model.safetensors). Needs update.

2. **Not integrated with scripts/start_all.sh**: None of these tools auto-start. All manual.

3. **Overlap with monitoring/**: Some functionality duplicated by autonomous monitoring daemons.

---

## 📝 Maintenance Notes

**Last verified:** 2025-11-24
**Training architecture:** Full model fine-tuning (no LoRA)
**Daemon location:** core/training_daemon.py
**Config location:** /path/to/training/config.json

**Action items:**
- [ ] Update verify_checkpoint_resume.py for full model architecture
- [ ] Determine if daemon_watchdog.py is replaced by monitoring/3090_watchdog.py
- [ ] Consider integrating health checks into start_all.sh
- [ ] Archive unused tools if monitoring/ daemons provide full coverage
