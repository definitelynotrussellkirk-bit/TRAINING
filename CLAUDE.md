# CLAUDE QUICK REFERENCE - Ultimate Trainer System

**Last Updated:** 2025-11-18 (Comprehensive Safety System Added)

This document contains key information for Claude to help with training operations.

## 🆕 LATEST: Comprehensive Safety System (2025-11-18)

**After crash at step 2500, built complete auto-recovery and prevention system:**

- **11 new safety tools** - Auto-restart, hang detection, config locking, etc.
- **All edge cases tested** - Won't stuck, won't train wrong version, handles data changes
- **Production-ready** - Multiple overlapping safeguards

**Quick access:**
- Full details: `SAFEGUARDS_SUMMARY.md`
- Complete guide: `CRASH_PREVENTION_GUIDE.md`
- See "Safety Tools" section below for usage

---

## 🚨 CRITICAL RULES (READ FIRST!)

### Deletion Policies (NEVER VIOLATE!)

1. **NEVER delete `current_model/` without explicit user permission**
2. **ALWAYS create backup before risky operations**
3. **NEVER modify config.json critical parameters (`max_length`, `model_name`, `base_model`) without user approval**

### Pre-Session Checklist

```bash
# 1. System state check
python3 state_tracker.py --check

# 2. Health check (NEW - recommended!)
python3 comprehensive_health_check.py

# 3. Verify checkpoint resume (NEW - before training)
python3 verify_checkpoint_resume.py

# 4. Validate config (NEW - before config changes)
python3 config_validator.py
```

**State tracker shows:** model status, saved versions, config, training state, daemon status, disk space, warnings.

**Health check shows:** All system components, resource availability, potential issues.

**Checkpoint verifier shows:** Resume will work correctly, no version mismatches.

**Config validator shows:** Config is safe, locked parameters, path validity.

---

## 🎯 CURRENT CONFIGURATION

**Model:** Qwen3 8B (DIO model)
**Location:** `/path/to/training/DIO_20251114/` *(base model)*
**Method:** QLoRA (4-bit quantization)
**LoRA:** r=128, alpha=128, dropout=0.02

**Key Settings (`config.json`):**
- Batch size: 1 (effective: 8 with gradient accumulation)
- Learning rate: 2e-4
- Eval steps: 500
- Save steps: 100 (checkpoint frequency)
- Max length: 3072 tokens
- Poll interval: 30 seconds

---

## ⚡ QUICK START

### 1. Prepare Data
```bash
cd /path/to/training

# Copy .jsonl file to inbox/
cp /path/to/your/data.jsonl inbox/

# Daemon auto-processes files every 30 seconds
```

### 2. Start Services
```bash
# Start everything
./start_all.sh

# OR manually:
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &  # Port 8080
nohup python3 memory_stats_api.py > /dev/null 2>&1 &      # Port 8081
```

### 3. Monitor Training
**Web UI:** http://localhost:8080/live_monitor_ui_v2.html

**Command line:**
```bash
# Real-time status
cat status/training_status.json | jq '{step, total, loss, val_loss, gap}'

# Watch logs
tail -f logs/daemon_$(date +%Y%m%d).log

# GPU usage
nvidia-smi
```

---

## 🎛️ CONTROL SYSTEM

### Graceful Control (Preferred)
```bash
python3 training_controller.py pause   # Pause within ~20-30s
python3 training_controller.py resume  # Resume training
python3 training_controller.py stop    # Stop after current file
python3 training_controller.py skip    # Skip current file
python3 training_controller.py status  # Check status
```

### Queue Management
```bash
python3 training_queue.py add mydata.jsonl --priority high
python3 training_queue.py status
python3 training_queue.py list
python3 training_queue.py set-priority mydata.jsonl high
```

### Emergency Stop
```bash
touch .stop  # Daemon checks every loop
```

---

## 🔀 MODEL MANAGEMENT

### Consolidation (Merge adapter into base)
```bash
# Must provide description for version tracking
python3 consolidate_model.py \
  --base-dir /path/to/training \
  --description "What was trained (e.g., Math 10k examples)"
```

**What happens:**
1. Verified backup created
2. Numbered version saved (v001, v002, etc.)
3. Adapter merged into base model
4. Safe cleanup with triple redundancy

### Version Management
```bash
python3 model_versioner.py list              # List all versions
python3 model_versioner.py restore v002      # Restore specific version
python3 model_versioner.py rollback          # Quick rollback to previous
python3 model_versioner.py delete v003 --confirm
```

### Backup Management
```bash
python3 backup_manager.py list
python3 backup_manager.py backup current_model/ --type emergency --reason "Before risky change"
python3 backup_manager.py cleanup --retention-days 30 --execute
```

---

## 📊 VALIDATION & MONITORING

### Check Metrics
```bash
cat status/training_status.json | jq '{
  step: .current_step,
  train_loss: .loss,
  val_loss: .validation_loss,
  gap: .val_train_gap,
  think_pct: .think_tag_percent
}'
```

**Interpret gap (train_loss - val_loss):**
- **< 0.3:** ✅ Excellent - generalizing well
- **0.3 - 0.5:** ⚠️ Warning - monitor closely
- **> 0.5:** 🚨 Overfitting - consider stopping

**Think tag percentage:**
- **100%:** Base model behavior (adding `<think>` tags)
- **< 20%:** ✅ Good - learning clean format
- **0%:** 🎯 Perfect - no unwanted tags

### Monitor URLs
- **Primary:** http://localhost:8080/live_monitor_ui_v2.html (New modular UI)
- **Legacy:** http://localhost:8080/live_monitor_ui.html
- **Test page:** http://localhost:8080/test_display.html

---

## 🔧 TROUBLESHOOTING

### Daemon Not Running
```bash
ps aux | grep training_daemon | grep -v grep
# If not running, restart:
nohup python3 training_daemon.py --base-dir /path/to/training > /dev/null 2>&1 &
```

### Monitor Not Working
```bash
./check_health.sh  # Diagnose all systems

# Manually check:
curl -s http://localhost:8080/status/training_status.json | jq .current_step
curl -s http://localhost:8081/api/memory_stats | jq .status
```

### Training Errors
```bash
# Check status
cat status/training_status.json | jq .error

# Check logs
tail -100 logs/daemon_$(date +%Y%m%d).log

# DON'T delete anything - ASK USER FIRST
```

### Disk Space Issues
```bash
# Check usage
du -sh current_model/ queue/ logs/

# Old checkpoints piling up:
# WARNING: Don't manually delete - could break training!
# Wait for consolidation or ask user
```

---

## 🛡️ DATA VALIDATION

**The daemon automatically validates data before training!**

It checks:
- Token lengths against `max_length` config
- Warns if >95% of examples exceed limit
- Blocks training if truncation is severe

**Manual validation:**
```bash
python3 validate_data.py --file my_data.jsonl
python3 validate_data.py --auto-adjust  # Auto-adjust config if needed
```

---

## 📁 DIRECTORY STRUCTURE

```
/path/to/training/
├── inbox/                    # Drop .jsonl files here
├── current_model/            # Active training (LoRA adapter)
├── models/
│   ├── versions/             # Versioned snapshots (v001, v002, ...)
│   └── backups/              # Safety backups
├── consolidated_models/      # Merged base models
├── data/validation/          # Fixed validation set
├── logs/                     # Training logs (daily rotation)
├── status/                   # Real-time status JSON
├── queue/                    # Training queue (high/normal/low priority)
├── DIO_20251114/             # Base model (Qwen3 8B)
├── config.json               # Configuration
├── train.py                  # Core training script
├── training_daemon.py        # Auto-ingestion daemon
├── training_controller.py    # Control system
├── training_queue.py         # Queue management
├── model_versioner.py        # Version management
├── backup_manager.py         # Backup system
└── state_tracker.py          # System state tracking
```

---

## 🔥 KNOWN ISSUES & FIXES

### CUDA Multiprocessing Crashes (FIXED)
**Status:** ✅ FIXED - `num_proc=None` in train.py:414 disables multiprocessing

### Monitoring Costs (LESSON LEARNED)
- Evolution tracking: DISABLED (was 10-15% overhead)
- Detail collector: DISABLED (was 0.2-1% overhead)
- Live inference: Every 200 steps only (~0.5% overhead)
- **Rule:** Monitor `monitoring_costs.json` - keep total <5% overhead

### Continuous Training System
**How it works:**
- HuggingFace Trainer manages checkpoints automatically
- Each checkpoint contains: weights, optimizer state, LR scheduler, global_step
- `save_steps=100` ensures checkpoints between batches
- `global_step` never resets - accumulates across files
- Seamless resumption from latest checkpoint

---

## 📝 NOTES FOR CLAUDE

### Critical Context
- **System stability is paramount** - we've lost 3+ weeks of training to accidental deletions
- **Always verify before deleting** - check version history first
- **Config drift happens** - always verify config matches docs
- **Disk space matters** - 31+ checkpoints = 120GB+ bloat

### Current System State
- Eval frequency: 500 steps
- Checkpoint frequency: 100 steps
- Validation system: Enabled (1000 examples fixed set)
- Queue system: Fully operational
- Daemon: Should be running (check with state_tracker.py)

### When in Doubt
1. Run `python3 state_tracker.py --check`
2. Read the warnings it shows
3. **ASK THE USER** before making changes

---

## 🛡️ SAFETY TOOLS (NEW - 2025-11-18)

### Auto-Recovery Tools

**Daemon Watchdog** (`daemon_watchdog.py`)
- Monitors daemon health every 30 seconds
- Auto-restarts on crash within 60 seconds
- Kills orphaned processes
```bash
nohup python3 daemon_watchdog.py > logs/watchdog.log 2>&1 &
```

**Anti-Stuck Monitor** (`anti_stuck_monitor.py`)
- Detects training hangs (15 min timeout)
- Detects eval step hangs (30 min timeout)
- Auto-kills stuck processes
```bash
nohup python3 anti_stuck_monitor.py > logs/anti_stuck.log 2>&1 &
```

### Safety Validation Tools

**Config Validator** (`config_validator.py`)
- **Locks config during training** (prevents wrong version training)
- Validates paths exist
- Prevents dangerous changes
```bash
python3 config_validator.py  # Run before config changes
```

**Checkpoint Verifier** (`verify_checkpoint_resume.py`)
- Confirms training will resume from correct checkpoint
- Validates checkpoint integrity
- Checks for version mismatches
```bash
python3 verify_checkpoint_resume.py  # Run before starting training
```

**Safe Checkpoint Cleanup** (`safe_checkpoint_cleanup.py`)
- Cleans old checkpoints safely
- **Blocks during active training**
- Keeps latest N checkpoints
```bash
python3 safe_checkpoint_cleanup.py --keep 5  # After training finishes
```

### Detection & Analysis Tools

**Crash Detector** (`crash_detector.py`)
- Analyzes logs for crash patterns
- Categorizes crash types (OOM, CUDA, etc.)
- Provides recovery suggestions
```bash
python3 crash_detector.py  # After any crash
```

**Health Check** (`comprehensive_health_check.py`)
- Tests all system components
- 10+ health checks
- Auto-fix mode available
```bash
python3 comprehensive_health_check.py [--fix]  # Daily
```

**Edge Case Tests** (`test_edge_cases.py`)
- Tests failure scenarios
- Validates recovery mechanisms
```bash
python3 test_edge_cases.py  # Weekly
```

### For Complete Details
- **`SAFEGUARDS_SUMMARY.md`** - Answers all edge case questions
- **`CRASH_PREVENTION_GUIDE.md`** - Complete documentation (400+ lines)

---

## 📚 DOCUMENTATION

**See also:**
- `SAFEGUARDS_SUMMARY.md` - **NEW!** Safety system overview
- `CRASH_PREVENTION_GUIDE.md` - **NEW!** Complete crash prevention guide
- `SYSTEM_HEALTH_REPORT_2025-11-17.md` - Crash analysis
- `README.md` - Comprehensive system overview
- `MONITORING_COSTS_GUIDE.md` - Monitoring overhead management
- `VALIDATION_SYSTEM_DOCS.md` - Validation system details
- `DOCS_INDEX.md` - Full documentation index
