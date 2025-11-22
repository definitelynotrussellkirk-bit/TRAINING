# CLAUDE INSTRUCTIONS - LLM Training System

**Last Updated:** 2025-11-22 (Production Integration Complete)
**Update Frequency:** Every ~50k tokens or when significant changes occur

This document contains instructions for Claude to help with training operations.

**MAJOR UPDATE:** Refactored trainer/ modules now integrated into production (core/train.py)
- ✅ Steps 1-5: Created trainer/ architecture (config, profiles, monitoring)
- ✅ **NEW:** Production integration - core/train.py uses ConfigLoader + profiles

---

## 📋 COMMUNICATION STYLE

**Default mode: Factual technical documentation**

- State facts about system behavior, configuration, and current state
- Do not include recommendations, suggestions, or opinions unless explicitly asked
- Do not add phrases like "I recommend", "you should", "it's best to", "consider"
- Present options without bias when multiple approaches exist
- Omit evaluative language ("excellent", "better", "perfect", "brilliant")
- When asked "how does X work", describe the mechanism without suggesting changes
- When asked "what are the options", list them without ranking

**Example:**
- ❌ "I recommend using batch_size=30 because it's more efficient"
- ✅ "batch_size=30 uses ~21GB VRAM. batch_size=16 uses ~14GB VRAM"

**Only add recommendations when:**
- Explicitly asked ("what should I do?", "which is better?")
- Critical safety issue (data loss, system damage)
- User makes factual error that needs correction

---

## 🚨 CRITICAL RULES

### Documentation Policies

1. **7 Canonical Docs** - Only write to these 7 files:
   - `README.md` - System overview
   - `QUICKSTART.md` - Getting started guide
   - `ARCHITECTURE.md` - How the system works
   - `TROUBLESHOOTING.md` - Common problems and solutions
   - `REMOTE_INFERENCE.md` - **⭐ Remote RTX 3090 inference server (primary reference)**
   - `DEVELOPMENT.md` - Working on the codebase
   - `CHANGELOG.md` - Track changes

2. **CLAUDE.md Updates** - Update this file every ~50k tokens or when significant changes occur

3. **No Other Docs** - Do NOT create any other .md files without explicit user permission

4. **Remote Inference Focus** - See `REMOTE_INFERENCE.md` for all inference/generation tasks. This training machine does NOT run inference.

### Safety Policies

1. **NEVER delete `models/current_model/` without explicit user permission**
2. **ALWAYS create backup before risky operations**
3. **NEVER modify `config.json` critical parameters without user approval:**
   - `max_length`
   - `model_name`
   - `base_model`
4. **ASK FIRST** before making system-wide changes

---

## 📁 DIRECTORY STRUCTURE

**Reorganized 2025-11-22** - All files organized + new trainer/ module added

```
/path/to/training/
│
├── CLAUDE.md                    # This file (Claude instructions)
├── config.json                  # Active configuration
│
├── trainer/                     # 🆕 NEW: Refactored training modules (3-layer architecture)
│   ├── config/                  # Layer 2: Configuration system
│   │   ├── schema.py            # 8 dataclasses (Hyperparams, ProfileConfig, etc.)
│   │   └── loader.py            # ConfigLoader (JSON + CLI merging)
│   ├── profiles/                # Layer 3: Pluggable data profiles
│   │   ├── base.py              # DataProfile ABC interface
│   │   ├── emoji_think.py       # Emoji thinking/stop profile
│   │   └── regime3.py           # Symbolic reasoning profile (NEW!)
│   ├── monitoring/              # Layer 3: Monitoring callbacks
│   │   ├── status_writer.py     # TrainingStatusWriter
│   │   └── callbacks.py         # LiveMonitorCallback
│   ├── core/                    # Layer 1: Engine API
│   │   └── engine.py            # TrainerEngine.run_job() (proof-of-concept)
│   └── cli_main.py              # CLI demonstration
│
├── README.md                    # System overview
├── QUICKSTART.md                # Getting started
├── ARCHITECTURE.md              # System design
├── TROUBLESHOOTING.md           # Problem solving
├── DEVELOPMENT.md               # Development guide
├── CHANGELOG.md                 # Change tracking
│
├── core/                        # Core training system (11 files)
│   ├── train.py                 # Main training script (HuggingFace Trainer) - STILL WORKS
│   ├── train_v1_backup.py       # Backup before refactor
│   ├── training_daemon.py       # File watcher + orchestrator
│   ├── training_controller.py   # Control commands (pause/resume/stop)
│   ├── training_queue.py        # Queue management
│   ├── training_status.py       # Status writer (copied to trainer/monitoring/)
│   ├── custom_collator.py       # Data collator
│   ├── logit_penalty.py         # Penalty processors
│   ├── validator.py             # Data validation
│   ├── model_db.py              # Model database
│   └── time_estimator.py        # Time estimation
│
├── monitoring/                  # Monitoring + Web UI
│   ├── servers/                 # API servers
│   │   ├── live_monitor.py      # Main monitor server
│   │   ├── memory_stats_api.py  # Memory stats API
│   │   └── enhanced_monitor.py  # Enhanced monitoring
│   ├── ui/                      # HTML files
│   │   └── *.html
│   ├── js/                      # JavaScript modules
│   └── css/                     # Stylesheets
│
├── management/                  # Model/backup management
│   ├── backup_manager.py        # Backup system
│   ├── model_versioner.py       # Version control
│   ├── consolidate_model.py     # Model consolidation
│   ├── checkpoint_retention.py  # Checkpoint cleanup
│   ├── safe_checkpoint_cleanup.py
│   ├── daily_snapshot.py        # Daily snapshots
│   └── auto_disk_manager.py     # Auto disk cleanup
│
├── safety/                      # Watchdogs + health checks
│   ├── daemon_watchdog.py       # Auto-restart daemon
│   ├── anti_stuck_monitor.py    # Detect hangs
│   ├── crash_detector.py        # Crash analysis
│   ├── comprehensive_health_check.py
│   ├── config_validator.py      # Config validation
│   └── verify_checkpoint_resume.py
│
├── tools/                       # Utilities
│   ├── data/                    # Data processing
│   │   ├── generate_syllo_batch.py
│   │   ├── validate_data.py
│   │   ├── convert_*.py
│   │   └── analyze_training_data.py
│   ├── config/
│   │   └── edit_config.py
│   └── analysis/
│       ├── state_tracker.py     # System state tracking
│       ├── calculate_data_value.py
│       └── compare_models.py
│
├── tests/                       # Test files
│   └── test_*.py
│
├── scripts/                     # Shell scripts
│   ├── start_all.sh             # Start all services
│   ├── check_health.sh          # Health check
│   └── bin/                     # Launcher scripts
│
├── data/                        # Training data
│   ├── validation/              # Fixed validation set
│   └── flagged_examples/        # Flagged outputs
│
├── models/                      # Model storage
│   ├── Qwen3-0.6B/              # Base model (1.5GB)
│   ├── current_model/           # Active checkpoint (EMPTY - needs setup)
│   └── current_model_small/     # Small model checkpoint
│
├── backups/                     # Backups
│   └── consolidated/            # Consolidated backups
│
├── logs/                        # Training logs (daily rotation)
├── status/                      # Status JSON files
├── control/                     # Control files (.stop, .pause, etc.)
├── inbox/                       # Drop zone for training files
├── queue/                       # Priority queues
│   ├── high/
│   ├── normal/
│   ├── low/
│   ├── processing/              # Currently training
│   ├── failed/                  # Failed files
│   └── recently_completed/
│
├── scratch/                     # Documentation & tests
│   ├── REFACTOR_PLAN.md         # Original refactor plan
│   ├── REFACTOR_COMPLETE.md     # Refactor completion summary
│   ├── STEP1_COMPLETE.md        # Step 1 validation
│   ├── STEP2_COMPLETE.md        # Step 2 validation
│   ├── REFACTOR_PROGRESS.md     # Progress tracking
│   ├── test_emoji_profile.py    # Emoji profile tests (6 tests)
│   └── test_regime3_profile.py  # Regime3 profile tests (7 tests)
│
└── archive/                     # Archived/experimental
    ├── configs/                 # Old configs
    ├── experiments/             # Experimental scripts
    └── PERMANENT_ERROR_TRAINING/

# IGNORED (user data/notes):
GOTCHA_BUSINESS_MODEL/
OBSERVATIONS/
```

---

## 🆕 RECENT UPDATES (2025-11-22)

**Production Integration Complete** - trainer/ modules now in core/train.py (commit: 5cdebe4)

**What's New:**
- ConfigLoader integrated - single TrainerConfig from args + config.json
- Profiles active - emoji_think & regime3 available via config
- Precision unified - model load + training use same precision (bf16/fp16/fp32)
- System prompt fixed - uses `--system-prompt` CLI arg
- 100% backward compatible - falls back to legacy if needed

**Quick Start:**
```bash
# Edit config.json to switch profiles
{"profile": {"name": "regime3"}}  # or "emoji_think"

# Run training (automatically uses profile + config)
python3 core/train.py --dataset data.jsonl --model qwen3 --output outputs
```

**Refactor Timeline:**
- Steps 1-5: Created trainer/ architecture (6 git tags)
- Step 6: Production integration (this update)
- Result: ~3,400 lines → 14 modules, fully tested, in production

**Key Files:**
- `trainer/config/` - ConfigLoader, TrainerConfig schema
- `trainer/profiles/` - emoji_think, regime3 data profiles
- `core/train.py` - Production script (now uses trainer/ modules)

See CHANGELOG.md for details

---

## 🎯 CURRENT SYSTEM STATE

**Last Verified:** 2025-11-22 04:40 AM

### Model Status
- **Base model:** Qwen3-0.6B (exists at `/path/to/training/models/Qwen3-0.6B/`)
  - Size: 1.5GB
  - Type: Qwen3ForCausalLM
  - Hidden size: 1024, Layers: 28, Vocab: 151936
- **Current model:** EMPTY (needs initialization)
- **Training method:** Full model fine-tuning (no LoRA)
- **Last training attempt:** Failed with OOM at 04:02 AM

### Queue Status
- **Normal queue:** 0 files
- **Stuck files:** 3 files (287MB each, 100k examples)
  - 2 in `queue/failed/`
  - 1 in `queue/processing/`
- **Issue:** OOM crashes during training

### Service Status
- ✅ Disk manager: Running
- ❌ Training daemon: NOT running (PID dead)
- ❌ Monitors: NOT running
- ❌ Watchdogs: NOT running

### Configuration (`config.json`)
```json
{
  "model_name": "qwen2.5_0.5b",
  "model_path": "/path/to/training/models/Qwen3-0.6B",
  "batch_size": 19,
  "learning_rate": 0.0002,
  "eval_steps": 50,
  "save_steps": 1000,
  "max_length": 4096,
  "poll_interval": 30
}
```

### Disk Space
- **Available:** 731GB / 1.8TB (58% used)
- **Status:** Healthy

---

## ⚡ QUICK OPERATIONS

### Start All Services
```bash
cd /path/to/training
scripts/start_all.sh
```

This starts:
1. Auto disk manager
2. Training daemon
3. Live monitor (port 8080)
4. Memory stats API (port 8081)
5. Enhanced monitor (port 8082)

### Control Training
```bash
# Check status
python3 core/training_controller.py status

# Pause/resume/stop
python3 core/training_controller.py pause
python3 core/training_controller.py resume
python3 core/training_controller.py stop
```

### Queue Management
```bash
# Add file to queue
python3 core/training_queue.py add mydata.jsonl --priority high

# Check queue status
python3 core/training_queue.py status
python3 core/training_queue.py list
```

### Health Check
```bash
scripts/check_health.sh
python3 safety/comprehensive_health_check.py
python3 tools/analysis/state_tracker.py --check
```

### Monitor URLs
- Main UI: http://localhost:8080/live_monitor_ui_v2.html
- Memory Stats: http://localhost:8081/api/memory_stats
- Status JSON: http://localhost:8080/status/training_status.json

---

## 🔧 COMMON ISSUES

### Training Daemon Not Running
```bash
# Check if running
ps aux | grep training_daemon | grep -v grep

# Restart if needed
cd /path/to/training
nohup python3 core/training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
```

### OOM (Out of Memory) Crashes
**Symptoms:** Training crashes with CUDA OOM error

**Common causes:**
1. Batch size too high for available VRAM
2. Multiple daemon processes running (check with `ps aux`)
3. Eval step inference not clearing cache
4. Model too large for GPU

**Actions:**
```bash
# Check for multiple processes
ps aux | grep "python3.*training_daemon" | grep -v grep

# Kill all if multiple found
pkill -f training_daemon
sleep 3

# Reduce batch size
python3 tools/config/edit_config.py batch_size 16

# Restart daemon
nohup python3 core/training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
```

### Stuck Files in Queue
**Check:**
```bash
ls -lh queue/processing/
ls -lh queue/failed/
```

**Clear:**
```bash
# Move back to normal queue (ask user first!)
mv queue/processing/* queue/normal/
mv queue/failed/* queue/normal/
```

---

## 📊 VALIDATION & METRICS

### Check Training Status
```bash
cat status/training_status.json | jq '{
  step: .current_step,
  train_loss: .loss,
  val_loss: .validation_loss,
  gap: .val_train_gap
}'
```

### Interpret Metrics
- **Train/Val Gap:**
  - < 0.3: Generalizing well
  - 0.3 - 0.5: Monitor closely
  - > 0.5: Possible overfitting

### Validate Data
```bash
python3 tools/data/validate_data.py --file my_data.jsonl
```

---

## 📝 NOTES FOR CLAUDE

### System History
- System reorganized 2025-11-22 (99 Python files → organized into 6 categories)
- All old documentation deleted (16 .md files removed)
- Fresh start - trust code as ground truth, not old docs

### Current Priorities
1. Fix stuck queue files (3 x 287MB files causing OOM)
2. Restart training daemon
3. Monitor system health

### When in Doubt
1. Run health check: `python3 safety/comprehensive_health_check.py`
2. Check system state: `python3 tools/analysis/state_tracker.py --check`
3. **ASK USER** before making changes
4. **ASK USER** before creating new documentation

---

## 🔄 UPDATE LOG

**2025-11-22 (Late - Refactor Complete):**
- ✅ Completed full 5-step refactor (~3 hours)
- ✅ Created trainer/ module with 3-layer architecture
- ✅ Extracted config system (8 dataclasses, type-safe)
- ✅ Extracted emoji_think profile (emoji patterns + stop signals)
- ✅ Created regime3 profile (symbolic reasoning) ⭐ NEW
- ✅ Extracted monitoring system (callbacks + status writer)
- ✅ Created TrainerEngine API (proof-of-concept)
- ✅ 13/13 tests passing, 100% backward compatible
- ✅ All pushed to GitHub with 6 git tags
- **Production ready:** Can use new modules today or continue with core/train.py

**2025-11-22 (Morning - Reorganization):**
- Reorganized entire codebase (99 Python files → 6 categories)
- Created 7 canonical documentation files
- Removed all old documentation (96+ files)
- Updated CLAUDE.md to reflect new structure
- **Architecture decision:** Pure training module - all inference on remote RTX 3090
- Added REMOTE_INFERENCE.md for remote server operations
- System state: Fresh start, daemon not running, 3 files stuck in queue
