# CLAUDE INSTRUCTIONS - LLM Training System

**Last Updated:** 2025-11-25 (Code Review + Cleanup)
**Update Frequency:** Every ~50k tokens or when significant changes occur

This document contains instructions for Claude to help with training operations.

**MAJOR UPDATE:** Code Review Validated Monitoring Systems (2025-11-25)
- ✅ API authentication added to inference server
- ✅ Test infrastructure cleaned up for CI
- ✅ RetentionManager wired into daemon
- ✅ Extracted daemon services: PIDManager, FileWatcher, SnapshotService, BackgroundWorker
- ✅ Extracted training components: ModelLoader, DatasetPreparer, MonitoringBundle
- ✅ Created pyproject.toml - GPU deps now optional `[training]` extra
- ✅ DataValidator (QUICK/STANDARD/DEEP) - integrated into daemon for early rejection
- ✅ Path auto-detection via get_base_dir() with resolution logging

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
├── pyproject.toml               # 🆕 Package config (pip install -e .)
│
├── core/                        # Core training system
│   ├── train.py                 # Main training script (HuggingFace Trainer)
│   ├── training_daemon.py       # File watcher + orchestrator
│   ├── training_controller.py   # Control commands (pause/resume/stop)
│   ├── training_queue.py        # Queue management
│   ├── training_status.py       # Status writer
│   ├── paths.py                 # 🆕 Path auto-detection (get_base_dir)
│   ├── daemon/                  # 🆕 Extracted daemon services
│   │   ├── pid_manager.py       # Single-instance enforcement
│   │   ├── file_watcher.py      # Directory monitoring + inbox flattening
│   │   ├── snapshot_service.py  # Checkpoint snapshots
│   │   └── background_worker.py # Non-blocking task runner
│   ├── training/                # 🆕 Extracted training components
│   │   ├── model_loader.py      # Model loading with precision config
│   │   ├── dataset_preparer.py  # Dataset preparation
│   │   └── monitoring_bundle.py # Training monitoring
│   ├── validation/              # 🆕 Two-layer validation system
│   │   ├── spec.py              # SpecValidator + DatasetSpec registry (deny-by-default)
│   │   └── validator.py         # DataValidator (QUICK/STANDARD/DEEP content checks)
│   ├── custom_collator.py       # Data collator
│   ├── logit_penalty.py         # Penalty processors
│   ├── validator.py             # Legacy validator (deprecated)
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
├── scratch/                     # Working space for design docs & experiments
│   ├── DAEMON_REFACTOR_PLAN.md  # Current work: daemon refactor planning
│   ├── IMPLEMENTATION_PLAN_TASKS.md  # Task breakdowns
│   ├── MONITORING_V2_DESIGN.md  # Monitoring system designs
│   ├── RETENTION_POLICY_DESIGN.md   # Policy documents
│   └── *.md                     # Other design/planning docs
│
└── archive/                     # Archived code & completed work
    ├── refactor_2025_11_22/     # Nov 22 trainer/ refactor
    │   ├── code/                # Backup train.py versions
    │   ├── docs/                # Refactor documentation
    │   └── tests/               # Profile & engine tests
    ├── configs/                 # Old config files
    ├── experiments/             # Old experiments
    └── PERMANENT_ERROR_TRAINING/

# IGNORED (user data/notes):
GOTCHA_BUSINESS_MODEL/
OBSERVATIONS/
```

---

## 🆕 RECENT UPDATES (2025-11-25)

**Chat Template Override** - Fix for Qwen3 `<think>` injection

**What Changed (2025-11-25 - Chat Template Fix):**
- Created `core/chat_templates.py` - Overrides Qwen3's auto-`<think>` injection
- Qwen3 models inject `<think></think>` around all assistant content by default
- This conflicted with our emoji_think paradigm (💭...🔚)
- New module detects Qwen3 template and replaces with clean ChatML
- Training now uses only emoji thinking without competing systems
- See ARCHITECTURE.md "Thinking Tokens / Chat Templates" section

---

**GPU Task Scheduler** - Central coordinator for 3090 GPU workloads

**What Changed (2025-11-25 - GPU Scheduler):**
- Created `monitoring/gpu_task_scheduler.py` - central daemon coordinating all GPU tasks
- Created `monitoring/task_client.py` - client library for task submission
- Updated daemons with `--use-scheduler` flag: curriculum_eval, self_correction, automated_testing
- Scheduler monitors GPU utilization and dispatches tasks from priority queue
- Target: maintain 20-80% GPU utilization, auto-dispatch idle tasks when under 20%
- 11 task types: curriculum_eval, self_correction, automated_test, adversarial_mine, etc.
- API on port 8766: `/api/health`, `/api/tasks/submit`, `/api/metrics`

**Start GPU Scheduler (on 3090):**
```bash
nohup python3 /path/to/training/monitoring/gpu_task_scheduler.py --port 8766 > logs/gpu_scheduler.log 2>&1 &
```

**Previous (2025-11-25 - Curriculum Integration):**
- DataManager rewritten to use `SkillAPIClient` instead of remote GPU (3090)
- Now connects to local singleSKILL APIs: SYLLO (8080), Binary (8090)
- Integrated `CurriculumManager` for adaptive difficulty progression
- **SYLLO-only** for now - master SYLLO before introducing Binary
- Files named: `train_SKILL_levelN_COUNT_TIMESTAMP.jsonl`
- Config updated: `auto_generate.count: 100` (was 100000)
- State file: `data_manager/curriculum_state.json` (tracks levels, history)

**Curriculum System:**
- SYLLO: 5 levels (4-8 word puzzles)
- Binary: 7 levels (magnitude 1-10 to 10K-100K)
- Progression: 80% accuracy over 3 evaluations to advance

**Requirement:** SYLLO API must be running:
```bash
cd /path/to/skills && python3 skill_syllo_variant/api_server.py --port 8080
```

**Previous (2025-11-25 - Code Review):**
- Verified adversarial miner supports `messages[]` format (lines 193-229 `extract_prompt_and_expected()`)
- Verified adversarial miner has plugin-compatible fields: `total_examples_mined`, `categories` (lines 377-402)
- Verified self-correction loop writes `status/self_correction.json` with correct schema (lines 61, 93-132)
- Archived stale task specs (TASK010, 012, 013) that described already-implemented features
- TASK011 (self-correction impact monitor) remains valid as future work ("did it help?" tracking)
- Only real gap: no mechanism to track if corrections improve error rates over time

**Previous (2025-11-24) - Code Cleanup Session:**
- Fixed pyproject.toml: Moved GPU deps (torch, transformers) to `[training]` optional extra
  - `pip install -e .` now lightweight (CI-friendly)
  - `pip install -e ".[training]"` for full GPU training
- **NEW: Two-layer validation architecture**
  - Layer 1: SpecValidator (outer gate) - denies unknown schemas
  - Layer 2: DataValidator (content) - QUICK/STANDARD/DEEP checks
- Added SpecValidator with deny-by-default schema validation
  - Jobs MUST map to a known spec (or use default: chat_sft_v1)
  - Registry: `DATASET_SPECS` in `core/validation/spec.py`
  - Known specs: `chat_sft_v1`, `syllo_v1`, `completion_v1`
- Integrated DataValidator into daemon: QUICK validation on inbox files
  - Files with schema errors rejected before entering queue
  - Comprehensive validation still runs before training
- Added answer leakage detection to DataValidator (DEEP level)
  - Detects full answer in prompt, answer previews, composition patterns
- Added resolution logging to paths.py (debug visibility)
- Documented BackgroundWorker timeout limitation (not enforced)
- Added deprecation notice to core/validator.py (use core/validation/validator.py)
- Added pytest markers: `slow`, `gpu`, `integration` for CI filtering
  - CI can run: `pytest -m "not slow and not gpu"`
- Added inference auth tests (tests/test_inference_auth.py) - 14 tests, all passing

**What Changed (Session 2 - Refactoring):**
- TASK004: Extracted ModelLoader, DatasetPreparer, MonitoringBundle from UltimateTrainer
- TASK005: Extracted PIDManager, FileWatcher, SnapshotService, BackgroundWorker from daemon
- TASK006: Added paths.py with get_base_dir() for path auto-detection
- TASK007: Created pyproject.toml - package now installable
- TASK008: Created DataValidator module (core/validation/validator.py)
- TASK009: Created BackgroundWorker for non-blocking heavy tasks

**What Changed (Session 1 - Auth & Tests):**
- TASK001: API authentication for inference server
- TASK002: Test infrastructure cleanup (pytest.ini, conftest.py)
- TASK003: RetentionManager wired into daemon

**Previous Update (2025-11-22):**
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

**Last Verified:** 2025-11-25

### Model Status
- **Base model:** Qwen3-0.6B (exists at `/path/to/training/models/Qwen3-0.6B/`)
  - Size: 1.5GB
  - Type: Qwen3ForCausalLM
- **Deployed model:** checkpoint-156000 (on 3090)
  - Location: `/path/to/models/deployed/`
  - Auto-deployed by orchestrator
- **Training method:** Full model fine-tuning (no LoRA)
- **Current training:** Running at step 156945+

### Service Status (Option C Architecture)

**4090 (Training Machine):**
- ✅ Training daemon: Running
- ✅ model_comparison_engine: Running (PID in .pids/model_comparison.pid)
- ✅ deployment_orchestrator: Running (PID in .pids/deployment_orchestrator.pid)
- ✅ Disk manager: Running

**3090 (Inference Server):**
- ✅ Inference API: Running (port 8765)
- ✅ Model loaded: checkpoint-156000 (1.2GB VRAM)
- ✅ Auto-reload: Enabled via /models/reload endpoint

### Configuration (`config.json`)

See `config.json` for current values. Key structure:
```json
{
  "model_name": "qwen3_0.6b",
  "profile": {"name": "emoji_think"},
  "hyperparams": {
    "max_length": 2048,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "learning_rate": 0.0004,
    "fp_precision": "bf16"
  },
  "auto_generate": {
    "enabled": true,
    "host": "localhost",
    "port": 8080
  }
}
```

**Note:** `config.json` is the source of truth. See `trainer/config/schema.py` for full TrainerConfig.

### Disk Space
- **Available:** 731GB / 1.8TB (58% used)
- **Status:** Healthy

---

## ⚡ QUICK OPERATIONS

### Start All Services (4090)
```bash
cd /path/to/training

# Start monitoring daemons
nohup python3 monitoring/model_comparison_engine.py --base-dir . --interval 600 > logs/model_comparison.log 2>&1 &
echo $! > .pids/model_comparison.pid

nohup python3 monitoring/deployment_orchestrator.py --base-dir . --interval 600 > logs/deployment_orchestrator.log 2>&1 &
echo $! > .pids/deployment_orchestrator.pid

# Start training daemon (if not already running)
ps aux | grep training_daemon | grep -v grep || \
nohup python3 core/training_daemon.py --base-dir . > logs/training_output.log 2>&1 &
```

### Check System Status
```bash
# Check 4090 daemons
ps aux | grep -E 'model_comparison|deployment_orchestrator|training_daemon' | grep python

# Check 3090 server
curl http://192.168.x.x:8765/health | jq .
curl http://192.168.x.x:8765/models/info | jq .

# Check deployment status
cat status/deployment_status.json | jq '.[0]'
```

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

### Important URLs

**3090 Inference Server:**
- Health: http://192.168.x.x:8765/health
- Model Info: http://192.168.x.x:8765/models/info
- Chat Completions: http://192.168.x.x:8765/v1/chat/completions (POST)

**3090 GPU Task Scheduler:**
- Health: http://192.168.x.x:8766/api/health
- Metrics: http://192.168.x.x:8766/api/metrics
- Submit Task: http://192.168.x.x:8766/api/tasks/submit (POST)
- Task Types: http://192.168.x.x:8766/api/task-types

**Synology NAS (192.168.x.x):**
- Storage API: http://localhost:8081/api/storage (via unified API)
- Storage Details: http://localhost:8081/api/storage/details
- Credentials: `.secrets/synology.json`
- Config: `config/storage.json`

**4090 Status Files:**
- Training: cat status/training_status.json
- Comparisons: cat status/model_comparisons.json
- Deployments: cat status/deployment_status.json
- Storage: cat status/storage_status.json

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

### Current State
1. ✅ Automated deployment working (Option C complete)
2. ✅ Training running normally (step 156945+)
3. ✅ 3090 serving trained model automatically
4. Next: Monitor system stability over 24 hours

### When in Doubt
1. Run health check: `python3 safety/comprehensive_health_check.py`
2. Check system state: `python3 tools/analysis/state_tracker.py --check`
3. **ASK USER** before making changes
4. **ASK USER** before creating new documentation

---

## 🔄 UPDATE LOG

**2025-11-24 (Option C Architecture):**
- ✅ Enhanced 3090 server with /models/info and /models/reload
- ✅ Created deployment_orchestrator.py (automated deployment)
- ✅ Created prediction_client.py (standardized API client)
- ✅ Moved monitoring to 4090 (comparison, orchestration)
- ✅ Achieved automated deployment: < 15 min from checkpoint to serving
- ✅ 3090 now serves trained model (was serving base model)
- ✅ Complete system operational and tested
- 📝 Documentation: OPTION_C_MIGRATION_STATUS.md

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

---

## 🤖 AUTONOMOUS SYSTEMS (Updated 2025-11-25)

**Central GPU Task Scheduler + Coordinated Daemons**

### RTX 3090 - GPU Task Scheduler (NEW)

**Central Coordinator** - `monitoring/gpu_task_scheduler.py` (port 8766)
- Monitors GPU utilization in real-time (target: 20-80%)
- Priority queue: CRITICAL(0) → HIGH(1) → NORMAL(2) → LOW(3) → IDLE(4)
- Auto-dispatches idle tasks when utilization < 20%
- 11 task types available
- API: http://192.168.x.x:8766/api/

**Scheduler-Managed Daemons** (submit tasks to scheduler):

1. **Self-Correction Loop** - `--use-scheduler` flag
   - Validates data, captures errors, generates corrections
   - Interval: 300s
   - Task type: `self_correction`

2. **Automated Testing Daemon** - `--use-scheduler` flag
   - Runs validation suite against checkpoints
   - Interval: 600s
   - Task type: `automated_test`

3. **Curriculum Eval Loop** - `--use-scheduler` flag
   - Tests model against curriculum problems
   - Task type: `curriculum_eval`

**Direct Daemons** (run independently):

4. **Inference Server** - `inference_server.py` (port 8765)
   - Serves model for all inference requests
   - Always running

5. **Model Comparison Engine** - `model_comparison_engine.py`
   - Ranks checkpoints by composite score
   - Runs every 10 minutes

### RTX 4090 Systems

6. **Training Daemon** - `core/training_daemon.py`
   - File watcher + training orchestrator
   - Always running

7. **Deployment Orchestrator** - `deployment_orchestrator.py`
   - Auto-deploys best checkpoint to 3090
   - Runs every 10 minutes

### Start All Systems

```bash
# 3090 - Start scheduler first
ssh 192.168.x.x "nohup python3 /path/to/training/monitoring/gpu_task_scheduler.py --port 8766 > logs/gpu_scheduler.log 2>&1 &"

# 3090 - Start scheduler-aware daemons
ssh 192.168.x.x "nohup python3 /path/to/training/monitoring/self_correction_loop.py --continuous --use-scheduler --interval 300 --base-dir /path/to/training > logs/self_correction.log 2>&1 &"
ssh 192.168.x.x "nohup python3 /path/to/training/monitoring/automated_testing_daemon.py --use-scheduler --interval 600 --base-dir /path/to/training > logs/automated_testing.log 2>&1 &"

# 4090 - Start local daemons
nohup python3 monitoring/deployment_orchestrator.py --base-dir . --interval 600 > logs/deployment_orchestrator.log 2>&1 &
```

### Verify Systems

```bash
# Check scheduler health
curl http://192.168.x.x:8766/api/health | jq .

# Check scheduler metrics
curl http://192.168.x.x:8766/api/metrics | jq .

# Check 3090 processes
ssh 192.168.x.x "ps aux | grep python3 | grep -E 'gpu_task|self_correction|automated_testing|inference' | grep -v grep"

# Check 4090 processes
ps aux | grep python3 | grep -E 'training_daemon|deployment_orchestrator' | grep -v grep
```

### System Outputs

Status files in `status/`:
- `self_correction.json` - Correction runs, error patterns
- `automated_testing.json` - Validation results
- `model_comparisons.json` - Checkpoint rankings
- `deployment_status.json` - Latest deployments

