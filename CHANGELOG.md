# Changelog

Track changes and updates to the system.

---

## 2025-11-27 - 4B Full Fine-Tune on Single GPU

### What
Full fine-tuning of Qwen3-4B (4 billion parameters) now works on a single 24GB RTX 4090.

### Key Stats
- Peak VRAM: **18.6 GB** (5.4 GB headroom)
- Effective batch size: 32
- Speed: ~8s/step

### Optimizations Used
1. **Paged 8-bit Adam** - Offloads optimizer states to CPU when GPU fills
2. **Liger Kernel** - Fused ops, logits never materialized
3. **Gradient Checkpointing** - Trades compute for VRAM
4. **bf16 Precision** - Half memory footprint

### What Didn't Work
- DeepSpeed ZeRO-3: Triggered Linux OOM killer (178GB virtual memory allocation)
- DeepSpeed ZeRO-2: Same issue
- Standard 8-bit Adam: OOM at 24.03 GB during optimizer init

### Files
- `scripts/train_4b_full.py` - Training script
- `configs/ds_zero2_offload.json` - DeepSpeed config (alternative)

### Usage
```bash
python3 scripts/train_4b_full.py
```

---

## 2025-11-27 - Muon Optimizer Integration

### What
Added Muon optimizer as an alternative to AdamW. Muon uses orthogonalized momentum (Newton-Schulz iteration) for faster convergence on transformer hidden layers.

### Why
- Muon holds NanoGPT and CIFAR-10 speedrun records
- Used in KIMI Moonshot 1T+ parameter training
- ~25% less optimizer memory than AdamW
- Built-in muP scaling for better hyperparameter transfer

### Implementation
- `trainer/optimizers/muon.py` - Vendored SingleDeviceMuonWithAuxAdam
- `trainer/optimizers/param_groups.py` - Automatic parameter splitting
- `trainer/optimizers/factory.py` - Config-driven optimizer creation
- Integration with HF Trainer via `optimizers=` parameter

### Parameter Grouping
Muon applies to 73.9% of parameters (hidden weights):
- Attention: q_proj, k_proj, v_proj, o_proj
- MLP: gate_proj, up_proj, down_proj

AdamW applies to 26.1% (other):
- embed_tokens, lm_head, layernorms, biases

### Usage
```json
{
  "optimizer": {
    "type": "muon",
    "muon": {"hidden_lr": 0.02, "aux_lr": 0.0003, "momentum": 0.95}
  }
}
```

### Attribution
Muon by Keller Jordan: https://github.com/KellerJordan/Muon

---

## 2025-11-27 - Critical Bug Fix: Packing + Masking

### The Bug
Model was outputting garbage like "You are happy. You enjoy helping others." - training on instruction text instead of responses.

### Root Cause
Packing combines multiple examples into one sequence, but the collator only masked the FIRST instruction segment. The model was being trained on ALL subsequent instructions, system prompts, and user messages.

### Fix
- `custom_collator.py` now finds ALL `<|im_start|>assistant` → `<|im_end|>` segments
- New `masking_validators.py` with 5 validators to prevent recurrence:
  1. `MaskingRatioValidator` - Ensures 30-85% masked
  2. `ResponseTemplateCountValidator` - Verifies template count matches regions
  3. `TrainedTokenContentValidator` - Checks for instruction markers in trained tokens
  4. `PackedSequenceValidator` - Validates mask/train alternation pattern
  5. `LabelDistributionValidator` - Detects anomalous label patterns
- `train.py` now runs full validation suite before training
- Training aborts if masking < 25%

---

## 2025-11-27 - Sparring with the Trainers + Task Master

### Sparring System
Self-correction training: DIO spars with skill trainers, learns from mistakes.

Every wrong answer generates 3 training examples:
1. Identify incorrect: "Is this correct?" → "It is incorrect."
2. Correct it: "Find the correct solution." → [golden answer]
3. Confirm correct: [fresh problem] "Is this correct?" → "It is correct."

Sparring data always goes to HIGH priority queue (checkpoint-specific, becomes stale).

### Task Master
GPU-aware background task scheduler (`guild/task_master.py`):
- Monitors 3090 GPU utilization
- Runs tasks when utilization <40%
- Task Registry (`guild/task_registry.py`) with 11 registered tasks
- Usage: `--status`, `--once`, `--daemon`, `--run TASK`

---

## 2025-11-27 - The Weaver (Daemon Orchestrator)

One daemon to rule them all - The Weaver watches all service threads:
- Auto-restart dead daemons
- Generates training data when queue runs low
- Threads monitored: Training Daemon, Tavern, VaultKeeper, Data Flow
- Simple startup: `./scripts/start_all.sh`
- Status check: `python3 weaver/weaver.py --status`

---

## 2025-11-27 - Oracle + Checkpoint Ledger + Host Registry

### Oracle (Talk to DIO)
- Strict version checking - no fallback to wrong model
- Chat API requires explicit step parameter
- Response includes server-confirmed `model` and `model_path`
- Multi-model display shows ALL loaded models

### Checkpoint Ledger
Single source of truth for checkpoint stats (`core/checkpoint_ledger.py`):
- Canonical naming: `checkpoint-{step}-{YYYYMMDD}-{HHMM}`
- Sidecar files: `.ledger.json` with stats at save time
- Ledger API on VaultKeeper: `/api/ledger/*`

### Host Registry
Service discovery for distributed operation (`core/hosts.py`):
- `config/hosts.json` is single source of truth
- `get_service_url("inference")` instead of hardcoded IPs
- Auto-detection of local host

---

## 2025-11-27 - Tavern UI Expansion

### Quests Page (`/quests`)
Full quest board with queue management:
- View queued/processing/completed/failed quests
- Change priority, delete, retry failed quests
- Auto-refresh every 10 seconds

### VRAM Calculator (`/settings`)
Estimate GPU memory usage:
- Based on batch size, max length, precision, gradient checkpointing
- GPU presets (RTX 4090/3090/4080/4070)
- Visual breakdown: model weights, optimizer, gradients, activations

### Scheduler in Settings
Full curriculum scheduler integration:
- Quick presets (8 options)
- Strategy selection (equal, focus, weighted, catch-up)
- Per-skill enable/disable, priority, weight

---

## 2025-11-27 - Task Master UI Integration

### API Endpoint
- Added `/api/task-master` to Tavern server
- Returns GPU status, last task, stats, daemon status

### Guild Hall Card
New "Task Master (3090)" section showing:
- 3090 GPU status (idle/busy/offline)
- Last task name + outcome + time
- Task stats (run/succeeded/failed)
- Daemon status (running/stopped)

### Action Hints
Smart contextual advice in main game UI:
- Overfitting warnings (high val-train gap)
- Queue status alerts
- 3090 idle suggestions
- Skill accuracy hints

---

## 2025-11-27 - VRAM Estimator Improvements

### New Parameters
`estimate_memory()` now accepts:
- `max_length` - Sequence length (default: 2048)
- `gradient_checkpointing` - Whether checkpointing enabled (default: True)

### Updated Formula
```
activation_memory = batch_size × 0.8 GB × (max_length / 2048) × checkpoint_factor
checkpoint_factor = 0.35 if gradient_checkpointing else 1.0
```

### Why This Matters
- **Before:** Changing `max_length` from 2048→4096 showed no VRAM change
- **After:** Properly shows ~2x activation memory for doubled sequence length
- Gradient checkpointing now reflected (~65% activation reduction)

### Validation
Estimates match empirical data from TROUBLESHOOTING.md:
- bs=16 → 13.6 GB estimated vs ~14 GB empirical
- bs=12 → 10.4 GB estimated vs ~10 GB empirical
- bs=8 → 7.2 GB estimated vs ~7 GB empirical

---

## 2025-11-26 - Data Lineage System

### Overview
Track generator and validator provenance for all training data. Answers questions like:
- "Which generator produces the most rejections?"
- "Did quality drop after generator version bump?"
- "Which validator is most aggressive?"

### New Files
- `core/lineage.py` - Generator registry + FileLineage dataclass + sidecar I/O utilities
- `core/lineage_tracker.py` - LineageTracker class for aggregating per-generator/validator stats

### Generator Versioning
Added `GENERATOR_ID` and `GENERATOR_VERSION` constants to:
- `monitoring/discrimination_generator.py` - `discrimination@1.0.0`
- `data_manager/generators/syllogism_generator.py` - `syllo_local@1.0.0`

Generators now write `.meta.json` sidecar files alongside JSONL output:
```json
{
  "generator_id": "discrimination",
  "generator_version": "1.0.0",
  "generated_at": "2025-11-26T...",
  "example_count": 100,
  "params": {"levels": [1], "difficulty": "L1"}
}
```

### Validator Versioning
Added `VALIDATOR_NAME` and `VALIDATOR_VERSION` constants to:
- `core/validation/validator.py` - `data_validator@1.0.0`
- `core/validation/spec.py` - `spec_validator@1.0.0`

`ValidationResult` extended with lineage fields: `generator_id`, `generator_version`, `validator_name`, `validator_version`

### Lineage Tracking Integration
- `core/training_daemon.py` - Integrated LineageTracker, records validation outcomes
- Stats persisted to `status/data_lineage.json`

### Dashboard & API
- New `/api/lineage` endpoint in `monitoring/api/server.py`
- New "Data Lineage" card in `monitoring/ui/master_dashboard.html`
- CSS styles in `monitoring/css/master_dashboard.css`
- JS functions `fetchDataLineage()` and `updateDataLineage()` in `monitoring/js/master_dashboard.js`

### Usage
```bash
# View lineage stats via API
curl http://localhost:8081/api/lineage | jq .

# View in dashboard
http://localhost:8081/master_dashboard.html  # Data Lineage card in left column

# Bump generator version when logic changes
# In monitoring/discrimination_generator.py:
GENERATOR_VERSION = "1.1.0"  # Was "1.0.0"
```

---

## 2025-11-25 - Synology NAS Storage Integration

### New Storage Module
- **New:** `monitoring/storage_manager.py` - Synology NAS manager with DSM API + SSH fallback
- **New:** `config/storage.json` - Storage allocation config (10TB total)
- **New:** `monitoring/api/plugins/storage.py` - Storage plugin for unified dashboard API

### NAS Folder Structure (10TB Allocated)
```
/volume1/data/llm_training/
├── checkpoints/     (3TB max, 30 days retention)
├── models/          (2TB max, permanent)
├── backups/         (500GB max, 90 days retention)
├── logs/            (100GB max, 180 days retention)
├── training_data/   (2TB max, permanent)
├── snapshots/       (2TB max, 14 days retention)
└── exports/         (500GB max, permanent)
```

### Features
- DSM API authentication with session management
- SSH fallback for basic disk usage
- File upload via FileStation API
- Status monitoring with health checks
- Dashboard integration via `/api/storage` endpoint

### Usage
```bash
# Check storage status
curl http://localhost:8081/api/storage

# Sync via GPU scheduler
curl -X POST http://192.168.x.x:8766/api/tasks/submit \
  -d '{"task_type": "storage_sync", "params": {"local_path": "...", "folder_type": "checkpoints"}}'
```

---

## 2025-11-25 - Curriculum-Based Data Generation

### Data Manager Rewrite
- **DataManager** now uses local singleSKILL APIs instead of remote GPU (3090)
- Uses `SkillAPIClient` to connect to SYLLO (8080) and Binary (8090) servers
- Integrates `CurriculumManager` for adaptive difficulty progression

### Curriculum System
- **Skill-based curriculum:** SYLLO (5 levels) and Binary (7 levels)
- **Active skill:** Set via `data_manager/curriculum_state.json`
- **Progression:** 80% accuracy over 3 evaluations to advance
- **SYLLO-only for now:** Focus on mastering SYLLO before introducing Binary

### File Changes
- `data_manager/manager.py` - Rewritten to use skill APIs + curriculum
- `data_manager/skill_api_client.py` - Client for singleSKILL APIs
- `data_manager/curriculum_manager.py` - Difficulty progression logic
- `config.json` - Updated `auto_generate` section (count: 100, switched from remote 192.168.x.x to local skill APIs: host: "localhost", port: 8080)

### Generated File Naming
- Old: `syllo_autogen_TIMESTAMP_countN.jsonl`
- New: `train_SKILL_levelN_COUNT_TIMESTAMP.jsonl` (e.g., `train_syllo_level1_100_20251125_060704.jsonl`)

### Requirements
- SYLLO API must be running: `cd /path/to/skills && python3 skill_syllo_variant/api_server.py --port 8080`

---

## 2025-11-25 - Transfer Learning Metrics + Dashboard Updates

### Skill Metrics Plugin
- **New:** `monitoring/api/plugins/skill_metrics.py` - Aggregates baseline test results
- Tracks 3 skill categories: trained (syllable, binary), primitives (26), benchmarks (bAbI, BIG-Bench)
- Compares base model vs trained model performance
- Reads from local + remote baseline results via SSH

### Dashboard Transfer Learning Card
- Added Transfer Learning summary card to master dashboard
- Shows base/trained accuracy and delta for each skill category
- Highlights best/worst transfer performers

### Data Organization
- Moved validation datasets to `data/validation/benchmarks/`, `data/validation/binary/`, `data/validation/primitives/`
- Old flat structure archived to `data/validation_archive_20251124/`

---

## 2025-11-24 - Two-Layer Validation System + Refactoring Cleanup

### Validation Architecture
- **New:** `core/validation/spec.py` - SpecValidator with deny-by-default schema validation
- **New:** `core/validation/validator.py` - DataValidator (QUICK/STANDARD/DEEP content checks)
- Jobs must map to known spec: `chat_sft_v1`, `syllo_v1`, `completion_v1`
- QUICK validation integrated into daemon for early rejection

### Daemon Service Extraction
- Extracted `PIDManager` to `core/daemon/pid_manager.py`
- Extracted `FileWatcher` to `core/daemon/file_watcher.py`
- Extracted `SnapshotService` to `core/daemon/snapshot_service.py`
- Extracted `BackgroundWorker` to `core/daemon/background_worker.py`

### Training Component Extraction
- Extracted `ModelLoader` to `core/training/model_loader.py`
- Extracted `DatasetPreparer` to `core/training/dataset_preparer.py`
- Extracted `MonitoringBundle` to `core/training/monitoring_bundle.py`

### Package Structure
- **New:** `pyproject.toml` - Package now pip-installable
- GPU deps (torch, transformers) moved to optional `[training]` extra
- `pip install -e .` for lightweight CI, `pip install -e ".[training]"` for GPU

### Path Auto-Detection
- **New:** `core/paths.py` with `get_base_dir()` function
- Auto-detects base directory with resolution logging

### Testing
- Added pytest markers: `slow`, `gpu`, `integration`
- Added `tests/test_inference_auth.py` (14 tests)
- Deprecation notice added to `core/validator.py`

---

## 2025-11-23 - Autonomous Training System: 10 Intelligence Systems Deployed

**STATUS: Production Ready** - Complete autonomous training intelligence deployed across dual-GPU setup

### What Was Built

Deployed 10 intelligent automation systems across dual-GPU setup:

**RTX 3090 (7 systems):**
- `curriculum_optimizer.py` - A/B tests curriculum strategies
- `adversarial_miner.py` - Mines hard examples from failures
- `continuous_regression_monitor.py` - Detects bad checkpoints
- `model_comparison_engine.py` - Ranks checkpoints
- `confidence_calibrator.py` - Calibrates predictions
- `self_correction_loop.py` - Data validation & error correction
- `automated_testing_daemon.py` - Continuous checkpoint validation

**RTX 4090 (2 systems):**
- `data_generation_automation.py` - Auto-generates when queue < 2
- `checkpoint_auto_deployment.py` - Auto-deploys best checkpoint

### Autonomous Capabilities
- Evaluates checkpoints against validation data
- Optimizes curriculum difficulty progression
- Detects performance regressions
- Auto-generates training data when depleted
- Auto-deploys best checkpoint to inference server

---

## 2025-11-22 - Complete 4-Phase Refactor Summary

### Phase 1: PreviewBackend System (da66b96)

**Created:** `trainer/monitoring/preview_backend.py` (431 lines)

- `PreviewBackend` protocol
- `LocalPreviewBackend` - Current behavior (training GPU)
- `Remote3090Backend` - Future 3090 API integration
- `create_preview_backend()` - Factory function

**Impact:** Can offload expensive preview generation to 3090 by config change

### Phase 2: Config Cleanup (da66b96)

**Expanded:** `MonitoringConfig` with all magic numbers:
- `max_output_tokens`, `max_eval_tokens`
- `preview_backend`, `preview_max_tokens`, `preview_temperature`
- `remote_3090_host`, `remote_3090_port`, `remote_3090_timeout`

**Updated:** `core/train.py` to use config values

**Impact:** All configuration centralized in schema

### Phase 3: TrainerEngine Design (45b6b64)

**Created:** `scratch/TRAINERENGINE_DESIGN.md` (250 lines)

- Complete architecture design
- Method signatures
- Code extraction plan
- Testing strategy

**Impact:** Clear roadmap for implementation

### Phase 4: TrainerEngine Implementation (156b046) ← THIS UPDATE

**Implemented:** Full `trainer/core/engine.py` (457 lines)

**Impact:** Production-ready clean training API

---

## 2025-11-22 (Earlier) - Production Integration: trainer/ modules → core/train.py

**Major architectural milestone:** Integrated refactored `trainer/` modules into production training script.

### What Changed

**core/train.py** now uses clean config/profile architecture while maintaining 100% backward compatibility.

#### 1. Unified Configuration (ConfigLoader)
- **Before:** Scattered `config.json` reads, mixed with CLI args, conflicting sources
- **After:** Single `TrainerConfig` created from `ConfigLoader.from_args_and_json()`
- **Impact:** One source of truth for all config values

#### 2. Profile-Based Data Transformation
- **Before:** Hard-coded `enforce_thinking_requirement()` and `enforce_stop_requirement()` in core
- **After:** Delegated to profiles via `profile.transform_example()`
- **Impact:** Can switch between `emoji_think` and `regime3` without code edits

#### 3. Unified Precision Handling
- **Before:** Model loaded with `torch.bfloat16`, Training used `fp16=True` (inconsistent)
- **After:** Both read from `config.hyperparams.fp_precision` (bf16/fp16/fp32)
- **Impact:** No more precision mismatch between model and training

#### 4. System Prompt Fixed
- **Before:** Hard-coded in `prepare_dataset()`, ignored `--system-prompt` CLI arg
- **After:** Uses `args.system_prompt` consistently
- **Impact:** CLI arg actually works now

### New Capabilities

- **Profile switching:** Change `config.json` → `"profile": {"name": "regime3"}` → symbolic reasoning mode
- **Precision config:** Set `"fp_precision": "bf16"` in config instead of editing code
- **Single config:** No more hunting for where values are set

### Files Modified

```
core/train.py              +160 -73   Integrated ConfigLoader, profiles, precision
config.json                +14        Added profile, hyperparams, locked sections
core/train_v2_backup.py    NEW        Backup before migration
```

### Migration Details

**Integration points:**
1. `UltimateTrainer.__init__()` - Creates `TrainerConfig` from args + config.json
2. `load_model()` - Loads profile, builds logit processors, sets precision
3. `prepare_dataset()` - Uses `profile.transform_example()` for data transformation
4. `setup_training()` - Sets `fp16`/`bf16` flags from config

**Fallback behavior:**
- If `TrainerConfig` creation fails → uses `args` directly
- If profile loading fails → uses legacy emoji_think logic
- No breaking changes to existing workflows

### Testing

```bash
# Test imports
python3 -c "from train import UltimateTrainer; print('✓ Success')"

# Test with regime3 profile
# Edit config.json: "profile": {"name": "regime3"}
python3 core/train.py --dataset data.jsonl --model qwen3 --output outputs
```

### Related Work

- **Steps 1-5 Refactor (2025-11-22 earlier):** Created `trainer/` module architecture
  - `trainer/config/` - ConfigLoader, TrainerConfig schema
  - `trainer/profiles/` - emoji_think, regime3 profiles
  - `trainer/monitoring/` - Callbacks, status writer
  - `trainer/core/` - TrainerEngine API (proof-of-concept)

- **This update:** Brought that architecture into production (`core/train.py`)

### Git Tags

- `trainer_v1_emoji_baseline` - Baseline before refactor
- `refactor_step1_config` - Config extraction
- `refactor_step2_profiles` - Profile extraction
- `refactor_step3_monitoring` - Monitoring extraction
- `refactor_step4_engine` - TrainerEngine API
- `refactor_step5_regime3` - Regime3 profile
- **NEW:** `production_integration` - This migration

---

## 2025-11-22 (Earlier) - System Reorganization

- Reorganized 99 Python files into 6 categories
- Created 7 canonical documentation files
- Removed 96+ old documentation files
- Fresh start with code as ground truth

---

## Prior History

See git log for earlier changes.

