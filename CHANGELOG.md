# Changelog

Track changes and updates to the system.

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
- `config.json` - Updated `auto_generate` section (count: 100, removed remote host/port)

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

