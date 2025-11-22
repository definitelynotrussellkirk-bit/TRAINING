# Changelog

Track changes and updates to the system.

---

## 2025-11-22 - Production Integration: trainer/ modules → core/train.py

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
