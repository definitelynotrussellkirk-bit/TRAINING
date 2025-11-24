# Daemon Refactor Plan - Critical Fixes

**Created:** 2025-11-24 03:45 AM
**Status:** Documentation Phase
**Priority:** HIGH - Fixes system prompt, OOM issues, and LoRA assumptions

---

## Executive Summary

The training system currently has **two parallel training stacks** living side-by-side:
- **Old Stack:** `UltimateTrainer` (used by daemon, has monitoring but partial profile support)
- **New Stack:** `TrainerEngine` (has full profile support but not wired to daemon)

**Critical Issues:**
1. System prompt not using profile templates (daemon uses old stack)
2. Daemon assumes LoRA/adapter model format (fails for full-model checkpoints)
3. Missing gradient checkpointing (causes OOM on 24GB GPU)
4. FlashAttention not detected in TrainerEngine
5. Config system split between flat and nested formats

**Impact:**
- Training works but uses wrong system prompt
- Checkpoint validation may fail spuriously
- OOM crashes likely with longer contexts
- Profile system only partially functional

---

## Problem 1: Daemon Still Uses Old Training Stack

### Current State

**File:** `core/training_daemon.py`
**Line ~1031:** `trainer = UltimateTrainer(...)`

The daemon creates `UltimateTrainer` instances, which:
- ✅ Has monitoring, status writing, evolution tracking
- ✅ Loads profiles for logit processors
- ❌ Does NOT use profile's system prompt template
- ❌ Partially uses TrainerConfig (builds it but doesn't fully integrate)

**Evidence from logs:**
```
System prompt: "Today is 2025-11-24. You are a helpful assistant."
```

**Expected from emoji_think profile:**
```python
def get_system_prompt_template(self) -> str:
    return "Current date: {date}. Respond naturally and concisely."
```

### Root Cause

`UltimateTrainer` builds TrainerConfig but doesn't pass it to dataset preparation:
```python
# core/train.py line ~580
def prepare_dataset(self, ...):
    # Uses system_prompt arg directly, not profile.get_system_prompt_template()
    system_prompt = self.system_prompt
    # ...
```

### Solution Options

**Option A: Quick Fix (Pragmatic)**
- Patch `UltimateTrainer.prepare_dataset()` to check profile for system prompt
- Keep daemon using `UltimateTrainer` for now
- Minimal code changes, low risk
- Estimated time: 30 minutes

**Option B: Full Migration (Ideal)**
- Wire daemon to call `TrainerEngine.run_job()` instead
- Add monitoring/status writing to `TrainerEngine`
- Migrate all daemon logic to use new stack
- Estimated time: 3-4 hours

**Recommendation:** **Option A for now**, Option B as next major refactor.

### Implementation Plan (Option A)

**Step 1:** Modify `UltimateTrainer.prepare_dataset()`
```python
# core/train.py around line 580
def prepare_dataset(self, raw_dataset, tokenizer, system_prompt, data_file):
    # NEW: Check if profile has system prompt template
    if self.profile is not None:
        template = self.profile.get_system_prompt_template()
        system_prompt = template.format(date=datetime.now().strftime("%Y-%m-%d"))
        logger.info(f"Using profile system prompt: {system_prompt}")

    # Continue with existing logic...
```

**Step 2:** Test with current training
- Let current run finish
- Restart daemon
- Verify logs show profile system prompt

**Files to modify:**
- `core/train.py` (UltimateTrainer.prepare_dataset)

**Risk:** Low - single function modification, fallback to old behavior if profile is None

---

## Problem 2: Daemon Assumes LoRA/Adapter Format

### Current State

**Affected Functions:**
1. `initialize_model()` - line 445
2. `verify_snapshot()` - line 490
3. `create_snapshot()` - line 544
4. `train_on_file()` - line 1005

**Issue:**
All check for `adapter_config.json` to determine if model/snapshot is valid.

**Evidence:**
```python
# Line 445
if (self.current_model_dir / "adapter_config.json").exists():
    logger.info(f"Using existing model: {self.current_model_dir}")
    return True
```

**Problem:**
Full-model training produces HuggingFace checkpoint structure:
```
current_model/
├── config.json              ✅ Present
├── model.safetensors        ✅ Present
├── tokenizer.json           ✅ Present
└── adapter_config.json      ❌ MISSING (not a LoRA!)
```

Daemon thinks this is invalid → tries to restore from snapshots → may fail

### Solution

Create **model-agnostic validation** function:

```python
def is_valid_model_dir(model_path: Path) -> bool:
    """
    Check if directory contains a valid HuggingFace model.

    Supports both:
    - Full models (config.json + model weights)
    - LoRA adapters (adapter_config.json + adapter weights)
    """
    # Check for HF model structure
    has_config = (model_path / "config.json").exists()
    has_tokenizer = (
        (model_path / "tokenizer.json").exists() or
        (model_path / "tokenizer_config.json").exists()
    )
    has_weights = (
        (model_path / "model.safetensors").exists() or
        (model_path / "pytorch_model.bin").exists() or
        (model_path / "model.safetensors.index.json").exists()  # Sharded
    )

    # Full model check
    if has_config and has_tokenizer and has_weights:
        return True

    # LoRA adapter check (backward compat)
    has_adapter_config = (model_path / "adapter_config.json").exists()
    has_adapter_weights = (model_path / "adapter_model.safetensors").exists()
    if has_adapter_config and has_adapter_weights:
        return True

    return False
```

### Implementation Plan

**Step 1:** Add `is_valid_model_dir()` helper
- Add to `core/training_daemon.py` around line 400
- Document both model types supported

**Step 2:** Replace all `adapter_config.json` checks
- Line 445: `initialize_model()`
- Line 490: `verify_snapshot()`
- Line 544: `create_snapshot()`
- Line 1005: `train_on_file()`

**Step 3:** Update snapshot creation logic
- Don't require adapter files in `REQUIRED_SNAPSHOT_FILES`
- Add flexible file detection

**Files to modify:**
- `core/training_daemon.py`

**Risk:** Medium - touches multiple validation points, but additive (doesn't remove existing checks)

---

## Problem 3: Missing Gradient Checkpointing

### Current State (updated 2025-11-24)

**UltimateTrainer:** ✅ Gradient checkpointing implemented via `use_gradient_checkpointing` flag
- `UltimateTrainer.load_model()` uses `config.hyperparams.use_gradient_checkpointing`
- Calls `model.gradient_checkpointing_enable()` when enabled

**TrainerEngine:** ❌ Gradient checkpointing still needs to be wired up

**Impact:**
For 0.6B model with 2048 context on 24GB GPU:
- Without checkpointing: Activations use ~8-10GB → OOM at batch_size=2
- With checkpointing: Activations use ~2-3GB → batch_size=4+ feasible

### Solution

Add gradient checkpointing flag to `TrainerConfig`:

```python
# trainer/config/schema.py
@dataclass
class Hyperparams:
    # ... existing fields ...
    use_gradient_checkpointing: bool = True  # NEW
```

Enable in both trainers:

```python
# core/train.py (UltimateTrainer)
def load_model(self, ...):
    # ... after loading model ...
    if self.config.hyperparams.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ Enabled gradient checkpointing")

    # Also pass to TrainingArguments
    training_args = TrainingArguments(
        gradient_checkpointing=self.config.hyperparams.use_gradient_checkpointing,
        # ...
    )
```

```python
# trainer/core/engine.py (TrainerEngine)
def _load_model_and_tokenizer(self, ...):
    # ... after loading model ...
    if self.config.hyperparams.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
```

### Implementation Plan

**Step 1:** Add field to `TrainerConfig`
- Modify `trainer/config/schema.py`
- Default to `True` (safe, small slowdown but prevents OOM)

**Step 2:** Enable in UltimateTrainer
- Modify `core/train.py` load_model()
- Add to TrainingArguments

**Step 3:** Enable in TrainerEngine
- Modify `trainer/core/engine.py` _load_model_and_tokenizer()

**Step 4:** Update config.json
```json
{
  "hyperparams": {
    "use_gradient_checkpointing": true
  }
}
```

**Files to modify:**
- `trainer/config/schema.py`
- `core/train.py`
- `trainer/core/engine.py`
- `config.json`

**Risk:** Low - pure addition, default is safe

---

## Problem 4: FlashAttention Not in TrainerEngine

### Current State

**UltimateTrainer** (line ~370):
```python
def load_model(self):
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        logger.info("✓ Flash Attention 2 detected")
    except ImportError:
        attn_implementation = "sdpa"
```

**TrainerEngine** (line ~220):
```python
def _load_model_and_tokenizer(self):
    model = AutoModelForCausalLM.from_pretrained(
        attn_implementation="sdpa",  # ❌ HARDCODED
    )
```

**Impact:**
- UltimateTrainer gets ~3GB memory savings with FlashAttention
- TrainerEngine doesn't → more likely to OOM

### Solution

Copy detection logic to TrainerEngine:

```python
def _load_model_and_tokenizer(self):
    # Detect FlashAttention availability
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        logger.info("✓ Flash Attention 2 detected, will use optimized attention")
    except ImportError:
        attn_implementation = "sdpa"
        logger.info("Flash Attention not available, using SDPA")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        self.config.paths.base_model,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,  # ✓ DYNAMIC
        device_map="auto",
        trust_remote_code=True,
    )
```

### Implementation Plan

**Step 1:** Add FlashAttention detection to TrainerEngine
- Modify `trainer/core/engine.py` _load_model_and_tokenizer()
- Match logic from UltimateTrainer

**Files to modify:**
- `trainer/core/engine.py`

**Risk:** Very low - pure optimization, graceful fallback

---

## Problem 5: Config System Split (Flat vs Nested)

### Current State

**Three sources of config truth:**
1. **Flat config** (legacy): `config["batch_size"]`
2. **Nested config** (new): `config["hyperparams"]["batch_size"]`
3. **CLI args**: Passed to trainer

**Issues:**
- `migrate_config.py` exists but not consistently applied
- Daemon uses flat config
- Trainers try to use both
- Easy to have drift

### Solution Strategy

**Phase 1 (This Refactor):**
- Keep both formats for backward compat
- Have ConfigLoader read flat first, nested overrides
- Document which is canonical (nested)

**Phase 2 (Future):**
- Deprecate flat format
- Require migration
- Remove dual support

### Implementation Plan

**Step 1:** Update ConfigLoader to prefer nested
```python
# trainer/config/loader.py
def load(cls, config_path):
    with open(config_path) as f:
        raw = json.load(f)

    # Read nested first (new format)
    hyperparams_dict = raw.get("hyperparams", {})

    # Backfill from flat (legacy)
    if "batch_size" in raw and "batch_size" not in hyperparams_dict:
        hyperparams_dict["batch_size"] = raw["batch_size"]
    # ... repeat for all fields ...
```

**Step 2:** Run migrate_config.py on production config.json
```bash
python3 tools/config/migrate_config.py config.json
```

**Step 3:** Document in README
- Flat format deprecated
- Nested format is canonical
- Migration tool available

**Files to modify:**
- `trainer/config/loader.py`
- `config.json` (run migration)
- `README.md` (documentation)

**Risk:** Medium - touches config loading, but additive

---

## Testing Strategy

### Pre-Flight Checks

Before applying any fixes:
1. ✅ Current training run completes or pauses safely
2. ✅ Backup current config.json
3. ✅ Backup current checkpoint directory
4. ✅ Verify git is clean (commit in-progress changes)

### Test Plan

**Phase 1: Local Testing (No Daemon)**
```bash
# Test 1: Model validation function
python3 -c "
from core.training_daemon import is_valid_model_dir
from pathlib import Path
assert is_valid_model_dir(Path('current_model'))
print('✓ Model validation works')
"

# Test 2: System prompt from profile
python3 core/train.py \
  --dataset data/validation/eval_dataset.jsonl \
  --model current_model \
  --output /tmp/test_output \
  --max-steps 5 \
  --config config.json

# Check logs for: "Using profile system prompt: Current date: 2025-11-24..."
tail -50 /tmp/test_output/trainer_log.txt | grep "profile system prompt"
```

**Phase 2: Daemon Testing**
```bash
# Test 3: Restart daemon with fixes
pkill -f training_daemon
sleep 3
nohup python3 core/training_daemon.py --base-dir $PWD > training_output.log 2>&1 &

# Test 4: Check logs for new behaviors
tail -100 training_output.log | grep -E "profile system prompt|gradient checkpoint|Flash Attention"

# Test 5: Monitor for OOM (should not happen with gradient checkpointing)
watch -n 10 "nvidia-smi | head -20"
```

**Phase 3: Verification**
- [ ] System prompt matches profile template
- [ ] No adapter_config.json errors in logs
- [ ] Gradient checkpointing enabled log message
- [ ] FlashAttention detected log message
- [ ] Training completes at least 100 steps
- [ ] GPU memory stable (no OOM)

### Rollback Plan

If anything fails:
```bash
# Stop daemon
pkill -f training_daemon

# Restore backup config
cp config.json.backup config.json

# Revert git changes
git checkout core/training_daemon.py core/train.py trainer/

# Restart with old code
nohup python3 core/training_daemon.py --base-dir $PWD > training_output.log 2>&1 &
```

---

## Implementation Order

### Sprint 1: Critical Fixes (1-2 hours)
1. ✅ Create this documentation
2. Fix LoRA assumptions (add is_valid_model_dir)
3. Fix system prompt in UltimateTrainer
4. Add gradient checkpointing
5. Add FlashAttention to TrainerEngine
6. Test locally

### Sprint 2: Config Migration (30 mins)
7. Update ConfigLoader for nested preference
8. Migrate config.json
9. Test daemon with new config

### Sprint 3: Production Deploy (15 mins)
10. Commit all changes
11. Restart daemon
12. Monitor for 30 minutes
13. Verify all fixes working

### Sprint 4: Future Work (Not Now)
- Wire daemon to TrainerEngine (full migration)
- Remove flat config support
- Add automated tests for all fixes

---

## Metrics for Success

**Before Fixes:**
- ❌ System prompt: Generic "helpful assistant"
- ❌ OOM at batch_size=2 with 2048 context
- ❌ Adapter validation errors in logs
- ❌ TrainerEngine not using FlashAttention

**After Fixes:**
- ✅ System prompt: Profile template with date
- ✅ Stable at batch_size=4+ with 2048 context
- ✅ No adapter validation errors
- ✅ FlashAttention detected and used
- ✅ Training completes full dataset without OOM

---

## Risk Assessment

| Fix | Risk | Impact if Fails | Mitigation |
|-----|------|-----------------|------------|
| LoRA validation | Medium | Training won't start | Test locally first, easy rollback |
| System prompt | Low | Wrong prompt but trains | Verify in logs before continuing |
| Gradient checkpoint | Low | Possible OOM (same as before) | Default on, easy to disable |
| FlashAttention | Very Low | Slightly more memory usage | Graceful fallback to SDPA |
| Config migration | Medium | Daemon won't start | Backup config, easy rollback |

**Overall Risk:** Medium
**Recommended:** Apply fixes during low-stakes time (not during critical training runs)

---

## Questions for User

Before proceeding:
1. Should we apply Sprint 1 fixes now (training continues running)?
2. Or wait for natural pause (file completes)?
3. Should we keep UltimateTrainer or start TrainerEngine migration?
4. Any other priorities not covered here?

---

## Appendix: Code Locations

### Files to Modify

**High Priority:**
- `core/training_daemon.py` - LoRA validation fixes
- `core/train.py` - System prompt fix, gradient checkpointing
- `trainer/core/engine.py` - FlashAttention, gradient checkpointing
- `trainer/config/schema.py` - Add gradient checkpointing flag
- `config.json` - Enable gradient checkpointing

**Medium Priority:**
- `trainer/config/loader.py` - Nested config preference
- `README.md` - Document config migration

**Future:**
- `core/training_daemon.py` - Wire to TrainerEngine
- Remove UltimateTrainer (make legacy CLI)

### Key Line Numbers

```
core/training_daemon.py:
  Line 445  - initialize_model() adapter check
  Line 490  - verify_snapshot() adapter check
  Line 544  - create_snapshot() adapter files
  Line 1005 - train_on_file() adapter check
  Line 1031 - Creates UltimateTrainer instance

core/train.py:
  Line 370  - load_model() FlashAttention detection
  Line 580  - prepare_dataset() system prompt usage

trainer/core/engine.py:
  Line 220  - _load_model_and_tokenizer() hardcoded attn

trainer/config/schema.py:
  Line 25   - Hyperparams dataclass
```

---

## Status: Ready for Review

**Next Steps:**
1. User reviews this plan
2. Decide on timing (now vs later)
3. Prioritize which sprints to execute
4. Begin implementation with testing at each step

**Document Version:** 1.0
**Last Updated:** 2025-11-24 03:45 AM
