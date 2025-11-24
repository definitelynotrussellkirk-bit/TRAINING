# Task-Based Implementation Plan - Critical Fixes

**Created:** 2025-11-24 03:50 AM
**Estimated Time:** 1 hour total
**Strategy:** Minimal surgical fixes to production code

---

## Executive Decision: Fix UltimateTrainer, Defer TrainerEngine Migration

**Rationale:**
- Daemon currently uses UltimateTrainer (running production training)
- 90% of profile system already works (logit processors, data transform)
- Only missing: system prompt integration + gradient checkpointing
- Full TrainerEngine migration = 3-4 hours + high risk
- These fixes = 30-45 minutes + low risk

**Defer to Later:**
- TrainerEngine migration
- Config format standardization
- FlashAttention in TrainerEngine (only needed if we migrate)

---

## Sprint 1: Critical Code Fixes (30-45 minutes)

### Task 1.1: Add Gradient Checkpointing Support (15 min)

**File:** `trainer/config/schema.py`
**Line:** ~25 (Hyperparams dataclass)

**Change:**
```python
@dataclass
class Hyperparams:
    # ... existing fields ...

    # NEW: Gradient checkpointing (saves ~6-8GB VRAM)
    use_gradient_checkpointing: bool = True
```

**File:** `core/train.py`
**Line:** ~370 (UltimateTrainer.load_model)

**Change:**
```python
def load_model(self):
    # ... existing model loading code ...

    # NEW: Enable gradient checkpointing if configured
    if hasattr(self.config, 'hyperparams') and \
       self.config.hyperparams.use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled (saves ~6-8GB VRAM)")

    # ... rest of function ...
```

**File:** `core/train.py`
**Line:** ~700 (prepare_trainer method, TrainingArguments creation)

**Change:**
```python
def prepare_trainer(self, ...):
    # ... existing code ...

    # NEW: Pass gradient checkpointing to TrainingArguments
    use_grad_ckpt = (
        hasattr(self.config, 'hyperparams') and
        self.config.hyperparams.use_gradient_checkpointing
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        # ... existing args ...
        gradient_checkpointing=use_grad_ckpt,  # NEW
        # ... rest of args ...
    )
```

**Expected Result:**
- GPU memory usage drops by ~6-8GB during training
- Can increase batch size 2-4x
- Small training slowdown (~10-15%) acceptable tradeoff

---

### Task 1.2: Fix System Prompt to Use Profile Template (10 min)

**File:** `core/train.py`
**Line:** ~580 (UltimateTrainer.prepare_dataset)

**Current Code:**
```python
def prepare_dataset(self, raw_dataset, tokenizer, system_prompt, data_file):
    # Uses system_prompt arg directly (wrong!)
```

**New Code:**
```python
def prepare_dataset(self, raw_dataset, tokenizer, system_prompt, data_file):
    # NEW: Check if profile provides system prompt template
    if self.profile is not None:
        from datetime import datetime
        template = self.profile.get_system_prompt_template()
        system_prompt = template.format(date=datetime.now().strftime("%Y-%m-%d"))
        logger.info(f"✓ Using profile system prompt: {system_prompt}")
    else:
        logger.info(f"Using provided system prompt: {system_prompt}")

    # ... rest of function unchanged ...
```

**Expected Result:**
- Logs show: "✓ Using profile system prompt: Current date: 2025-11-24. Respond naturally and concisely."
- Model trains with correct personality/instructions
- Backward compatible (falls back if no profile)

---

### Task 1.3: Add Model-Agnostic Validation Helper (15 min)

**File:** `core/training_daemon.py`
**Line:** ~400 (add new helper function)

**Add Function:**
```python
def is_valid_model_dir(model_path: Path) -> bool:
    """
    Check if directory contains a valid HuggingFace model.

    Supports both:
    - Full fine-tuned models (config.json + weights)
    - LoRA adapters (adapter_config.json + adapter weights)

    Args:
        model_path: Path to model directory

    Returns:
        True if valid model structure found
    """
    if not model_path.exists():
        return False

    # Check for full HuggingFace model structure
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
        logger.debug(f"✓ Valid full model found: {model_path}")
        return True

    # LoRA adapter check (backward compatibility)
    has_adapter_config = (model_path / "adapter_config.json").exists()
    has_adapter_weights = (model_path / "adapter_model.safetensors").exists()

    if has_adapter_config and has_adapter_weights:
        logger.debug(f"✓ Valid LoRA adapter found: {model_path}")
        return True

    logger.debug(f"✗ Invalid model structure: {model_path}")
    return False
```

**File:** `core/training_daemon.py`
**Line:** 445 (initialize_model method)

**Replace:**
```python
# OLD:
if (self.current_model_dir / "adapter_config.json").exists():
    logger.info(f"Using existing model: {self.current_model_dir}")
    return True

# NEW:
if is_valid_model_dir(self.current_model_dir):
    logger.info(f"✓ Using existing model: {self.current_model_dir}")
    return True
```

**File:** `core/training_daemon.py`
**Line:** 490 (verify_snapshot method)

**Replace:**
```python
# OLD:
config_file = snapshot_dir / "adapter_config.json"
if not config_file.exists():
    return False, "Missing adapter_config.json"

# NEW:
if not is_valid_model_dir(snapshot_dir):
    return False, "Invalid model structure (missing config/weights/tokenizer)"
```

**File:** `core/training_daemon.py`
**Line:** 1005 (train_on_file method)

**Replace:**
```python
# OLD:
if (self.current_model_dir / "adapter_config.json").exists():
    # ...

# NEW:
if is_valid_model_dir(self.current_model_dir):
    # ...
```

**Expected Result:**
- No more spurious "missing adapter_config.json" errors
- Works with both full models and LoRA adapters
- Logs clearly identify model type

---

### Task 1.4: Update Config to Enable Gradient Checkpointing (2 min)

**File:** `config.json`

**Add to hyperparams section:**
```json
{
  "hyperparams": {
    "fp_precision": "bf16",
    "max_length": 2048,
    "batch_size": 1,
    "gradient_accumulation": 1,
    "learning_rate": 0.0004,
    "use_gradient_checkpointing": true
  }
}
```

**Expected Result:**
- Config ready for gradient checkpointing
- Can increase batch_size after verification

---

## Sprint 2: Local Testing (15 minutes)

### Task 2.1: Test Model Validation Function (3 min)

```bash
# Create test script
cat > /tmp/test_validation.py << 'EOF'
import sys
sys.path.insert(0, '/path/to/training')
from pathlib import Path
from core.training_daemon import is_valid_model_dir

# Test with current model
model_path = Path('/path/to/training/current_model')
result = is_valid_model_dir(model_path)
print(f"Current model valid: {result}")
assert result, "Current model should be valid!"

# Test with base model
base_path = Path('/path/to/training/models/Qwen3-0.6B')
result = is_valid_model_dir(base_path)
print(f"Base model valid: {result}")
assert result, "Base model should be valid!"

# Test with non-existent path
fake_path = Path('/tmp/nonexistent')
result = is_valid_model_dir(fake_path)
print(f"Fake path valid: {result}")
assert not result, "Fake path should be invalid!"

print("✓ All validation tests passed!")
EOF

python3 /tmp/test_validation.py
```

**Expected Output:**
```
Current model valid: True
Base model valid: True
Fake path valid: False
✓ All validation tests passed!
```

**If fails:** Debug is_valid_model_dir() logic before proceeding

---

### Task 2.2: Test Gradient Checkpointing Flag (5 min)

```bash
# Test training args creation
python3 << 'EOF'
import sys
sys.path.insert(0, '/path/to/training')
from trainer.config import ConfigLoader
from pathlib import Path

config = ConfigLoader.load('/path/to/training/config.json')
print(f"Gradient checkpointing enabled: {config.hyperparams.use_gradient_checkpointing}")
assert config.hyperparams.use_gradient_checkpointing == True
print("✓ Config loads gradient checkpointing correctly")
EOF
```

**Expected Output:**
```
Gradient checkpointing enabled: True
✓ Config loads gradient checkpointing correctly
```

**If fails:** Check config.json syntax

---

### Task 2.3: Smoke Test System Prompt Integration (7 min)

```bash
# Run short training test
python3 core/train.py \
  --config config.json \
  --dataset data/validation/eval_dataset.jsonl \
  --model current_model \
  --output /tmp/smoke_test \
  --max-steps 10 \
  --skip-eval \
  2>&1 | tee /tmp/smoke_test.log

# Check for success indicators
echo "=== Checking Results ==="
grep "Using profile system prompt" /tmp/smoke_test.log && echo "✓ Profile system prompt used"
grep "Gradient checkpointing enabled" /tmp/smoke_test.log && echo "✓ Gradient checkpointing enabled"
grep "Flash Attention" /tmp/smoke_test.log && echo "✓ Flash Attention detected"

# Check for error indicators
if grep -i "error\|failed\|exception" /tmp/smoke_test.log | grep -v "no module named"; then
    echo "❌ Errors found in smoke test!"
    exit 1
else
    echo "✓ No errors in smoke test"
fi
```

**Expected Output:**
```
✓ Profile system prompt used
✓ Gradient checkpointing enabled
✓ Flash Attention detected
✓ No errors in smoke test
```

**If fails:** Review /tmp/smoke_test.log for specific errors

---

## Sprint 3: Production Deployment (20 minutes)

### Task 3.1: Backup Current State (2 min)

```bash
cd /path/to/training

# Backup config
cp config.json config.json.backup_$(date +%Y%m%d_%H%M%S)

# Git commit current changes
git add -A
git commit -m "Pre-deployment backup: gradient checkpointing + system prompt fixes"
```

---

### Task 3.2: Stop Training Daemon (1 min)

```bash
# Kill daemon gracefully
pkill -f training_daemon

# Wait for shutdown
sleep 5

# Verify stopped
if ps aux | grep training_daemon | grep -v grep; then
    echo "❌ Daemon still running, force kill"
    pkill -9 -f training_daemon
    sleep 2
fi

echo "✓ Daemon stopped"
```

---

### Task 3.3: Apply Code Changes (already done in Sprint 1)

Changes to apply:
- [x] Task 1.1 - Gradient checkpointing
- [x] Task 1.2 - System prompt fix
- [x] Task 1.3 - Model validation
- [x] Task 1.4 - Config update

---

### Task 3.4: Restart Daemon with Fixes (2 min)

```bash
cd /path/to/training

# Start daemon
nohup python3 core/training_daemon.py --base-dir $PWD > training_output.log 2>&1 &

# Get PID
DAEMON_PID=$!
echo "Daemon started with PID: $DAEMON_PID"

# Wait for initialization
sleep 5

# Verify running
if ps -p $DAEMON_PID > /dev/null; then
    echo "✓ Daemon running successfully"
else
    echo "❌ Daemon failed to start!"
    tail -50 training_output.log
    exit 1
fi
```

---

### Task 3.5: Monitor Initial Training (5 min)

```bash
# Watch logs in real-time
tail -f training_output.log | while read line; do
    echo "$line"

    # Success indicators
    echo "$line" | grep -q "Using profile system prompt" && echo "✓✓✓ PROFILE PROMPT ACTIVE ✓✓✓"
    echo "$line" | grep -q "Gradient checkpointing enabled" && echo "✓✓✓ GRADIENT CHECKPOINTING ACTIVE ✓✓✓"
    echo "$line" | grep -q "Flash Attention" && echo "✓✓✓ FLASH ATTENTION ACTIVE ✓✓✓"

    # Error indicators
    echo "$line" | grep -qi "error\|failed\|exception" && echo "❌❌❌ ERROR DETECTED ❌❌❌"
    echo "$line" | grep -q "adapter_config" && echo "❌❌❌ ADAPTER ASSUMPTION ERROR ❌❌❌"

    # Stop after first training step completes
    echo "$line" | grep -q "Step 1/" && break
done
```

**Expected to See:**
```
✓✓✓ PROFILE PROMPT ACTIVE ✓✓✓
✓✓✓ GRADIENT CHECKPOINTING ACTIVE ✓✓✓
✓✓✓ FLASH ATTENTION ACTIVE ✓✓✓
```

**Should NOT See:**
```
❌ adapter_config.json not found
❌ CUDA out of memory
```

---

### Task 3.6: Verify GPU Memory Usage (5 min)

```bash
# Check memory before training step
echo "=== GPU Memory Before Training ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# Wait for training to stabilize
sleep 60

# Check memory during training
echo "=== GPU Memory During Training ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# Compare
echo "Expected: ~14-16GB used with gradient checkpointing"
echo "Without gradient checkpointing: ~20-22GB used"
```

**Success Criteria:**
- Used memory: 14-16GB (vs 20-22GB before)
- Free memory: 8-10GB available
- No OOM errors in logs

---

### Task 3.7: Verify System Prompt (5 min)

```bash
# Check training status JSON
cat status/training_status.json | jq '{
  model_display_name: .model_display_name,
  current_step: .current_step,
  current_file: .current_file
}'

# Check recent training logs for system prompt
tail -500 training_output.log | grep -A2 "System prompt:" | tail -5

# Expected output:
# System prompt: "Current date: 2025-11-24. Respond naturally and concisely."
# NOT: "Today is 2025-11-24. You are a helpful assistant."
```

**Success Criteria:**
- System prompt matches emoji_think profile template
- Contains "Current date: YYYY-MM-DD"
- Contains "Respond naturally and concisely"

---

## Sprint 4: Extended Monitoring (30 minutes)

### Task 4.1: Monitor for OOM (10 min)

```bash
# Watch for OOM in logs
watch -n 30 '
echo "=== Training Progress ==="
tail -20 training_output.log | grep -E "Step|loss"
echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo ""
echo "=== Checking for OOM ==="
if tail -100 training_output.log | grep -i "out of memory"; then
    echo "❌ OOM DETECTED!"
else
    echo "✓ No OOM"
fi
'
```

**Monitor for 10 minutes, verify:**
- No OOM errors
- Training progressing steadily
- GPU memory stable

---

### Task 4.2: Monitor for Adapter Errors (5 min)

```bash
# Check for adapter-related errors
grep -i "adapter" training_output.log | tail -20

# Should see NOTHING or only debug messages
# Should NOT see:
# - "adapter_config.json not found"
# - "Invalid adapter"
# - "Failed to load adapter"
```

---

### Task 4.3: Verify Training Quality (15 min)

```bash
# Wait for first eval step (if enabled)
# Or check loss progression

tail -100 training_output.log | grep -E "Step [0-9]+/.*loss"

# Verify:
# - Loss is reasonable (not NaN, not exploding)
# - Steps progressing
# - No errors
```

---

## Rollback Procedure (If Anything Fails)

### Immediate Rollback

```bash
# Stop daemon
pkill -9 -f training_daemon

# Restore backup config
cp config.json.backup_YYYYMMDD_HHMMSS config.json

# Revert code changes
git checkout core/train.py core/training_daemon.py trainer/config/schema.py

# Restart with old code
nohup python3 core/training_daemon.py --base-dir $PWD > training_output.log 2>&1 &
```

---

## Success Metrics

### Must Have (Critical)
- [x] Training starts without errors
- [x] System prompt uses profile template
- [x] Gradient checkpointing enabled
- [x] No adapter_config.json errors
- [x] No OOM errors

### Nice to Have (Quality)
- [x] GPU memory reduced by 6-8GB
- [x] Training speed acceptable (~10-15% slower OK)
- [x] Batch size can be increased
- [x] All 3 fixes visible in logs

---

## Post-Deployment Tasks

### After 1 Hour of Stable Training

1. **Update CLAUDE.md** with changes made
2. **Git commit** with descriptive message
3. **Update CHANGELOG.md** with fix summary
4. **Consider increasing batch_size** (test 2, then 4)

### After 24 Hours of Stable Training

1. **Mark Sprint 1 complete** in todo list
2. **Plan Sprint 2** (config migration) if needed
3. **Evaluate TrainerEngine migration** timeline

---

## Task Checklist

### Sprint 1: Code Fixes
- [ ] 1.1 - Add gradient checkpointing to schema + UltimateTrainer
- [ ] 1.2 - Fix system prompt in prepare_dataset
- [ ] 1.3 - Add is_valid_model_dir helper + replace checks
- [ ] 1.4 - Update config.json

### Sprint 2: Local Testing
- [ ] 2.1 - Test model validation function
- [ ] 2.2 - Test gradient checkpointing flag
- [ ] 2.3 - Smoke test system prompt

### Sprint 3: Production Deploy
- [ ] 3.1 - Backup current state
- [ ] 3.2 - Stop daemon
- [ ] 3.3 - Apply code changes
- [ ] 3.4 - Restart daemon
- [ ] 3.5 - Monitor initial training
- [ ] 3.6 - Verify GPU memory
- [ ] 3.7 - Verify system prompt

### Sprint 4: Extended Monitor
- [ ] 4.1 - Monitor for OOM (10 min)
- [ ] 4.2 - Monitor for adapter errors (5 min)
- [ ] 4.3 - Verify training quality (15 min)

---

## Estimated Timeline

| Sprint | Duration | Start | End |
|--------|----------|-------|-----|
| 1. Code Fixes | 30-45 min | T+0 | T+45 |
| 2. Local Testing | 15 min | T+45 | T+60 |
| 3. Deploy | 20 min | T+60 | T+80 |
| 4. Monitor | 30 min | T+80 | T+110 |
| **Total** | **~2 hours** | | |

**Ready to Start:** Yes
**Risk Level:** Low (minimal changes, well-tested approach)
**Rollback Time:** <5 minutes if needed
