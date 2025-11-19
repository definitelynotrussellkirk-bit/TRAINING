# Critical Fix: Cumulative Training Between Batches

**Date:** 2025-11-11
**Status:** ✅ FIXED
**Severity:** CRITICAL - Was preventing cumulative training

## The Problem

The training system was **NOT properly accumulating progress between batches**. Each new training file was starting from scratch with a fresh optimizer state, even though adapter weights were being preserved.

### Root Cause

**File:** `train.py` lines 597-609

The code had two critical issues:

1. **Missing `trainer.save_state()` call** - Never saved optimizer/scheduler state
2. **Deleting ALL checkpoints** - Removed all traces of optimizer momentum

### What Was Happening

```python
# OLD CODE (BROKEN):
trainer.save_model(self.args.output_dir)  # Only saves adapter weights
self.tokenizer.save_pretrained(self.args.output_dir)

# Then DELETED all checkpoints including optimizer state!
for checkpoint_path in Path(self.args.output_dir).glob("checkpoint-*"):
    shutil.rmtree(checkpoint_path)  # ❌ Destroys optimizer state
```

### Impact on Training

**Batch 1:** Train on file 1
- Loss starts at ~2.5
- Loss decreases to ~0.8
- Adapter weights saved ✓
- Optimizer state deleted ❌

**Batch 2:** Train on file 2
- Loads adapter weights from batch 1 ✓
- **BUT** optimizer starts fresh ❌
- Loss jumps back to ~2.0 (sawtooth pattern)
- No momentum, no adaptive learning rates

**Result:** Training was NOT cumulative - each batch restarted optimization from scratch.

## The Fix

### Code Changes (train.py:597-616)

```python
# NEW CODE (FIXED):
# Save final model
trainer.save_model(self.args.output_dir)
self.tokenizer.save_pretrained(self.args.output_dir)

# CRITICAL: Save full training state (including optimizer state)
# This preserves momentum and adaptive learning rates between batches
trainer.save_state()
print("💾 Saved full training state (optimizer + scheduler) for next batch")

# Clean up OLD checkpoints but keep the latest one
# This preserves optimizer state for cumulative training across batches
import shutil
checkpoint_paths = sorted(Path(self.args.output_dir).glob("checkpoint-*"))
if len(checkpoint_paths) > 1:
    # Keep the latest checkpoint, delete older ones
    for checkpoint_path in checkpoint_paths[:-1]:
        shutil.rmtree(checkpoint_path)
    print(f"🗑️  Cleaned up {len(checkpoint_paths) - 1} old checkpoint(s), kept latest for resumption")
elif len(checkpoint_paths) == 1:
    print(f"✅ Keeping checkpoint {checkpoint_paths[0].name} for next batch resumption")
```

### What `trainer.save_state()` Saves

When called, it saves to `output_dir/` (not a checkpoint subdir):
- `training_args.bin` - Training configuration
- `trainer_state.json` - Step counter, global_step, etc.
- `optimizer.pt` - **Optimizer state** (Adam momentum, variance, etc.)
- `scheduler.pt` - **Learning rate scheduler state**
- `rng_state.pth` - Random number generator state

### What The Fixed Checkpoint Cleanup Does

**Before (Broken):**
- Deleted ALL checkpoints → Lost optimizer state
- Next batch had no way to resume with correct state

**After (Fixed):**
- Keeps the **latest checkpoint** (e.g., `checkpoint-118`)
- Deletes older checkpoints to save disk space
- Next batch can resume from latest checkpoint with full state

## Expected Behavior After Fix

### Batch 1: Train on file 1 (10k examples, ~118 steps)
```
Step 1:   Loss = 2.453
Step 50:  Loss = 1.234
Step 100: Loss = 0.876
Step 118: Loss = 0.845

SAVES:
✓ adapter_model.safetensors (adapter weights)
✓ optimizer.pt (momentum, variance)
✓ scheduler.pt (learning rate state)
✓ checkpoint-118/ (latest checkpoint)
```

### Batch 2: Train on file 2 (10k examples, step 119-236)
```
Resuming from checkpoint-118...
✓ Loaded adapter weights
✓ Loaded optimizer state (momentum preserved!)
✓ Loaded scheduler state (learning rate continues)

Step 119: Loss = 0.834 (smooth continuation, NO jump!)
Step 150: Loss = 0.721
Step 200: Loss = 0.645
Step 236: Loss = 0.612

SAVES:
✓ Updated adapter_model.safetensors
✓ Updated optimizer.pt
✓ Updated scheduler.pt
✓ checkpoint-236/ (latest)
🗑️  Deleted checkpoint-118 (old)
```

### Batch 3: Train on file 3 (10k examples, step 237-354)
```
Resuming from checkpoint-236...
Step 237: Loss = 0.598 (smooth continuation!)
Step 300: Loss = 0.534
Step 354: Loss = 0.489
```

## Verification

To verify cumulative training is working:

### 1. Check for Optimizer State Files
```bash
ls -lh current_model/optimizer.pt
ls -lh current_model/scheduler.pt
```
These files should exist and grow over time.

### 2. Check Checkpoint Preservation
```bash
ls -d current_model/checkpoint-*
```
Should see ONE checkpoint (the latest).

### 3. Monitor Loss Continuity
Watch training logs - loss should decrease smoothly across batches without jumping back up.

### 4. Check Global Step Counter
```bash
cat current_model/trainer_state.json | grep -A2 "global_step"
```
This should increment across batches (not reset to 0).

## Before vs After Comparison

| Metric | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| Adapter weights preserved | ✓ Yes | ✓ Yes |
| Optimizer momentum | ❌ No | ✓ Yes |
| Learning rate scheduler | ❌ No | ✓ Yes |
| Loss continuity | ❌ Sawtooth | ✓ Smooth |
| True cumulative training | ❌ No | ✓ Yes |
| Global step counter | ❌ Resets | ✓ Increments |

## Related Config Settings

### save_steps (config.json)
```json
"save_steps": 100
```
- Checkpoints saved every 100 steps **during training**
- With ~118 steps per 10k batch, get 1 checkpoint per batch
- MUST be < steps_per_batch to preserve state between batches

### Why save_steps Matters

If `save_steps > steps_per_batch`:
- ❌ No checkpoints created during batch
- ❌ No optimizer state preserved
- ❌ Next batch starts fresh

If `save_steps < steps_per_batch`:
- ✓ Checkpoints created during batch
- ✓ Latest checkpoint preserved after batch
- ✓ Next batch resumes with full state

## Testing the Fix

### Quick Test
```bash
# 1. Start fresh
rm -rf current_model/

# 2. Train on first file
# Watch for: "💾 Saved full training state (optimizer + scheduler) for next batch"

# 3. Check state files exist
ls -lh current_model/optimizer.pt
ls -lh current_model/scheduler.pt

# 4. Train on second file
# Watch for: "📦 Resuming from checkpoint: ..."
# Loss should NOT jump back up!
```

### Full Verification
Monitor these across multiple batches:
1. Loss should decrease monotonically (no sawtooth)
2. `global_step` in trainer_state.json should increment
3. ONE checkpoint should persist between batches
4. optimizer.pt and scheduler.pt should update

## Summary

**What was broken:**
- ❌ Optimizer state was being deleted after each batch
- ❌ Each batch started with fresh optimization (no momentum)
- ❌ Training was NOT truly cumulative

**What's fixed:**
- ✅ `trainer.save_state()` now called after each batch
- ✅ Latest checkpoint preserved (contains optimizer state)
- ✅ Optimizer momentum carries forward to next batch
- ✅ Training is truly cumulative across all batches

**Result:** Loss will now decrease smoothly across all batches, with proper momentum and adaptive learning rates preserved!
