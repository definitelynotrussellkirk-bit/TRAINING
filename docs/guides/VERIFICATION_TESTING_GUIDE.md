# Continuous Training Verification & Testing Guide

**Date:** 2025-11-11
**Purpose:** Verify that continuous training is working correctly after the 2025-11-11 fix

---

## Prerequisites

Before testing, ensure:
1. The daemon is stopped: `ps aux | grep training_daemon | grep -v grep` (should be empty)
2. You have backed up `current_model/` if it contains important state
3. You have 2-3 small test data files (~1000 examples each) in a safe location

---

## Option A: Fresh Start Testing (Recommended)

This option starts from scratch to verify everything works cleanly.

### Step 1: Clean Slate

```bash
cd /path/to/training

# Stop daemon if running
ps aux | grep training_daemon | grep -v grep | awk '{print $2}' | xargs kill

# Archive current state (optional backup)
if [ -d current_model ]; then
    mv current_model current_model.backup_$(date +%Y%m%d_%H%M%S)
fi

# Clean inbox
rm -rf inbox/*

# Verify clean state
ls current_model  # Should not exist or be empty
ls inbox          # Should be empty
```

### Step 2: Prepare Test Data

```bash
# Create 3 small test files (or copy from LEO outputs)
# Each file should have ~1000-2000 examples for quick testing

# Option A: Copy from LEO
cp /path/to/leo/outputs/small_dataset_1.jsonl inbox/test_batch_1.jsonl
cp /path/to/leo/outputs/small_dataset_2.jsonl inbox/test_batch_2.jsonl
cp /path/to/leo/outputs/small_dataset_3.jsonl inbox/test_batch_3.jsonl

# Option B: Create small test files from existing data
head -n 1000 /path/to/big_dataset.jsonl > inbox/test_batch_1.jsonl
tail -n 1000 /path/to/big_dataset.jsonl > inbox/test_batch_2.jsonl
```

### Step 3: Start Daemon

```bash
# Start daemon
rm -f .stop
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# Verify it started
ps aux | grep training_daemon | grep -v grep

# Watch initial logs
tail -f training_output.log
# (Press Ctrl+C to stop watching)
```

### Step 4: Monitor First Batch

Watch for these key indicators:

```bash
# Check status every 30 seconds
watch -n 30 'cat status/training_status.json | jq "{status, current_step, total_steps, batch_number, batch_queue_size}"'

# Check checkpoint creation
watch -n 30 'ls -lhrt current_model/checkpoint-*/ 2>/dev/null | tail -20'

# Check daemon logs
tail -f logs/daemon_$(date +%Y%m%d).log
```

**Expected behavior for Batch 1:**
- Training starts from step 0
- Checkpoints appear at step 100, 200, etc. (every 100 steps)
- Each checkpoint directory contains: `adapter_model.safetensors`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`
- Batch completes (status → "idle")
- File `test_batch_1.jsonl` is deleted from inbox

### Step 5: Verify Checkpoint State After Batch 1

```bash
# List checkpoints
ls current_model/checkpoint-*/

# Check latest checkpoint's global_step
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "Latest checkpoint: $LATEST"
cat $LATEST/trainer_state.json | jq '.global_step'

# Example: If batch had ~1000 examples with effective batch size 8:
# Total steps ≈ 1000 / 8 = 125 steps
# Latest checkpoint should be checkpoint-100/ or checkpoint-200/
# global_step in trainer_state.json should be 100 or 200
```

**Verification Checklist for Batch 1:**
- [ ] Checkpoints exist in `current_model/checkpoint-*/`
- [ ] Latest checkpoint contains all required files (optimizer.pt, scheduler.pt, trainer_state.json)
- [ ] `trainer_state.json` shows `global_step` > 0
- [ ] Training completed without errors
- [ ] `test_batch_1.jsonl` was deleted from inbox

### Step 6: Monitor Second Batch

**Expected behavior for Batch 2:**
- Daemon picks up `test_batch_2.jsonl`
- Finds latest checkpoint (e.g., `checkpoint-100`)
- Logs: "Resuming from checkpoint: checkpoint-100"
- Training continues from step 100 (not step 0!)
- New checkpoints appear at step 200, 300, etc.
- Batch completes
- File `test_batch_2.jsonl` is deleted

```bash
# Watch for resumption message
tail -f logs/daemon_$(date +%Y%m%d).log | grep -i "resuming"

# Check checkpoints after batch 2
ls -lhrt current_model/checkpoint-*/

# Verify global_step incremented (should be ~225-250 now, not reset to ~125!)
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
cat $LATEST/trainer_state.json | jq '.global_step'
```

**Verification Checklist for Batch 2:**
- [ ] Logs show "Resuming from checkpoint: checkpoint-XXX"
- [ ] global_step continued from Batch 1 (e.g., 100 → 200+, not reset to 0)
- [ ] Loss continued decreasing smoothly (no spike back to initial loss)
- [ ] New checkpoints created (e.g., checkpoint-200, checkpoint-300)
- [ ] Old checkpoints auto-deleted (only 3 newest kept due to save_total_limit=3)
- [ ] `test_batch_2.jsonl` was deleted from inbox

### Step 7: Monitor Third Batch

Same process, verify:

```bash
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
cat $LATEST/trainer_state.json | jq '.global_step'

# Should be ~375-400 now (cumulative across all 3 batches!)
```

**Verification Checklist for Batch 3:**
- [ ] Resumed from latest checkpoint
- [ ] global_step ~350-400 (cumulative from all batches)
- [ ] Loss smoothly decreasing (no resets or spikes)
- [ ] Checkpoints auto-managed (only last 3 kept)

---

## Option B: Testing With Existing State

If you have a working `current_model/` and want to verify it continues correctly:

### Step 1: Check Current State

```bash
cd /path/to/training

# Find latest checkpoint
ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1

# Check global_step
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
cat $LATEST/trainer_state.json | jq '.global_step'

# Note this number (e.g., 237)
```

### Step 2: Add New Test Batch

```bash
# Add a small test file to inbox
cp /path/to/small_test.jsonl inbox/verification_test.jsonl

# Count lines (to estimate steps)
wc -l inbox/verification_test.jsonl
# Example: 1000 lines = ~125 steps (with effective batch size 8)
```

### Step 3: Start Daemon and Monitor

```bash
# Start if not running
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# Watch for resumption
tail -f logs/daemon_$(date +%Y%m%d).log | grep -i "resuming\|checkpoint"

# After training completes, check new global_step
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
cat $LATEST/trainer_state.json | jq '.global_step'

# Should be OLD_STEP + NEW_STEPS
# Example: 237 + 125 = 362
```

**Verification:**
- [ ] Resumed from existing checkpoint
- [ ] global_step increased by expected amount
- [ ] No errors in logs
- [ ] Loss continued decreasing smoothly

---

## Key Success Indicators

### ✅ Working Correctly:

1. **global_step increases continuously**
   ```bash
   # After each batch:
   # Batch 1: checkpoint-100/trainer_state.json → global_step: 100
   # Batch 2: checkpoint-200/trainer_state.json → global_step: 200
   # Batch 3: checkpoint-300/trainer_state.json → global_step: 300
   ```

2. **Checkpoints auto-managed**
   ```bash
   # Only 3 most recent checkpoints exist
   ls current_model/checkpoint-*
   # Example: checkpoint-300, checkpoint-400, checkpoint-500
   ```

3. **Loss decreases smoothly**
   ```
   Batch 1: 2.5 → 2.0 → 1.7
   Batch 2: 1.7 → 1.4 → 1.2  # Continues from 1.7, no jump!
   Batch 3: 1.2 → 1.0 → 0.9  # Continues from 1.2
   ```

4. **Optimizer state preserved**
   ```bash
   # Each checkpoint has optimizer.pt
   ls -lh current_model/checkpoint-*/optimizer.pt
   ```

### ❌ Something Wrong:

1. **global_step resets between batches**
   ```
   Batch 1: checkpoint-100 → global_step: 100
   Batch 2: checkpoint-100 → global_step: 100  # WRONG! Should be ~200
   ```

2. **Loss spikes between batches**
   ```
   Batch 1: 2.5 → 1.5
   Batch 2: 2.5 → 1.6  # WRONG! Restarted from high loss
   ```

3. **Checkpoints accumulating**
   ```bash
   ls current_model/checkpoint-*
   # 50 checkpoints exist  # WRONG! Should only keep 3
   ```

4. **Missing state files**
   ```bash
   ls current_model/checkpoint-300/
   # Only adapter_model.safetensors exists
   # Missing: optimizer.pt, scheduler.pt  # WRONG!
   ```

---

## Debugging Commands

If something doesn't look right:

```bash
# Check latest checkpoint contents
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "Latest checkpoint: $LATEST"
ls -lh $LATEST/

# Verify all required files exist
for file in adapter_model.safetensors optimizer.pt scheduler.pt trainer_state.json; do
    if [ -f "$LATEST/$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file MISSING!"
    fi
done

# Check global_step
cat $LATEST/trainer_state.json | jq '.global_step'

# Check logs for errors
grep -i error logs/daemon_$(date +%Y%m%d).log | tail -20

# Check if resumption happened
grep -i "resuming from checkpoint" logs/daemon_$(date +%Y%m%d).log | tail -5

# Verify TrainingArguments settings
grep -A 5 "save_total_limit" train.py
grep -A 5 "save_steps" train.py
```

---

## Common Issues and Fixes

### Issue: global_step not incrementing

**Symptom:** Each batch shows global_step starting from 0

**Cause:** Not resuming from checkpoint

**Fix:**
- Check logs for "Resuming from checkpoint" message
- Verify checkpoint-finding logic in train.py (lines 649-659)
- Ensure checkpoints exist before second batch starts

### Issue: Loss spikes between batches

**Symptom:** Loss drops during batch, then jumps back up for next batch

**Cause:** Optimizer state not being loaded

**Fix:**
- Verify `optimizer.pt` exists in checkpoints
- Check that `resume_from_checkpoint` is passed to `trainer.train()`
- Ensure using correct checkpoint directory (not just weights)

### Issue: Too many checkpoints

**Symptom:** Dozens of checkpoint directories exist

**Cause:** `save_total_limit` not working or set too high

**Fix:**
- Verify `save_total_limit=3` in TrainingArguments (train.py:411)
- Check no manual checkpoint deletion is interfering

### Issue: Checkpoints missing files

**Symptom:** Checkpoint directories don't contain optimizer.pt or scheduler.pt

**Cause:** Using `save_model()` instead of Trainer's checkpoint system

**Fix:**
- Verify code uses `trainer.train()` (not manual saves)
- Check `save_strategy="steps"` in TrainingArguments
- Ensure not manually creating checkpoint directories

---

## Final Validation

After running 3 test batches successfully:

```bash
# Generate report
echo "=== Continuous Training Verification Report ===" > verification_report.txt
echo "Date: $(date)" >> verification_report.txt
echo "" >> verification_report.txt

# List all checkpoints
echo "Checkpoints:" >> verification_report.txt
ls -lh current_model/checkpoint-*/ >> verification_report.txt 2>&1

# Show global_step progression
echo "" >> verification_report.txt
echo "Global step progression:" >> verification_report.txt
for ckpt in $(ls -d current_model/checkpoint-* | sort -t- -k2 -n); do
    step=$(cat $ckpt/trainer_state.json | jq -r '.global_step')
    echo "$ckpt: global_step=$step" >> verification_report.txt
done

# Check for required files in latest checkpoint
echo "" >> verification_report.txt
echo "Latest checkpoint contents:" >> verification_report.txt
LATEST=$(ls -d current_model/checkpoint-* | sort -t- -k2 -n | tail -1)
ls -lh $LATEST/ >> verification_report.txt

# Show report
cat verification_report.txt
```

**Success criteria:**
- ✅ 3 checkpoint directories exist (or fewer, depending on save_total_limit)
- ✅ global_step increases across checkpoints (e.g., 100, 200, 300)
- ✅ Each checkpoint contains: adapter_model.safetensors, optimizer.pt, scheduler.pt, trainer_state.json
- ✅ No errors in daemon logs
- ✅ All test files were deleted from inbox after successful training

If all checks pass: **Continuous training is working correctly!** 🎉

---

## Next Steps After Verification

Once verified:

1. **Remove test data:** Clean up test batches and checkpoints if desired
2. **Start real training:** Drop real training data in inbox/
3. **Monitor daily:** Check snapshots at 3 AM, consolidation runs correctly
4. **Long-term validation:** After a week, verify global_step continues increasing correctly

For ongoing monitoring, see CLAUDE.md for daily operations and maintenance.
